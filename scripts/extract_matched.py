"""Selective, resumable JevK5 activation extraction for matched 1/25-premise prompts.

Default: all 300 matched questions, including errors and successful controls.
No full-token activations or vocabulary logits are retained. Each atomic .npz
contains a whole pair: states [context, layer, slot, hidden], answer-letter
projections [context, layer, slot], positions [context, slot], and JSON metadata.
Contexts are [1, 25]; layer 0 is embeddings, subsequent layers are pre-final-norm
decoder outputs. Absent slots have position -1 and zero data (not observations).

The final norm and A-minus-B head are also applied to intermediate states as an
exploratory logit lens. This is not a trained probe or a causal intervention.

Examples:
  .venv/bin/python scripts/extract_matched.py --dry-run
  .venv/bin/python scripts/extract_matched.py
  .venv/bin/python scripts/extract_matched.py --max-pairs 2
  .venv/bin/python scripts/extract_matched.py --analyze-only
Rerun the identical command to resume; completed pairs are validated and skipped.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from transformers import AutoConfig

from jevprobe.data.transitive import load_jsonl
from jevprobe.evaluate import atomic_json, digest, run_lock, source_digest
from jevprobe.extract import check_storable
from jevprobe.hooks import find_decoder_layers, find_final_norm, find_text_decoder
from jevprobe.models import load_model, parse_model_ref
from jevprobe.prompts import render
from jevprobe.readout import letter_ids
from jevprobe.tokens import last_token_index, positions

SLOTS = ("decision", "relevant_premise_end", "relevant_x", "relevant_y",
         "question_x", "question_y", "other_x", "other_y")
CONTEXTS = (1, 25)
RESERVE_BYTES = 2 * 1024**3


def selected_positions(ex, rendered, offsets):
    """Individual mention states, never averaged; missing mentions are explicit."""
    targets = [i for i, p in enumerate(ex.premises) if {p.greater, p.lesser} == {ex.x, ex.y}]
    if ex.d != 1 or len(targets) != 1:
        raise ValueError("expected a one-hop example with one answer-giving premise")
    target = targets[0]
    start, end = rendered.premise_spans[target]
    relevant, other = [], []
    for name in (ex.x, ex.y):
        inside = [s for s in rendered.entity_spans[name] if start <= s[0] and s[1] <= end]
        outside = [s for s in rendered.entity_spans[name] if s not in inside]
        if len(inside) != 1 or len(outside) > 1:
            raise ValueError("unexpected entity mention layout")
        relevant.append(last_token_index(offsets, *inside[0]))
        other.append(last_token_index(offsets, *outside[0]) if outside else -1)
    pos = positions(rendered, offsets, [])
    question = [last_token_index(offsets, *rendered.question_spans[k]) for k in ("x", "y")]
    return [pos.decision, pos.premises[target], *relevant, *question, *other]


@torch.inference_mode()
def capture_selected(model, input_ids, indices):
    """Clone only selected token states while each layer runs; avoid full LM logits."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("requires a batch of one")
    if any(i < -1 or i >= input_ids.shape[1] for i in indices):
        raise ValueError("token index out of range")
    idx = torch.tensor([max(i, 0) for i in indices], device=input_ids.device)
    valid = torch.tensor([i >= 0 for i in indices], device=input_ids.device)
    layers, store = find_decoder_layers(model), {}

    def save(k):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            store[k] = tensor[0].index_select(0, idx).detach().clone().masked_fill(~valid[:, None], 0)
        return hook

    handles = [model.get_input_embeddings().register_forward_hook(save(0))]
    handles += [layer.register_forward_hook(save(k + 1)) for k, layer in enumerate(layers)]
    try:
        find_text_decoder(model)(input_ids=input_ids, use_cache=False, return_dict=True,
                                 output_hidden_states=False, output_attentions=False)
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack([store[k] for k in range(len(layers) + 1)]).float()


@torch.inference_mode()
def project_letters(model, states, ids):
    head = model.get_output_embeddings()
    normed = find_final_norm(model)(states.to(device=head.weight.device, dtype=head.weight.dtype)).float()
    a, b = head.weight[list(ids)].float()
    scores = normed @ (a - b)
    if head.bias is not None:
        scores += head.bias[ids[0]].float() - head.bias[ids[1]].float()
    if not torch.isfinite(scores).all():
        raise ValueError("non-finite answer-letter projection")
    return scores


def atomic_pair(path, states, projections, token_positions, metadata):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez(f, states=states, projections=projections, positions=token_positions,
                 metadata=np.array(json.dumps(metadata, allow_nan=False)))
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def read_pair(path, case, shape, manifest_hash):
    """Fully read arrays (including ZIP CRCs), validate, then allow resume/analysis."""
    with np.load(path, allow_pickle=False) as saved:
        states, lens, pos = saved["states"], saved["projections"], saved["positions"]
        meta = json.loads(saved["metadata"].item())
    if meta["case"] != case or meta["manifest_sha256"] != manifest_hash:
        raise ValueError(f"pair provenance mismatch: {path}")
    if states.shape != tuple(shape) or states.dtype != np.float16:
        raise ValueError(f"invalid activation shape/dtype: {path}")
    if lens.shape != tuple(shape[:-1]) or lens.dtype != np.float32 or pos.shape != (2, len(SLOTS)) or pos.dtype != np.int32:
        raise ValueError(f"invalid projection/position shape: {path}")
    if not np.isfinite(states).all() or not np.isfinite(lens).all():
        raise ValueError(f"non-finite saved data: {path}")
    if np.any(pos[:, :6] < 0) or np.any(pos < -1):
        raise ValueError(f"missing required token positions: {path}")
    for context in range(2):
        if not math.isclose(float(lens[context, -1, 0]), case["expected_scores"][context], abs_tol=.05, rel_tol=.01):
            raise ValueError(f"saved readout mismatch: {path}")
        token_ids = meta["input_ids"][context]
        if not token_ids or np.any(pos[context] >= len(token_ids)) or pos[context, 0] != len(token_ids) - 1:
            raise ValueError(f"invalid saved token positions: {path}")
        missing = pos[context] < 0
        if np.any(states[context][:, missing]) or np.any(lens[context][:, missing]):
            raise ValueError(f"missing slots must be zero-filled: {path}")
    return states, lens, pos, meta


def load_sources(short_run, long_run):
    examples, records, configs = [], [], []
    for count, run in zip(CONTEXTS, (short_run, long_run)):
        manifest = json.loads((run / "manifest.json").read_text())
        config = manifest["config"]
        path = Path(manifest["data_path"])
        if digest(path.read_bytes()) != config["data_sha256"]:
            raise ValueError(f"source data changed: {path}")
        exs = {e.uid: e for e in load_jsonl(path) if e.d == 1}
        rows = [r for r in map(json.loads, (run / "meta.jsonl").read_text().splitlines()) if r["d"] == 1]
        recs = {r["uid"]: r for r in rows}
        if len(rows) != 300 or len(recs) != len(rows) or exs.keys() != recs.keys():
            raise ValueError("expected 300 unique completed one-hop examples per context")
        for uid, ex in exs.items():
            if len(ex.premises) != count or digest(render(ex).text.encode()) != recs[uid]["prompt_sha256"]:
                raise ValueError(f"source prompt changed: {uid}, context {count}")
            if ex.answer != recs[uid]["answer"] or not math.isfinite(recs[uid]["score"]):
                raise ValueError(f"invalid source score/answer: {uid}")
        examples.append(exs)
        records.append(recs)
        configs.append(config)
    for key in ("revision", "source_sha256", "prompt_sha256", "device", "dtype", "versions", "readout"):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"source runs differ in {key}")
    if examples[0].keys() != examples[1].keys():
        raise ValueError("source runs contain different questions")
    if source_digest() != configs[0]["source_sha256"]:
        raise ValueError("core code changed since behavior runs; use the matching code revision")
    import jevk5.prompt
    if digest(Path(jevk5.prompt.__file__).read_bytes()) != configs[0]["prompt_sha256"]:
        raise ValueError("JevK5 prompt implementation changed")
    for package, expected in configs[0]["versions"].items():
        if version(package) != expected:
            raise ValueError(f"package version changed: {package}")
    cases = []
    for uid in sorted(examples[0]):
        a, b = examples[0][uid], examples[1][uid]
        if (a.x, a.y, a.answer) != (b.x, b.y, b.answer):
            raise ValueError(f"unmatched query: {uid}")
        scores = [r[uid]["score"] for r in records]
        cases.append({"uid": uid, "answer": a.answer, "x": a.x, "y": a.y,
                      "expected_scores": scores, "long_correct": (scores[1] > 0) == a.answer,
                      "prompt_sha256": [r[uid]["prompt_sha256"] for r in records]})
    return configs[0], examples, cases


def disk_budget(out, n_pairs, shape):
    parent = out
    while not parent.exists():
        parent = parent.parent
    # 1 MiB per pair for projections, metadata, ZIP headers and temporary-file slack.
    raw = n_pairs * math.prod(shape) * np.dtype(np.float16).itemsize
    needed = raw + n_pairs * 1024**2 + RESERVE_BYTES
    free = shutil.disk_usage(parent).free
    return {"activation_bytes": raw, "required_free_bytes": needed, "available_bytes": free}


def summarize(out, manifest, manifest_hash):
    """Small descriptive tables; no probing, emergence or causal claims."""
    shape = manifest["shape_per_pair"]
    buckets, completed, flips = {}, 0, 0
    maximum_error = 0.0
    for i, case in enumerate(manifest["cases"]):
        path = out / "pairs" / f"{i:04d}.npz"
        if not path.exists():
            continue
        states, lens, pos, meta = read_pair(path, case, shape, manifest_hash)
        completed += 1
        scores = lens[:, -1, 0]
        maximum_error = max(maximum_error, float(np.max(np.abs(scores - np.asarray(case["expected_scores"])))))
        flips += sum((float(a) > 0) != (b > 0) for a, b in zip(scores, case["expected_scores"]))
        groups = ("all", "long_correct" if case["long_correct"] else "long_wrong")
        valid = np.flatnonzero(np.all(pos >= 0, axis=0))
        for slot in valid:
            a, b = states[0, :, slot].astype(np.float32), states[1, :, slot].astype(np.float32)
            denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
            cosine = np.sum(a * b, axis=-1) / np.maximum(denom, 1e-12)
            sign = 1 if case["answer"] else -1
            for group in groups:
                for layer in range(shape[1]):
                    buckets.setdefault((group, layer, SLOTS[slot]), []).append(
                        (float(cosine[layer]), float(sign * lens[0, layer, slot]), float(sign * lens[1, layer, slot])))
    if not completed:
        raise ValueError("no completed pairs to summarize")
    fields = ["group", "layer", "slot", "n", "cosine_mean", "cosine_std", "short_signed_projection_mean", "long_signed_projection_mean"]
    with (out / "layers.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (group, layer, slot), values in sorted(buckets.items()):
            arr = np.asarray(values)
            writer.writerow(dict(zip(fields, (group, layer, slot, len(arr), arr[:, 0].mean(), arr[:, 0].std(), arr[:, 1].mean(), arr[:, 2].mean()))))
    report = {"completed_pairs": completed, "total_pairs": len(manifest["cases"]),
              "complete": completed == len(manifest["cases"]), "max_readout_difference": maximum_error,
              "prediction_disagreements_with_behavior": int(flips),
              "layer_table": "layers.csv", "storage_bytes": sum(p.stat().st_size for p in (out / "pairs").glob("*.npz")),
              "notes": ["Projections use the final norm/head on intermediate states; they are exploratory, uncalibrated, and not trained probes.",
                        "Groups are selected by final correctness; endpoint differences between groups are by construction.",
                        "States are matched by semantic token role, not absolute position. Missing slots are excluded from comparisons.",
                        "These summaries do not establish causality or a reasoning-depth ceiling."]}
    atomic_json(out / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--short-run", type=Path, default=Path("runs/context-sweep/p1"))
    ap.add_argument("--long-run", type=Path, default=Path("runs/e1-jevk5-v03"))
    ap.add_argument("--out", type=Path, default=Path("runs/activations-matched"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--max-pairs", type=int, help="maximum new pairs this invocation; omit to finish all")
    args = ap.parse_args(argv)
    if args.max_pairs is not None and args.max_pairs < 1:
        ap.error("--max-pairs must be positive")
    config, examples, cases = load_sources(args.short_run, args.long_run)
    repo, _ = parse_model_ref(config["model"])
    model_config = AutoConfig.from_pretrained(repo, revision=config["revision"], local_files_only=True).get_text_config()
    shape = (2, model_config.num_hidden_layers + 1, len(SLOTS), model_config.hidden_size)
    manifest = {"format_version": 1, "model": repo, "revision": config["revision"],
                "device": config["device"], "dtype": config["dtype"], "storage_dtype": "float16",
                "script_sha256": digest(Path(__file__).read_bytes()), "source_sha256": source_digest(),
                "behavior_config": config, "shape_per_pair": list(shape), "contexts": list(CONTEXTS), "slots": list(SLOTS),
                "layer_convention": "0=embedding; 1..L=decoder output before final norm", "cases": cases}
    manifest_hash = digest(json.dumps(manifest, sort_keys=True).encode())
    budget = disk_budget(args.out, len(cases), shape)
    print(json.dumps({"pairs": len(cases), "shape_per_pair": shape, **budget}, indent=2), flush=True)
    if args.dry_run:
        if budget["available_bytes"] < budget["required_free_bytes"]:
            raise RuntimeError("insufficient disk space")
        return
    with run_lock(args.out):
        manifest_path = args.out / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("activation manifest mismatch; use a new output directory")
        elif any((args.out / "pairs").glob("*.npz")):
            raise ValueError("refusing to adopt existing pairs without a manifest")
        else:
            atomic_json(manifest_path, manifest)
        pair_dir = args.out / "pairs"
        pair_dir.mkdir(exist_ok=True)
        pending = []
        for i, case in enumerate(cases):
            path = pair_dir / f"{i:04d}.npz"
            if path.exists():
                read_pair(path, case, shape, manifest_hash)
            else:
                pending.append((i, case))
        completed = len(cases) - len(pending)

        def status(state):
            atomic_json(args.out / "status.json", {"state": state, "completed_pairs": completed, "total_pairs": len(cases),
                                                    "updated_at": datetime.now(timezone.utc).isoformat()})

        if args.analyze_only:
            summarize(args.out, manifest, manifest_hash)
            return
        budget = disk_budget(args.out, len(pending), shape)
        if pending and budget["available_bytes"] < budget["required_free_bytes"]:
            raise RuntimeError("insufficient disk space for remaining pairs plus 2 GiB reserve")
        status("running")
        try:
            if pending:
                if config["device"] == "mps" and not torch.backends.mps.is_available():
                    raise RuntimeError("Apple GPU unavailable; run from a native terminal")
                print(f"Loading {repo}@{config['revision']}; {completed} pairs already saved", flush=True)
                model, tok = load_model(repo, config["device"], getattr(torch, config["dtype"]), revision=config["revision"])
                ids = letter_ids(tok)
                started = time.monotonic()
                for counter, (i, case) in enumerate(pending[:args.max_pairs], 1):
                    if shutil.disk_usage(args.out).free < RESERVE_BYTES + math.prod(shape) * 2 + 1024**2:
                        raise RuntimeError("disk reserve reached; completed pairs are safe")
                    arrays, projections, all_pos, all_ids, maxima = [], [], [], [], []
                    for context in range(2):
                        ex = examples[context][case["uid"]]
                        rendered = render(ex)
                        enc = tok(rendered.text, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
                        indices = selected_positions(ex, rendered, enc["offset_mapping"][0].tolist())
                        states = capture_selected(model, enc["input_ids"].to(config["device"]), indices)
                        check_storable(states)
                        lens = project_letters(model, states, ids).clone()
                        lens[:, np.flatnonzero(np.asarray(indices) < 0).tolist()] = 0
                        score = float(lens[-1, 0])
                        expected = case["expected_scores"][context]
                        if not math.isclose(score, expected, abs_tol=.05, rel_tol=.01):
                            raise ValueError(f"readout mismatch {case['uid']} context {CONTEXTS[context]}: {score} vs {expected}")
                        maxima.append(float(states.abs().max()))
                        arrays.append(states.cpu().numpy().astype(np.float16))
                        projections.append(lens.cpu().numpy())
                        all_pos.append(indices)
                        all_ids.append(enc["input_ids"][0].tolist())
                    meta = {"case": case, "manifest_sha256": manifest_hash, "input_ids": all_ids, "max_abs": maxima}
                    atomic_pair(pair_dir / f"{i:04d}.npz", np.stack(arrays), np.stack(projections), np.array(all_pos, dtype=np.int32), meta)
                    completed += 1
                    status("running")
                    if counter == 1 or counter % 10 == 0:
                        print(f"{completed}/{len(cases)} pairs saved; {(time.monotonic()-started)/counter:.2f} s/pair", flush=True)
                del model
            summarize(args.out, manifest, manifest_hash)
            status("complete" if completed == len(cases) else "partial")
        except BaseException:
            status("interrupted")
            raise


if __name__ == "__main__":
    main()
