"""Streaming, layer-16/20 causal patching on token-aligned transitive pairs at d=1..6."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from jevprobe.data.counterfactual import build_pairs
from jevprobe.data.transitive import Example
from jevprobe.evaluate import atomic_json, digest, run_lock, source_digest
from jevprobe.hooks import find_decoder_layers
from jevprobe.models import load_model
from jevprobe.prompts import render
from jevprobe.readout import forward_score, letter_ids
from jevprobe.tokens import last_token_index, token_aligned
from patch_matched import patched_score

MODEL = "alibiserikbay/JevK5"
REVISION = "c4f7fdb3aeab5582336406e78d3bef11bf98833d"
LAYERS = (16, 20)
GROUPS = ("decision", "query_names", "bridge_entities", "path_premises", "distractor_premises")
KINDS = ("source", "counterfactual")


def spans_for_name(rendered, name, premise_indices):
    spans = []
    for start, end in (rendered.premise_spans[i] for i in premise_indices):
        spans.extend((a, b) for a, b in rendered.entity_spans.get(name, []) if start <= a and b <= end)
    return spans


def layout(ex, tok):
    r = render(ex)
    enc = tok(r.text, return_offsets_mapping=True, add_special_tokens=False)
    off = [tuple(x) for x in enc["offset_mapping"]]
    edges = {frozenset((ex.chain[k], ex.chain[k + 1]))
             for k in range(min(ex.rank_x, ex.rank_y), max(ex.rank_x, ex.rank_y))}
    path = [i for i, p in enumerate(ex.premises) if frozenset((p.greater, p.lesser)) in edges]
    if len(path) != ex.d:
        raise ValueError("could not map every chain edge to exactly one premise")
    distractors = [i for i, p in enumerate(ex.premises)
                   if p.greater in ex.distractors and p.lesser in ex.distractors]
    question = [last_token_index(off, *r.question_spans[k]) for k in ("x", "y")]
    q = {"decision": [len(off) - 1], "query_names": question,
         "path_premises": [last_token_index(off, *r.premise_spans[i]) for i in path],
         "distractor_premises": [last_token_index(off, *r.premise_spans[i]) for i in distractors]}
    q["bridge_entities"] = []
    for name in ex.bridges:
        spans = spans_for_name(r, name, path)
        if not spans:
            raise ValueError(f"missing bridge mention {name}")
        q["bridge_entities"].extend(last_token_index(off, *span) for span in spans)
    return {"input_ids": enc["input_ids"], "offsets": off, "groups": q,
            "path_indices": path, "distractor_indices": distractors,
            "rendered": r, "prompt_sha256": digest(r.text.encode())}


def prepare(n_per_depth, seed, *, ds=range(1, 7), fixed_depth=False):
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    aligned = token_aligned(tok)
    pairs = build_pairs(n_per_depth, "test", seed, ds=ds,
                        aligned=lambda a, b: (not fixed_depth or a.d == b.d) and aligned(a, b))
    cases = []
    for clean, cf in pairs:
        a, b = layout(clean, tok), layout(cf, tok)
        if len(a["input_ids"]) != len(b["input_ids"]):
            raise ValueError("counterfactual pair is not sequence aligned")
        for k in ("decision", "query_names", "distractor_premises"):
            if len(a["groups"][k]) != len(b["groups"][k]):
                raise ValueError(f"aligned patch positions differ for {k}")
        # Bridge states are taken from the source path, then mapped to all mentions
        # of the same entity in the counterfactual. The entity token IDs must match.
        bpos = []
        for name in clean.bridges:
            source_spans = spans_for_name(a["rendered"], name, a["path_indices"])
            target_spans = b["rendered"].entity_spans.get(name, [])
            if len(source_spans) != len(target_spans):
                raise ValueError("bridge mentions do not map one-to-one across prompts")
            source_pos = [last_token_index(a["offsets"], *s) for s in source_spans]
            target_pos = [last_token_index(b["offsets"], *s) for s in target_spans]
            if any(a["input_ids"][i] != b["input_ids"][j] for i, j in zip(source_pos, target_pos)):
                raise ValueError("bridge name token mismatch")
            bpos.extend(target_pos)
        a["groups"]["bridge_entities"] = [last_token_index(a["offsets"], *s)
            for name in clean.bridges for s in spans_for_name(a["rendered"], name, a["path_indices"])]
        b["groups"]["bridge_entities"] = bpos
        # Transfer the source path premise states at the same premise slots in the
        # counterfactual, even when its own shortest path has a different depth.
        b["groups"]["path_premises"] = [last_token_index(b["offsets"], *b["rendered"].premise_spans[i])
                                               for i in a["path_indices"]]
        for g in GROUPS:
            if len(a["groups"][g]) != len(b["groups"][g]):
                raise ValueError(f"donor/target patch count differs: {g}")
            for i, j in zip(a["groups"][g], b["groups"][g]):
                if a["input_ids"][i] != b["input_ids"][j]:
                    raise ValueError(f"patch token mismatch at {g}")
        cases.append({"uid": clean.uid, "d": clean.d, "answer": clean.answer,
                      "counterfactual_d": cf.d, "source": clean.to_dict(),
                      "counterfactual": cf.to_dict(), "prompt_hashes": [a["prompt_sha256"], b["prompt_sha256"]],
                      "positions": {k: [a["groups"][k], b["groups"][k]] for k in GROUPS}})
    return {"schema": 1, "model": MODEL, "revision": REVISION, "layers": list(LAYERS),
            "fixed_counterfactual_depth": fixed_depth,
            "groups": list(GROUPS), "n_per_depth": n_per_depth, "seed": seed,
            "cases": cases, "source_sha256": source_digest(),
            "script_sha256": digest(Path(__file__).read_bytes()),
            "versions": {p: version(p) for p in ("torch", "transformers", "numpy", "jevk5")},
            "design": "Aligned counterfactuals, 200 pairs/depth by default, balanced by source answer. Patch source states into opposite-answer prompt at layers 16 and 20; positions: decision, query names, source bridge mentions, source path premise ends, distractor premise ends.",
            "limitations": "Counterfactual intervention is not a unique algorithm identification. 95% pointwise stratified bootstrap intervals; no multiplicity adjustment."}


def capture(model, ids, input_ids, positions):
    blocks = find_decoder_layers(model)
    want = {l: [] for l in LAYERS}
    unique = sorted(set(positions))
    idx = torch.tensor(unique, device=input_ids.device)
    handles = []
    for layer in LAYERS:
        def hook(_m, _i, out, layer=layer):
            x = out[0] if isinstance(out, tuple) else out
            want[layer] = x[0].index_select(0, idx).detach().clone()
        handles.append(blocks[layer - 1].register_forward_hook(hook))
    try:
        score = forward_score(model, input_ids, ids)
    finally:
        for h in handles:
            h.remove()
    if any(v.shape[0] != len(unique) for v in want.values()):
        raise RuntimeError("selected-state capture failed")
    return score, unique, want


def _state_at(pos, unique, states):
    return states[[unique.index(p) for p in pos]]


def validate_row(row, case, plan_hash):
    if row["uid"] != case["uid"] or row["plan_sha256"] != plan_hash:
        raise ValueError("pair provenance mismatch")
    for k in ("source_score", "cf_score"):
        if not math.isfinite(row[k]):
            raise ValueError("nonfinite baseline score")
    expected = {f"{l}/{g}" for l in LAYERS for g in GROUPS if case["positions"][g][0]}
    if set(row["patch_scores"]) != expected or not all(math.isfinite(s) for s in row["patch_scores"].values()):
        raise ValueError("missing/nonfinite patch result")
    if case["self_control"]:
        expected_self = {f"{l}/{g}/{s}" for l in LAYERS for g in GROUPS
                         if case["positions"][g][0] for s in KINDS}
        if set(row["self_scores"]) != expected_self:
            raise ValueError("missing self-control results")
        for key, score in row["self_scores"].items():
            l, g, s = key.split("/")
            baseline = row["source_score"] if s == "source" else row["cf_score"]
            if not math.isclose(score, baseline, abs_tol=.05, rel_tol=.01) or (score > 0) != (baseline > 0):
                raise ValueError("self patch failed")
    elif row["self_scores"]:
        raise ValueError("unexpected self-control results")


def read_results(path, cases, plan_hash):
    rows = {}
    if not path.exists():
        return rows
    by = {c["uid"]: c for c in cases}
    with path.open("r+b") as f:
        while True:
            start = f.tell(); raw = f.readline()
            if not raw: break
            if not raw.endswith(b"\n"):
                f.truncate(start); break
            row = json.loads(raw)
            if row["uid"] in rows: raise ValueError("duplicate result")
            validate_row(row, by[row["uid"]], plan_hash)
            rows[row["uid"]] = row
    return rows


def summarize(rows, cases):
    by = {r["uid"]: r for r in rows}
    out = []
    rng = np.random.default_rng(0)
    for d in range(1, 7):
        subset = [c for c in cases if c["d"] == d]
        ans = np.array([c["answer"] for c in subset])
        weights = np.concatenate([rng.choice(np.flatnonzero(ans == a), (2000, int(sum(ans == a))), replace=True)
                                  for a in (False, True)], axis=1)
        for layer in LAYERS:
            for group in GROUPS:
                selected = [c for c in subset if c["positions"][group][0]]
                if not selected: continue
                cf = np.array([by[c["uid"]]["cf_score"] for c in selected])
                clean = np.array([by[c["uid"]]["source_score"] for c in selected])
                patched = np.array([by[c["uid"]]["patch_scores"][f"{layer}/{group}"] for c in selected])
                labels = np.array([c["answer"] for c in selected])
                n = len(selected)
                draws = np.concatenate([rng.choice(np.flatnonzero(labels == a), (2000, int(sum(labels == a))), replace=True)
                                        for a in (False, True)], axis=1)
                recovery = (patched - cf) / np.where(np.abs(clean - cf) > .05, clean - cf, np.nan)
                signed = np.where(labels, 1., -1.)
                delta = signed * (patched - cf)
                patch_source = (patched > 0) == labels
                cf_correct = (cf > 0) != labels
                to_source = patch_source & cf_correct
                to_cf = ~patch_source & ~cf_correct
                out.append({"d": d, "layer": layer, "group": group, "n": n,
                    "source_accuracy": float(np.mean((clean > 0) == labels)),
                    "counterfactual_accuracy": float(np.mean((cf > 0) != labels)),
                    "patched_source_label_rate": float(np.mean((patched > 0) == labels)),
                    "patched_counterfactual_accuracy": float(np.mean((patched > 0) != labels)),
                    "flipped_correct_cf_to_source_label": int(to_source.sum()),
                    "flipped_wrong_cf_to_cf_label": int(to_cf.sum()),
                    "mean_signed_logit_change": float(delta.mean()),
                    "mean_signed_change_ci95": np.quantile(delta[draws].mean(1), [.025, .975]).tolist(),
                    "mean_recovery_fraction": float(np.nanmean(recovery)) if np.isfinite(recovery).any() else None,
                    "recovery_n": int(np.isfinite(recovery).sum())})
    return {"pairs": len(rows), "layers": list(LAYERS), "effects": out,
            "controls": "Self patches on 12 sentinel pairs; distractor premise patches serve as negative location control.",
            "limitations": "Pointwise intervals; no multiplicity adjustment; the counterfactual swap changes four premises and can change the counterfactual hop distance."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("runs/composition-patching"))
    ap.add_argument("--n-per-depth", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--prepare-only", action="store_true", help="build and save the aligned plan without loading a model")
    ap.add_argument("--max-pairs", type=int)
    args = ap.parse_args()
    if args.n_per_depth < 2 or args.n_per_depth % 2: ap.error("n-per-depth must be positive and even")
    plan_path = args.out / "plan.json"
    if plan_path.exists() and not args.dry_run:
        plan = json.loads(plan_path.read_text())
        if (plan["n_per_depth"], plan["seed"], plan["script_sha256"], plan["source_sha256"]) != \
           (args.n_per_depth, args.seed, digest(Path(__file__).read_bytes()), source_digest()):
            raise ValueError("saved plan provenance or options differ")
    else:
        plan = prepare(args.n_per_depth, args.seed)
        # One self-patch sentinel per (depth, answer) stratum.
        sent = set()
        for c in plan["cases"]:
            key = (c["d"], c["answer"])
            c["self_control"] = key not in sent
            sent.add(key)
    plan_hash = digest(json.dumps(plan, sort_keys=True).encode())
    self_passes = sum(2 * sum(bool(c["positions"][g][0]) for g in GROUPS) * len(LAYERS)
                      for c in plan["cases"] if c["self_control"])
    interventions = sum(sum(bool(c['positions'][g][0]) for g in GROUPS) * len(LAYERS) for c in plan['cases'])
    estimated = len(plan['cases']) * 2 + interventions + self_passes
    print(json.dumps({"pairs": len(plan["cases"]), "pairs_per_depth": dict(Counter(c["d"] for c in plan["cases"])),
        "balanced_answers": True, "layers": LAYERS, "groups": GROUPS, "interventions": interventions,
        "self_patch_passes": self_passes, "forward_passes_approx": estimated,
        "self_control_pairs": sum(c['self_control'] for c in plan['cases']), "plan_sha256": plan_hash}, indent=2), flush=True)
    if args.dry_run: return
    if args.prepare_only:
        args.out.mkdir(parents=True, exist_ok=True)
        atomic_json(plan_path, plan)
        print(f"Saved aligned plan to {plan_path}", flush=True)
        return
    with run_lock(args.out):
        pp = args.out / "plan.json"
        if pp.exists() and json.loads(pp.read_text()) != plan: raise ValueError("plan changed; choose new output directory")
        if not pp.exists() and (args.out / "results.jsonl").exists(): raise ValueError("results exist without plan")
        if not pp.exists(): atomic_json(pp, plan)
        rows = read_results(args.out / "results.jsonl", plan["cases"], plan_hash)
        pending = [c for c in plan["cases"] if c["uid"] not in rows]
        def status(state):
            atomic_json(args.out / "status.json", {"state": state, "completed_pairs": len(rows),
                "total_pairs": len(plan["cases"]), "updated_at": datetime.now(timezone.utc).isoformat()})
        status("running")
        try:
            if pending:
                if not torch.backends.mps.is_available(): raise RuntimeError("Apple GPU unavailable")
                model, tok = load_model(MODEL, "mps", torch.bfloat16, revision=REVISION)
                ids = letter_ids(tok)
                with (args.out / "results.jsonl").open("a") as f:
                    t0 = time.monotonic()
                    for i, case in enumerate(pending[:args.max_pairs], 1):
                        ex = Example.from_dict(case["source"]); cf = Example.from_dict(case["counterfactual"])
                        lay = [layout(ex, tok), layout(cf, tok)]
                        inputs = [torch.tensor([x["input_ids"]], device="mps") for x in lay]
                        allpos = sorted({p for k in GROUPS for ctx in (0, 1) for p in case["positions"][k][ctx]})
                        scores, poslists, states = [], [], []
                        for j in range(2):
                            score, ps, st = capture(model, ids, inputs[j], allpos)
                            scores.append(score); poslists.append(ps); states.append(st)
                        patch_scores, self_scores = {}, {}
                        for layer in LAYERS:
                            for group in GROUPS:
                                srcpos, dstpos = case["positions"][group]
                                if not srcpos: continue
                                vectors = _state_at(srcpos, poslists[0], states[0][layer]).to(dtype=torch.bfloat16)
                                patch_scores[f"{layer}/{group}"] = patched_score(model, inputs[1], ids, layer, dstpos, vectors)
                                if case["self_control"]:
                                    for j, kind in ((0, "source"), (1, "counterfactual")):
                                        ownpos = case["positions"][group][j]
                                        if ownpos:
                                            own = _state_at(ownpos, poslists[j], states[j][layer]).to(dtype=torch.bfloat16)
                                            self_scores[f"{layer}/{group}/{kind}"] = patched_score(model, inputs[j], ids, layer, ownpos, own)
                        row = {"uid": case["uid"], "plan_sha256": plan_hash,
                               "source_score": scores[0], "cf_score": scores[1],
                               "patch_scores": patch_scores, "self_scores": self_scores}
                        validate_row(row, case, plan_hash)
                        f.write(json.dumps(row, allow_nan=False)+"\n"); f.flush(); os.fsync(f.fileno())
                        rows[case["uid"]] = row; status("running")
                        if i == 1 or i % 10 == 0:
                            print(f"{len(rows)}/{len(plan['cases'])} pairs; {2*i+sum(len(r['patch_scores']) for r in rows.values())/max(1,len(rows)):.1f} mean passes/pair; "
                                  f"{(time.monotonic()-t0)/i:.1f} s/pair", flush=True)
                del model
            if len(rows) == len(plan["cases"]):
                atomic_json(args.out / "summary.json", summarize([rows[c["uid"]] for c in plan["cases"]], plan["cases"]))
            status("complete" if len(rows) == len(plan["cases"]) else "partial")
        except BaseException:
            status("interrupted"); raise


if __name__ == "__main__": main()
