"""Resumable, controlled cross-context activation-patching pilot.

Saved layer k means output of decoder block k (1-based, before final norm).
Patch only exact matching token IDs. No full-vocabulary logits or activations
are saved. Default pilot: six examples per (long-correct, answer) stratum.
This selected cohort is not a population accuracy estimate.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from extract_matched import read_pair
from jevprobe.evaluate import atomic_json, digest, run_lock, source_digest
from jevprobe.hooks import find_decoder_layers
from jevprobe.models import load_model
from jevprobe.readout import forward_score, letter_ids

LAYERS = (16, 17, 18, 19, 20, 24)
GROUPS = {"decision": (0,), "query_entities": (4, 5), "evidence_entities": (2, 3)}


def replace_positions(output, positions, values):
    tensor = output[0] if isinstance(output, tuple) else output
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError("patching requires a batch of one")
    if len(set(positions)) != len(positions) or any(p < 0 or p >= tensor.shape[1] for p in positions):
        raise ValueError("invalid patch positions")
    if tuple(values.shape) != (len(positions), tensor.shape[-1]):
        raise ValueError("patch vector shape mismatch")
    patched = tensor.clone()
    patched[0, positions] = values.to(device=tensor.device, dtype=tensor.dtype)
    return (patched, *output[1:]) if isinstance(output, tuple) else patched


@torch.inference_mode()
def patched_score(model, input_ids, ids, layer, positions, values):
    layers = find_decoder_layers(model)
    if not 1 <= layer <= len(layers):
        raise ValueError("patch layer must be a 1-based decoder block")
    called = 0

    def patch(_module, _inputs, output):
        nonlocal called
        called += 1
        return replace_positions(output, positions, values)

    handle = layers[layer - 1].register_forward_hook(patch)
    try:
        score = forward_score(model, input_ids, ids)
    finally:
        handle.remove()
    if called != 1:
        raise RuntimeError(f"patch hook ran {called} times, expected once")
    return score


def choose_cohort(catalog, per_stratum, seed):
    slots = sorted({s for group in GROUPS.values() for s in group})
    eligible = [c for c in catalog if all(c["tokens"][0][s] == c["tokens"][1][s] and c["tokens"][0][s] is not None for s in slots)]
    chosen = []
    for correct in (False, True):
        for answer in (False, True):
            pool = [c for c in eligible if c["long_correct"] == correct and c["answer"] == answer]
            if len(pool) < per_stratum:
                raise ValueError("insufficient token-matched examples in a stratum")
            random.Random(f"patch-{seed}-{correct}-{answer}").shuffle(pool)
            chosen.extend(pool[:per_stratum])
    return chosen, len(eligible)


def make_jobs(chosen, catalog, seed, final_layer):
    by_uid = {c["uid"]: c for c in catalog}
    jobs = []

    def add(target, kind, layer=0, group="none", donor=None):
        donor = donor or target["uid"]
        job = {"id": f"{target['uid']}|{kind}|{layer}|{group}", "uid": target["uid"],
               "kind": kind, "layer": layer, "group": group, "donor_uid": donor}
        if kind != "baseline":
            context = 1 if kind == "self" else 0
            for slot in GROUPS[group]:
                if by_uid[donor]["tokens"][context][slot] != target["tokens"][1][slot]:
                    raise ValueError("patch would replace a different token ID")
        jobs.append(job)

    sentinels = [next(c for c in chosen if c["long_correct"] == correct) for correct in (False, True)]
    # Run implementation controls first, before spending time on the experimental grid.
    for c in sentinels:
        add(c, "baseline")
        add(c, "positive", final_layer, "decision")
        for layer in LAYERS:
            for group in GROUPS:
                add(c, "self", layer, group)
    for c in chosen:
        if c not in sentinels:
            add(c, "baseline")
            add(c, "positive", final_layer, "decision")
        donors = [d for d in catalog if d["uid"] != c["uid"] and d["answer"] == c["answer"]
                  and (d["expected_scores"][0] > 0) == d["answer"]
                  and d["tokens"][0][0] == c["tokens"][1][0]]
        if not donors:
            raise ValueError("no matched-label decision donor")
        donor = random.Random(f"donor-{seed}-{c['uid']}").choice(donors)
        for layer in LAYERS:
            for group in GROUPS:
                add(c, "same_question", layer, group)
            add(c, "unrelated_same_answer", layer, "decision", donor["uid"])
    return jobs


def prepare(run, per_stratum, seed):
    activation_manifest = json.loads((run / "manifest.json").read_text())
    activation_hash = digest(json.dumps(activation_manifest, sort_keys=True).encode())
    if digest(Path(__file__).with_name("extract_matched.py").read_bytes()) != activation_manifest["script_sha256"]:
        raise ValueError("extractor changed since activations were saved")
    if source_digest() != activation_manifest["source_sha256"]:
        raise ValueError("core code changed since extraction")
    for package, expected in activation_manifest["behavior_config"]["versions"].items():
        if version(package) != expected:
            raise ValueError(f"package version changed: {package}")
    catalog = []
    for i, case in enumerate(activation_manifest["cases"]):
        with np.load(run / "pairs" / f"{i:04d}.npz", allow_pickle=False) as z:
            meta, pos = json.loads(z["metadata"].item()), z["positions"]
        if meta["case"] != case or meta["manifest_sha256"] != activation_hash:
            raise ValueError("saved case provenance mismatch")
        tokens = [[meta["input_ids"][context][p] if p >= 0 else None for p in pos[context]] for context in range(2)]
        catalog.append({**case, "index": i, "tokens": tokens})
    chosen, eligible = choose_cohort(catalog, per_stratum, seed)
    jobs = make_jobs(chosen, catalog, seed, activation_manifest["shape_per_pair"][1] - 1)
    plan = {"schema": 1, "activation_manifest_sha256": activation_hash,
            "script_sha256": digest(Path(__file__).read_bytes()), "seed": seed, "per_stratum": per_stratum,
            "eligible_questions": eligible, "cohort": chosen, "jobs": jobs,
            "layers": list(LAYERS), "groups": {k: list(v) for k, v in GROUPS.items()},
            "model": activation_manifest["model"], "revision": activation_manifest["revision"],
            "device": activation_manifest["device"], "dtype": activation_manifest["dtype"],
            "scope": "exploratory pilot selected by final correctness and token matching; not population accuracy",
            "controls": "self at every layer/group on two sentinels; final-decision positives on all cases; matched-label unrelated donors at decision only"}
    return plan, activation_manifest, catalog


def read_results(path, jobs, plan_hash):
    if not path.exists():
        return {}
    expected, rows = {j["id"]: j for j in jobs}, {}
    with path.open("r+b") as f:
        while True:
            start = f.tell()
            raw = f.readline()
            if not raw:
                break
            if not raw.endswith(b"\n"):
                f.truncate(start)
                break
            row = json.loads(raw)
            key = row["job"]["id"]
            if key in rows or row["job"] != expected.get(key) or row["plan_sha256"] != plan_hash or not math.isfinite(row["score"]):
                raise ValueError("invalid/duplicate saved patch result")
            rows[key] = row
    return rows


def check_control(job, score, target, baseline):
    if job["kind"] not in ("baseline", "self", "positive"):
        return
    expected = target["expected_scores"][0 if job["kind"] == "positive" else 1]
    if job["kind"] == "self":
        expected = baseline
    if not math.isclose(score, expected, abs_tol=.05, rel_tol=.01) or (score > 0) != (expected > 0):
        raise ValueError(f"control failed: {job['id']}: score={score}, expected={expected}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", type=Path, default=Path("runs/activations-matched"))
    ap.add_argument("--out", type=Path, default=Path("runs/patch-pilot"))
    ap.add_argument("--per-stratum", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-new-jobs", type=int)
    args = ap.parse_args(argv)
    if args.per_stratum < 1 or (args.max_new_jobs is not None and args.max_new_jobs < 1):
        ap.error("counts must be positive")
    plan, activation_manifest, catalog = prepare(args.activations, args.per_stratum, args.seed)
    print(json.dumps({"questions": len(plan["cohort"]), "eligible": plan["eligible_questions"],
                      "forward_passes": len(plan["jobs"]), "job_counts": dict(Counter(j["kind"] for j in plan["jobs"]))}, indent=2), flush=True)
    if args.dry_run:
        return
    plan_hash = digest(json.dumps(plan, sort_keys=True).encode())
    by_uid = {c["uid"]: c for c in catalog}
    with run_lock(args.out):
        plan_path = args.out / "plan.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError("patch plan changed; use a new output directory")
        elif (args.out / "results.jsonl").exists():
            raise ValueError("refusing to adopt results without a plan")
        else:
            atomic_json(plan_path, plan)
        rows = read_results(args.out / "results.jsonl", plan["jobs"], plan_hash)
        for row in rows.values():
            job = row["job"]
            baseline_row = rows.get(f"{job['uid']}|baseline|0|none")
            check_control(job, row["score"], by_uid[job["uid"]], baseline_row["score"] if baseline_row else None)
        pending = [j for j in plan["jobs"] if j["id"] not in rows]

        def status(state):
            atomic_json(args.out / "status.json", {"state": state, "completed": len(rows), "total": len(plan["jobs"]),
                                                    "updated_at": datetime.now(timezone.utc).isoformat()})

        @lru_cache(maxsize=8)
        def saved(uid):
            case = by_uid[uid]
            return read_pair(args.activations / "pairs" / f"{case['index']:04d}.npz",
                             activation_manifest["cases"][case["index"]], activation_manifest["shape_per_pair"], plan["activation_manifest_sha256"])

        status("running")
        try:
            if pending:
                if plan["device"] == "mps" and not torch.backends.mps.is_available():
                    raise RuntimeError("Apple GPU unavailable in this process")
                print(f"Loading {plan['model']}; {len(rows)} results already saved", flush=True)
                model, tok = load_model(plan["model"], plan["device"], getattr(torch, plan["dtype"]), revision=plan["revision"])
                ids = letter_ids(tok)
                started = time.monotonic()
                with (args.out / "results.jsonl").open("a") as f:
                    for i, job in enumerate(pending[:args.max_new_jobs], 1):
                        target = by_uid[job["uid"]]
                        _, _, positions, meta = saved(job["uid"])
                        inputs = torch.tensor([meta["input_ids"][1]], device=plan["device"])
                        if job["kind"] == "baseline":
                            score = forward_score(model, inputs, ids)
                            baseline = score
                        else:
                            baseline = rows[f"{job['uid']}|baseline|0|none"]["score"]
                            slots = list(GROUPS[job["group"]])
                            donor_states, _, _, _ = saved(job["donor_uid"])
                            source_context = 1 if job["kind"] == "self" else 0
                            vectors = torch.from_numpy(donor_states[source_context, job["layer"], slots].copy())
                            score = patched_score(model, inputs, ids, job["layer"], positions[1, slots].tolist(), vectors)
                        check_control(job, score, target, baseline)
                        row = {"job": job, "plan_sha256": plan_hash, "score": score,
                               "answer": target["answer"], "long_correct": target["long_correct"],
                               "correct": (score > 0) == target["answer"],
                               "signed_score_change": (1 if target["answer"] else -1) * (score - baseline)}
                        f.write(json.dumps(row, allow_nan=False) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                        rows[job["id"]] = row
                        status("running")
                        if i == 1 or i % 10 == 0:
                            print(f"{len(rows)}/{len(plan['jobs'])} passes saved; {job['kind']} layer={job['layer']} {job['group']}; "
                                  f"{(time.monotonic()-started)/i:.2f} s/pass", flush=True)
                del model
            status("complete" if len(rows) == len(plan["jobs"]) else "partial")
            print(f"Finished: {len(rows)}/{len(plan['jobs'])} passes saved", flush=True)
        except BaseException:
            status("interrupted")
            raise


if __name__ == "__main__":
    main()
