"""Validate and summarize the paired depth × context-load run."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from jevprobe.evaluate import atomic_json, digest

MODELS = ("jevk5", "qwen35-4b")
DEPTHS = (1, 2, 3, 4)
DOSES = (0, 4, 12, 24)


def wilson(k: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0., float(mid - half)), min(1., float(mid + half))


def paired_ci(values, seed: int, n_boot: int = 10000) -> dict:
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return {"estimate": float(v.mean()),
            "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/context-factorial")
    ap.add_argument("--runs", default="runs/context-factorial")
    ap.add_argument("--out", default="reports/context-factorial")
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args(argv)
    data, runs, out = Path(args.data), Path(args.runs), Path(args.out)
    design = json.loads((data / "design.json").read_text())
    dataset = Path(design["dataset_path"])
    if digest(dataset.read_bytes()) != design["dataset_sha256"]:
        raise ValueError("dataset hash differs from the frozen design")
    layout = {r["uid"]: r for r in map(json.loads, (data / "layouts.jsonl").read_text().splitlines())}
    if len(layout) != design["items_per_model"]:
        raise ValueError("layout manifest has missing or duplicate ids")
    expected = set(layout)
    scored = {}
    revisions = {}
    for model in MODELS:
        run = runs / model
        manifest = json.loads((run / "manifest.json").read_text())
        config = manifest["config"]
        if config["data_sha256"] != design["dataset_sha256"]:
            raise ValueError(f"{model}: run used a different dataset")
        rows = [json.loads(line) for line in (run / "meta.jsonl").read_text().splitlines()]
        by_uid = {r["uid"]: r for r in rows}
        if len(rows) != len(by_uid) or set(by_uid) != expected:
            raise ValueError(f"{model}: incomplete, duplicate, or unexpected records")
        for uid, row in by_uid.items():
            if row["prompt_sha256"] != layout[uid]["prompt_sha256"]:
                raise ValueError(f"{model}: prompt hash mismatch for {uid}")
        scored[model] = by_uid
        revisions[model] = config["revision"]

    diagnostic = json.loads((data / "format-diagnostic.json").read_text())
    diagnostic_uids = set(diagnostic["uids"])
    if len(diagnostic_uids) != 100 or not diagnostic_uids <= expected:
        raise ValueError("format diagnostic must name 100 unique factorial items")
    conditions = []
    correct = {m: {} for m in MODELS}
    cells = {m: {} for m in MODELS}
    for model in MODELS:
        for d in DEPTHS:
            for q in DOSES:
                subset = [layout[uid] for uid in expected
                          if layout[uid]["d"] == d and layout[uid]["n_distractors"] == q]
                subset.sort(key=lambda r: r["base_uid"])
                by_base = {}
                scores, labels, tokens = [], [], []
                for item in subset:
                    row = scored[model][item["uid"]]
                    ok = (row["score"] > 0) == row["answer"]
                    by_base[item["base_uid"]] = int(ok)
                    scores.append(row["score"])
                    labels.append(row["answer"])
                    tokens.append(row["n_tokens"])
                if len(by_base) != 300 or sum(labels) != 150:
                    raise ValueError(f"unbalanced or incomplete cell: {model}, d={d}, q={q}")
                graph_ids = {r["graph_id"] for r in subset}
                if len(graph_ids) != 300:
                    raise ValueError(f"expected one unique graph per question in d={d}")
                k, n = sum(by_base.values()), len(by_base)
                lo, hi = wilson(k, n)
                row = {"model": model, "depth": d, "off_path_premises": q,
                       "total_premises": d + q, "n": n, "correct": k,
                       "accuracy": k / n, "wilson95": [lo, hi],
                       "auroc": float(roc_auc_score(labels, scores)),
                       "mean_answer_margin": float(np.mean(scores)),
                       "mean_tokens": float(np.mean(tokens))}
                conditions.append(row)
                cells[model][d, q] = by_base
                for uid, value in by_base.items():
                    correct[model][d, q, uid] = value

    context_effects, model_interactions, model_gaps = [], [], []
    for d in DEPTHS:
        base_ids = sorted(cells["jevk5"][d, 0])
        for q in DOSES[1:]:
            model_deltas = {}
            item_deltas = {}
            for model in MODELS:
                delta = np.array([cells[model][d, q][uid] - cells[model][d, 0][uid]
                                  for uid in base_ids], dtype=float)
                model_deltas[model] = paired_ci(delta, 1000 + d * 10 + q,
                                                args.bootstrap)
                item_deltas[model] = delta
                context_effects.append({"model": model, "depth": d, "contrast": f"{q}-minus-0",
                                        **model_deltas[model],
                                        "rescued": int((delta > 0).sum()),
                                        "regressed": int((delta < 0).sum())})
            interaction = item_deltas["jevk5"] - item_deltas["qwen35-4b"]
            model_interactions.append({"depth": d, "contrast": f"{q}-minus-0",
                                       "meaning": "JevK5 context effect minus Qwen3.5-4B context effect",
                                       **paired_ci(interaction, 5000 + d * 10 + q, args.bootstrap)})
        for q in DOSES:
            gap = np.array([cells["jevk5"][d, q][uid] - cells["qwen35-4b"][d, q][uid]
                            for uid in base_ids], dtype=float)
            model_gaps.append({"depth": d, "off_path_premises": q,
                               "meaning": "JevK5 accuracy minus Qwen3.5-4B accuracy on identical prompts",
                               **paired_ci(gap, 9000 + d * 10 + q, args.bootstrap)})

    qwen_diag = [scored["qwen35-4b"][uid] for uid in diagnostic_uids]
    diag_correct = sum((r["score"] > 0) == r["answer"] for r in qwen_diag)
    diag_lo, diag_hi = wilson(diag_correct, len(qwen_diag))
    report = {"design": design, "model_revisions": revisions,
              "bootstrap": {"resamples": args.bootstrap, "unit": "unique source graph/base question; paired across context doses and models"},
              "qwen_ab_readout_diagnostic": {
                  "purpose": diagnostic["purpose"], "n": len(qwen_diag), "correct": diag_correct,
                  "accuracy": diag_correct / len(qwen_diag), "wilson95": [diag_lo, diag_hi],
                  "interpretation_limit": "This uses simple direct relation questions with the same prompt and A/B readout; low accuracy would flag format or direct-retrieval difficulty, not isolate which one."},
              "conditions": conditions, "paired_context_effects": context_effects,
              "difference_in_context_effect_by_model": model_interactions,
              "paired_model_accuracy_gaps": model_gaps}
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "summary.json", report)
    with (out / "conditions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "depth", "off_path_premises", "total_premises",
                                            "n", "correct", "accuracy", "wilson95", "auroc",
                                            "mean_answer_margin", "mean_tokens"])
        w.writeheader()
        w.writerows(conditions)
    print("Context effects (accuracy change; paired 95% bootstrap CI):")
    for r in context_effects:
        print(f"{r['model']:10s} d={r['depth']} {r['contrast']:10s} {r['estimate']:+.3f} {r['ci95']}")
    print("\nBetween-model differences in context effect (JevK5 minus Qwen):")
    for r in model_interactions:
        print(f"d={r['depth']} {r['contrast']:10s} {r['estimate']:+.3f} {r['ci95']}")


if __name__ == "__main__":
    main()
