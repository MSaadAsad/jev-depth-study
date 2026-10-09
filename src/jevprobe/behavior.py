"""E1: accuracy (threshold 0) and AUROC (threshold-free) per hop distance, with bootstrap CIs."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_ci(values, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for d in sorted({r["d"] for r in rows}):
        sub = [r for r in rows if r["d"] == d]
        correct = [(r["score"] > 0) == r["answer"] for r in sub]
        acc, lo, hi = bootstrap_ci(correct)
        y, scores = np.array([r["answer"] for r in sub]), np.array([r["score"] for r in sub])
        auroc, auc_lo, auc_hi = auroc_ci(y, scores)
        out.append({"d": d, "n": len(sub), "acc": acc, "acc_lo": lo, "acc_hi": hi,
                    "auroc": auroc, "auroc_lo": auc_lo, "auroc_hi": auc_hi})
    return out


def auroc_ci(y, scores, n_boot: int = 1000, seed: int = 0):
    """Ordinary example bootstrap; omit resamples containing only one class."""
    y, scores = np.asarray(y), np.asarray(scores)
    if len(np.unique(y)) < 2:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            values.append(roc_auc_score(y[idx], scores[idx]))
    return float(roc_auc_score(y, scores)), float(np.quantile(values, .025)), float(np.quantile(values, .975))


def write_summary(rows: list[dict], run: Path) -> None:
    if not rows:
        raise ValueError("no completed examples to summarize")
    table = summarize(rows)
    with open(run / "behavior.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    for r in table:
        print(f"d={r['d']}  n={r['n']:4d}  acc={r['acc']:.3f} [{r['acc_lo']:.3f}, {r['acc_hi']:.3f}]  "
              f"auroc={r['auroc']:.3f} [{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}]", flush=True)


def main(argv=None) -> None:
    run = Path((argv or sys.argv[1:])[0])
    rows = [json.loads(line) for line in open(run / "meta.jsonl")]
    write_summary(rows, run)


if __name__ == "__main__":
    main()
