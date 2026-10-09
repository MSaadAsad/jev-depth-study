"""Paired analysis of completed nested-context conditions on fixed one-hop questions."""
import argparse
import csv
import json
import math
from pathlib import Path

from jevprobe.behavior import bootstrap_ci
from jevprobe.evaluate import atomic_json, digest


def wilson_interval(correct, n):
    """Supplement bootstrap intervals, which degenerate for perfect sample accuracy."""
    z = 1.959963984540054
    p = correct / n
    denom = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0, mid - half), min(1, mid + half)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/context-sweep")
    ap.add_argument("--runs", default="runs/context-sweep")
    args = ap.parse_args(argv)
    data, out = Path(args.data), Path(args.runs)
    design = json.loads((data / "design.json").read_text())
    original = Path(design["source_run"])
    source_config = json.loads((original / "manifest.json").read_text())["config"]
    layouts = {(r["uid"], r["n_premises"]): r for r in map(json.loads, (data / "layouts.jsonl").read_text().splitlines())}
    all_rows, summaries = {}, []
    for count in design["counts"]:
        run = original if count == 25 else out / f"p{count}"
        config = json.loads((run / "manifest.json").read_text())["config"]
        for field in ("revision", "source_sha256", "prompt_sha256", "dtype", "device", "versions", "readout"):
            if config[field] != source_config[field]:
                raise ValueError(f"incompatible provenance for p{count}: {field}")
        dataset = design["datasets"][str(count)]
        if digest(Path(dataset["path"]).read_bytes()) != dataset["sha256"]:
            raise ValueError(f"changed dataset for p{count}")
        expected_data_hash = design["source_data_sha256"] if count == 25 else dataset["sha256"]
        if config["data_sha256"] != expected_data_hash:
            raise ValueError(f"run data mismatch for p{count}")
        rows = [r for r in map(json.loads, (run / "meta.jsonl").read_text().splitlines()) if r["d"] == 1]
        by_uid = {r["uid"]: r for r in rows}
        if len(rows) != len(by_uid) or len(rows) != design["n_questions"]:
            raise ValueError(f"incomplete or duplicate records for p{count}")
        for r in rows:
            if r["prompt_sha256"] != layouts[r["uid"], count]["prompt_sha256"]:
                raise ValueError(f"prompt mismatch for {r['uid']} p{count}")
        all_rows[count] = by_uid
        with (run / "behavior.csv").open() as f:
            summary = next({k: float(v) for k, v in row.items()} for row in csv.DictReader(f) if row["d"] == "1")
        correct = sum((r["score"] > 0) == r["answer"] for r in rows)
        if summary["n"] != len(rows) or abs(summary["acc"] - correct / len(rows)) > 1e-12:
            raise ValueError(f"stale summary for p{count}")
        lo, hi = wilson_interval(correct, len(rows))
        summaries.append({"premises": count, "n": len(rows), "correct": correct,
                          **{k: v for k, v in summary.items() if k not in ("d", "n")},
                          "acc_wilson_lo": lo, "acc_wilson_hi": hi,
                          "mean_tokens": sum(r["n_tokens"] for r in rows) / len(rows)})
    full = all_rows[25]
    comparisons = {}
    for count, rows in all_rows.items():
        if rows.keys() != full.keys():
            raise ValueError("conditions contain different questions")
        differences = []
        for uid in sorted(full):
            a, b = rows[uid], full[uid]
            if a["answer"] != b["answer"]:
                raise ValueError("condition changed the answer")
            differences.append(int((a["score"] > 0) == a["answer"]) - int((b["score"] > 0) == b["answer"]))
        mean, lo, hi = bootstrap_ci(differences)
        comparisons[str(count)] = {"accuracy_difference_vs_25": mean, "ci95": [lo, hi],
                                   "corrected": differences.count(1), "regressed": differences.count(-1)}
    report = {
        "model": source_config["model"], "revision": source_config["revision"],
        "design": design, "conditions": summaries, "paired_comparisons_vs_25": comparisons,
        "primary_comparison": comparisons["1"], "bootstrap": {"resamples": 1000, "seed": 0},
        "limitations": [design["limitations"], "One fixed subset seed; subset-selection variability is not estimated.",
                        "This measures context sensitivity on one-hop questions, not a multi-hop reasoning ceiling."],
    }
    atomic_json(out / "summary.json", report)
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for r in summaries:
        print(f"{r['premises']:2d} premises: {r['correct']}/{r['n']} = {r['acc']:.1%}; AUROC={r['auroc']:.3f}")
    print("Paired 1-minus-25 accuracy:", comparisons["1"])


if __name__ == "__main__":
    main()
