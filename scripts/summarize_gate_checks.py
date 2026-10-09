"""Summarize completed one-hop comparisons and paired controls with provenance."""
import argparse
import csv
import json
from pathlib import Path

from jevprobe.behavior import bootstrap_ci
from jevprobe.evaluate import atomic_json, digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jev-run", default="runs/e1-jevk5-v03")
    ap.add_argument("--base-run", default="runs/e1-qwen35-4b")
    ap.add_argument("--out", default="runs/gate-checks/summary.json")
    args = ap.parse_args()
    runs, manifests, rows, metrics = {}, {}, {}, {}
    for label, name in (("jevk5", args.jev_run), ("qwen", args.base_run)):
        path = runs[label] = Path(name)
        manifests[label] = json.loads((path / "manifest.json").read_text())
        records = [json.loads(line) for line in (path / "meta.jsonl").read_text().splitlines()]
        records = [r for r in records if r["d"] == 1]
        rows[label] = {r["uid"]: r for r in records}
        assert len(rows[label]) == len(records) == 300, "both depth-1 runs must finish"
        with (path / "behavior.csv").open() as f:
            metrics[label] = next({k: float(v) for k, v in r.items()} for r in csv.DictReader(f) if r["d"] == "1")
    for field in ("data_sha256", "source_sha256", "dtype", "device", "versions", "prompt_sha256"):
        assert manifests["jevk5"]["config"][field] == manifests["qwen"]["config"][field], field
    assert rows["jevk5"].keys() == rows["qwen"].keys()
    differences = []
    for uid in sorted(rows["jevk5"]):
        a, b = rows["jevk5"][uid], rows["qwen"][uid]
        assert (a["answer"], a["prompt_sha256"]) == (b["answer"], b["prompt_sha256"])
        differences.append(int((a["score"] > 0) == a["answer"]) - int((b["score"] > 0) == b["answer"]))
    mean, lo, hi = bootstrap_ci(differences)
    diagnostics = json.loads((runs["jevk5"] / "gate_diagnostics.json").read_text())
    controls = diagnostics["controls"]
    report = {
        "scope": "one-hop gate and diagnostics only; no depth-ceiling claim",
        "gate": {"minimum_accuracy": 0.85, "jevk5_passed": metrics["jevk5"]["acc"] >= .85},
        "metrics": metrics,
        "paired_accuracy_difference_jevk5_minus_qwen": {"mean": mean, "ci95": [lo, hi], "bootstrap_resamples": 1000, "seed": 0},
        "diagnostics": {k: v for k, v in diagnostics.items() if k not in ("controls", "native_comparisons")},
        "control_counts": {
            "total": len(controls),
            "full_correct": sum((r["full_score"] > 0) == r["answer"] for r in controls),
            "single_correct": sum((r["single_premise_score"] > 0) == r["answer"] for r in controls),
            "corrected_by_removing_other_premises": sum((r["full_score"] > 0) != r["answer"] and (r["single_premise_score"] > 0) == r["answer"] for r in controls),
            "regressions": sum((r["full_score"] > 0) == r["answer"] and (r["single_premise_score"] > 0) != r["answer"] for r in controls),
        },
        "provenance": {label: manifest["config"] for label, manifest in manifests.items()},
        "diagnostic_script_sha256": digest(Path("scripts/check_e1_gate.py").read_bytes()),
        "limitations": [
            "Native agreement was checked on 12 examples, not the entire dataset.",
            "The 48 controls were balanced by relation, answer and phrasing and selected independently of correctness.",
            "Removing other premises changes both context length and evidence selection difficulty; these effects are not separated.",
            "The failed one-hop gate prevents a clean interpretation of deeper-hop errors as a reasoning-depth ceiling.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=2))


if __name__ == "__main__":
    main()
