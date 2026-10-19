"""Export only the completed headline results used by the article explorer."""
import csv
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "blog-draft-v2" / "ResearchExplorerData.ts"


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()
    data = {"depth": [], "context": [], "patching": []}
    for model, path in (("JevK5 v0.3", ROOT / "runs/e1-jevk5-v03/behavior.csv"),
                        ("Qwen3.5-4B", ROOT / "runs/e1-qwen35-4b/behavior.csv")):
        for row in rows(path):
            data["depth"].append({
                "model": model, "depth": int(row["d"]), "n": int(row["n"]),
                "accuracy": float(row["acc"]), "accuracy_lo": float(row["acc_lo"]),
                "accuracy_hi": float(row["acc_hi"]),
            })

    sweep = json.loads((ROOT / "reports/context-factorial/summary.json").read_text())
    for row in sweep["conditions"]:
        data["context"].append({
            "model": "JevK5 v0.3" if row['model'] == 'jevk5' else 'Qwen3.5-4B',
            "depth": row['depth'], "off_path": row['off_path_premises'],
            "premises": row["total_premises"], "n": row["n"],
            "accuracy": row["accuracy"], "accuracy_lo": row["wilson95"][0],
            "accuracy_hi": row["wilson95"][1],
        })
    patch = json.loads((ROOT / "runs/layer-remainder/combined-summary.json").read_text())
    for key, summary in patch.items():
        assert summary['pairs'] == 160 and summary['layers'] == list(range(1, 33))
        assert summary['repeated_baseline_max_error'] <= 1e-5
        for row in summary["effects"]:
            ci = row["mean_signed_change_ci95"]
            data["patching"].append({"model": "JevK5 v0.3" if key == 'jev' else 'Qwen3.5-4B',
                                 "d": row["d"], "layer": row["layer"], "group": row["group"],
                                 "n": row["n"], "mean_signed_logit_change": row["mean_signed_logit_change"],
                                 "low": ci[0], "high": ci[1]})

    serial = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("// Generated from completed aggregate summaries; prompt-level outputs excluded.\n"
                   f"export const researchData: any = JSON.parse({json.dumps(serial, ensure_ascii=False)});\n")
    print(f"Wrote {args.out} ({len(serial):,} data characters)")


if __name__ == "__main__":
    main()
