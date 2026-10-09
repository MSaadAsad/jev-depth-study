"""Write train/test/pairs datasets plus the per-d shortcut report; refuse to finish if the gate fails."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevprobe.data.shortcuts import gate, shortcut_report
from jevprobe.data.transitive import build_dataset, save_jsonl


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-per-d", type=int, default=1000)
    ap.add_argument("--test-per-d", type=int, default=300)
    ap.add_argument("--pairs-per-d", type=int, default=200)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B", help="for token-aligned counterfactual pairs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    train = build_dataset(args.train_per_d, "train", args.seed)
    test = build_dataset(args.test_per_d, "test", args.seed)
    save_jsonl(train, out / "train.jsonl")
    save_jsonl(test, out / "test.jsonl")

    if args.pairs_per_d:
        from transformers import AutoTokenizer

        from jevprobe.data.counterfactual import build_pairs
        from jevprobe.tokens import token_aligned

        aligned = token_aligned(AutoTokenizer.from_pretrained(args.tokenizer))
        with open(out / "pairs.jsonl", "w") as f:
            for clean, cf in build_pairs(args.pairs_per_d, "test", args.seed, aligned):
                f.write(json.dumps({"clean": clean.to_dict(), "cf": cf.to_dict()}) + "\n")

    report = shortcut_report(test)
    failures = gate(report, args.test_per_d)
    (out / "shortcuts.json").write_text(json.dumps({"test": report, "gate_failures": failures}, indent=2))
    print(json.dumps({m: {d: round(v, 3) for d, v in vals.items()} for m, vals in report.items()}, indent=1))
    if failures:
        raise SystemExit("shortcut gate failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
