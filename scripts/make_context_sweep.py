"""Create nested context subsets for the same one-hop questions; never select by correctness."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import random

from jevprobe.data.transitive import load_jsonl, save_jsonl
from jevprobe.evaluate import atomic_json, digest
from jevprobe.prompts import render

COUNTS = (1, 5, 13, 25)


def context_variant(ex, n_premises: int, seed: int = 0):
    if ex.d != 1:
        raise ValueError("context sweep requires one-hop examples")
    if not 1 <= n_premises <= len(ex.premises):
        raise ValueError("invalid premise count")
    targets = [i for i, p in enumerate(ex.premises) if {p.greater, p.lesser} == {ex.x, ex.y}]
    if len(targets) != 1:
        raise ValueError("expected exactly one answer-giving premise")
    target = targets[0]
    others = [i for i in range(len(ex.premises)) if i != target]
    random.Random(f"context-{seed}-{ex.uid}").shuffle(others)
    chosen = sorted([target, *others[:n_premises - 1]])
    return replace(ex, premises=[ex.premises[i] for i in chosen]), chosen, chosen.index(target)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-run", default="runs/e1-jevk5-v03")
    ap.add_argument("--out", default="data/context-sweep")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    source = Path(args.source_run)
    manifest = json.loads((source / "manifest.json").read_text())
    data = Path(manifest["data_path"])
    if digest(data.read_bytes()) != manifest["config"]["data_sha256"]:
        raise ValueError("source dataset no longer matches the completed run")
    examples = sorted([e for e in load_jsonl(data) if e.d == 1], key=lambda e: e.uid)
    scores = {r["uid"]: r for r in map(json.loads, (source / "meta.jsonl").read_text().splitlines()) if r["d"] == 1}
    if len(examples) != 300 or {e.uid for e in examples} != scores.keys():
        raise ValueError("expected all 300 completed one-hop examples")
    for ex in examples:
        if digest(render(ex).text.encode()) != scores[ex.uid]["prompt_sha256"]:
            raise ValueError(f"original prompt changed: {ex.uid}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    design = {
        "source_run": str(source.resolve()), "source_data_sha256": digest(data.read_bytes()),
        "generator_sha256": digest(Path(__file__).read_bytes()), "seed": args.seed,
        "counts": COUNTS, "n_questions": len(examples),
        "selection": "all 300 original one-hop examples; nested random subsets per uid; original relative premise order retained",
        "primary_comparison": "paired accuracy difference, 1 premise minus 25 premises; 1000 bootstrap samples, seed 0",
        "limitations": "premise count, entity count, token count and relevant-fact position change together",
        "datasets": {},
    }
    records = []
    for count in COUNTS:
        variants = []
        for ex in examples:
            variant, indices, target_position = context_variant(ex, count, args.seed)
            variants.append(variant)
            records.append({"uid": ex.uid, "n_premises": count, "original_premise_indices": indices,
                            "relevant_premise_index": target_position, "prompt_sha256": digest(render(variant).text.encode())})
        path = out / f"p{count}.jsonl"
        save_jsonl(variants, path)
        design["datasets"][str(count)] = {"path": str(path.resolve()), "sha256": digest(path.read_bytes())}
    (out / "layouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    atomic_json(out / "design.json", design)
    print(json.dumps(design, indent=2))


if __name__ == "__main__":
    main()
