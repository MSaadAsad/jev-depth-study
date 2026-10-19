"""Build a paired depth × off-path-premise evaluation from the pinned E1 cohort."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import random
from pathlib import Path

from jevprobe.data.transitive import Example, Premise, load_jsonl, save_jsonl
from jevprobe.data.vocab import name_pool
from jevprobe.evaluate import atomic_json, digest
from jevprobe.prompts import render

DISTRACTOR_COUNTS = (0, 4, 12, 24)
DEPTHS = (1, 2, 3, 4)


def path_edges(ex: Example) -> set[frozenset[str]]:
    lo, hi = sorted((ex.rank_x, ex.rank_y))
    return {frozenset((ex.chain[i], ex.chain[i + 1])) for i in range(lo, hi)}


def graph_id(ex: Example) -> str:
    payload = json.dumps({"main_chain": ex.chain, "separate_chain": ex.distractors},
                         separators=(",", ":")).encode()
    return digest(payload)


def context_variant(ex: Example, n_distractors: int, seed: int = 0) -> tuple[Example, dict]:
    """Keep the complete query path; add a nested sample of off-path premise records."""
    if ex.d not in DEPTHS or n_distractors not in DISTRACTOR_COUNTS:
        raise ValueError("unsupported depth or distractor count")
    path = path_edges(ex)
    relevant = [(i, p) for i, p in enumerate(ex.premises)
                if frozenset((p.greater, p.lesser)) in path]
    if len(relevant) != ex.d:
        raise ValueError(f"{ex.uid}: expected {ex.d} path premises, found {len(relevant)}")
    irrelevant = [(i, p) for i, p in enumerate(ex.premises)
                  if frozenset((p.greater, p.lesser)) not in path]
    needed = 24 - len(irrelevant)
    # This seed/order reproduces the existing JevK5 one-hop context sweep exactly.
    rng = random.Random(f"context-{seed}-{ex.uid}")
    extras: list[tuple[int, Premise]] = []
    if needed > 0:
        used = set(ex.chain) | set(ex.distractors)
        used |= {name for p in ex.premises for name in (p.greater, p.lesser)}
        available = [n for n in name_pool("test") if n not in used]
        if len(available) < 2 * needed:
            raise ValueError("not enough held-out names to create disconnected fillers")
        extra_names = rng.sample(available, 2 * needed)
        for j in range(needed):
            a, b = extra_names[2 * j:2 * j + 2]
            greater, lesser = (a, b) if rng.random() < .5 else (b, a)
            template_id = rng.randrange(2)
            # Synthetic filler indices follow all original premise positions.
            extras.append((len(ex.premises) + j, Premise(greater, lesser, template_id)))

    candidates = [*irrelevant, *extras]
    rng.shuffle(candidates)
    chosen = candidates[:n_distractors]
    active = [*relevant, *chosen]
    # Keep source order and the same nested distractor set at every dose.
    active.sort(key=lambda item: item[0])
    premises = [p for _, p in active]
    variant = replace(ex, uid=f"{ex.uid}-x{n_distractors:02d}", premises=premises)
    rel_positions = [i for i, p in enumerate(premises)
                     if frozenset((p.greater, p.lesser)) in path]
    main_edges = {frozenset((ex.chain[i], ex.chain[i + 1]))
                  for i in range(len(ex.chain) - 1)}
    separate_edges = {frozenset((ex.distractors[i], ex.distractors[i + 1]))
                      for i in range(len(ex.distractors) - 1)}
    categories = []
    category_counts = {"off_path_main_chain": 0, "separate_distractor_chain": 0,
                       "disconnected_single_fact": 0}
    for _, p in active:
        edge = frozenset((p.greater, p.lesser))
        if edge in path:
            categories.append("query_path")
        elif edge in main_edges:
            categories.append("off_path_main_chain")
            category_counts["off_path_main_chain"] += 1
        elif edge in separate_edges:
            categories.append("separate_distractor_chain")
            category_counts["separate_distractor_chain"] += 1
        else:
            categories.append("disconnected_single_fact")
            category_counts["disconnected_single_fact"] += 1
    layout = {
        "uid": variant.uid, "base_uid": ex.uid, "d": ex.d,
        "n_distractors": n_distractors, "n_premises": len(premises),
        "answer": ex.answer,
        "relevant_premise_positions": rel_positions,
        "premise_categories": categories, **category_counts,
        "graph_id": graph_id(ex),
        "prompt_sha256": digest(render(variant).text.encode()),
    }
    return variant, layout


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="data/e1/test.jsonl")
    ap.add_argument("--out", default="data/context-factorial")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    source = Path(args.source)
    examples = [e for e in load_jsonl(source) if e.d in DEPTHS]
    counts = {d: sum(e.d == d for e in examples) for d in DEPTHS}
    if counts != {d: 300 for d in DEPTHS}:
        raise ValueError(f"expected 300 E1 items at each depth 1–4, found {counts}")
    if any(sum(e.answer for e in examples if e.d == d) != 150 for d in DEPTHS):
        raise ValueError("source cohort is not exactly answer-balanced")
    graph_counts = {d: len({graph_id(e) for e in examples if e.d == d}) for d in DEPTHS}
    if graph_counts != {d: 300 for d in DEPTHS}:
        raise ValueError(f"expected 300 distinct source graphs per depth, got {graph_counts}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    variants, layouts = [], []
    for ex in examples:
        for n in DISTRACTOR_COUNTS:
            variant, row = context_variant(ex, n, args.seed)
            variants.append(variant)
            layouts.append(row)
    clean_one_hop = [r for r in layouts if r["d"] == 1 and r["n_distractors"] == 0]
    diagnostic_rng = random.Random(f"format-diagnostic-{args.seed}")
    diagnostic_uids = []
    for answer in (False, True):
        candidates = [r["uid"] for r in clean_one_hop if r["answer"] is answer]
        diagnostic_uids.extend(diagnostic_rng.sample(candidates, 50))
    diagnostic_uids.sort()
    variants.sort(key=lambda e: e.uid)
    layouts.sort(key=lambda r: r["uid"])
    dataset = out / "test.jsonl"
    save_jsonl(variants, dataset)
    (out / "layouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in layouts))
    (out / "format-diagnostic.json").write_text(json.dumps({
        "purpose": "diagnose Qwen A/B readout on direct one-premise questions",
        "subset_of": "d=1, zero off-path premises in the primary factorial",
        "selection": "50 true and 50 false questions selected before model evaluation",
        "uids": diagnostic_uids,
    }, indent=2) + "\n")
    design = {
        "design": "paired 4 × 4 depth-by-off-path-premise factorial",
        "source_path": str(source.resolve()), "source_sha256": digest(source.read_bytes()),
        "dataset_path": str(dataset.resolve()), "dataset_sha256": digest(dataset.read_bytes()),
        "seed": args.seed, "depths": DEPTHS, "n_off_path_premises": DISTRACTOR_COUNTS,
        "items_per_cell": 300, "items_per_model": len(variants),
        "answers_per_cell": {"true": 150, "false": 150},
        "cohort": "the pinned E1 test examples, held-out names/relations/templates; identical queries across context doses and models",
        "context_definition": "all shortest-path premises are retained; context dose adds off-path main-chain edges, separate distractor-chain edges, and disconnected single-fact fillers only where needed to reach 24",
        "distractor_types": "each layout records counts and premise-level labels for original off-path main-chain edges, separate distractor-chain edges, and disconnected single-fact fillers",
        "graph_cluster_check": "300 unique source graph signatures per depth; each base question is its own graph-level bootstrap cluster",
        "format_diagnostic": "100 preselected clean one-hop prompts (50 per answer), reported as a Qwen A/B-readout diagnostic using primary-run scores",
        "limitations": [
            "This measures interference from off-path factual premises, not malicious instructions or contradictory evidence.",
            "The number of premises and prompt tokens increase together; token count is reported, so this does not isolate semantic distraction from raw context length.",
            "Premise position varies with the selected subset; the path-fact positions are recorded for diagnosis.",
            "One fixed synthetic task and prompt template; conclusions do not establish a general property of either model.",
        ],
        "primary_estimands": [
            "paired accuracy difference, 24 minus 0 off-path premises, within each model and depth",
            "difference in that paired context effect between JevK5 and Qwen3.5-4B",
            "accuracy change across hop depth at each fixed distractor dose",
        ],
    }
    atomic_json(out / "design.json", design)
    print(json.dumps(design, indent=2))


if __name__ == "__main__":
    main()
