import importlib.util
import json
from pathlib import Path

from jevprobe.data.transitive import build_dataset, load_jsonl
from jevprobe.prompts import render

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("make_context_factorial", ROOT / "scripts/make_context_factorial.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_factorial_keeps_query_path_and_nests_unrelated_facts():
    for d in (1, 2, 3, 4):
        ex = build_dataset(2, "test", 43, ds=[d])[0]
        prev = set()
        for q in module.DISTRACTOR_COUNTS:
            variant, layout = module.context_variant(ex, q)
            present = {frozenset((p.greater, p.lesser)) for p in variant.premises}
            assert module.path_edges(ex) <= present
            assert len(variant.premises) == d + q
            assert layout["n_distractors"] == q
            assert prev <= (present - module.path_edges(ex))
            assert len(layout["relevant_premise_positions"]) == d
            assert render(variant).text == render(module.context_variant(ex, q)[0]).text
            prev = present - module.path_edges(ex)


def test_one_hop_prompts_match_existing_context_sweep_exactly():
    old_spec = importlib.util.spec_from_file_location("make_context_sweep", ROOT / "scripts/make_context_sweep.py")
    old = importlib.util.module_from_spec(old_spec)
    old_spec.loader.exec_module(old)
    examples = build_dataset(2, "test", 7, ds=[1])
    for ex in examples:
        for q in module.DISTRACTOR_COUNTS:
            new, _ = module.context_variant(ex, q)
            previous, _, _ = old.context_variant(ex, q + 1, seed=0)
            assert render(new).text == render(previous).text


def test_generated_dataset_has_complete_balanced_factorial(tmp_path):
    from jevprobe.data.transitive import build_dataset, save_jsonl

    source = tmp_path / "source.jsonl"
    save_jsonl(build_dataset(300, "test", 17, ds=range(1, 5)), source)
    out = tmp_path / "out"
    module.main(["--source", str(source), "--out", str(out)])
    data = load_jsonl(out / "test.jsonl")
    assert len(data) == 4800
    layout = [json.loads(line) for line in (out / "layouts.jsonl").read_text().splitlines()]
    assert len(layout) == len({r["uid"] for r in layout}) == 4800
    for d in range(1, 5):
        for q in module.DISTRACTOR_COUNTS:
            cell = [e for e in data if e.d == d and e.uid.endswith(f"-x{q:02d}")]
            assert len(cell) == 300
            assert sum(e.answer for e in cell) == 150
