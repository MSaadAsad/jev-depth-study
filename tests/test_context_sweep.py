import importlib.util
from pathlib import Path

import pytest

from jevprobe.data.transitive import build_dataset
from jevprobe.prompts import render

spec = importlib.util.spec_from_file_location("make_context_sweep", Path(__file__).parents[1] / "scripts/make_context_sweep.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
context_variant = module.context_variant


def test_nested_contexts_preserve_question_answer_fact_and_relative_order():
    for ex in build_dataset(24, "test", 3, ds=[1]):
        previous = set()
        for n in module.COUNTS:
            variant, indices, target = context_variant(ex, n)
            assert len(variant.premises) == len(indices) == n
            assert previous <= set(indices)
            assert indices == sorted(indices)
            p = variant.premises[target]
            assert {p.greater, p.lesser} == {ex.x, ex.y}
            assert (variant.x, variant.y, variant.answer, variant.d, variant.uid) == (ex.x, ex.y, ex.answer, 1, ex.uid)
            assert render(variant).text.split('"criterion": ')[1] == render(ex).text.split('"criterion": ')[1]
            assert context_variant(ex, n) == (variant, indices, target)
            previous = set(indices)
        assert render(variant).text == render(ex).text


def test_rejects_non_one_hop_and_invalid_counts():
    ex = build_dataset(2, "test", 1, ds=[2])[0]
    with pytest.raises(ValueError, match="one-hop"):
        context_variant(ex, 1)
    ex = build_dataset(2, "test", 1, ds=[1])[0]
    for n in (0, 26):
        with pytest.raises(ValueError, match="count"):
            context_variant(ex, n)
