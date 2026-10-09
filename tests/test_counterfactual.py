import random
from collections import Counter

import pytest

from jevprobe.data.counterfactual import build_pairs, make_counterfactual
from jevprobe.data.transitive import sample_example, valid_ranks
from jevprobe.prompts import render


@pytest.mark.parametrize("d", range(1, 7))
@pytest.mark.parametrize("answer", [True, False])
def test_counterfactual_flips_answer_with_four_edits(d, answer):
    rng, made = random.Random(d), 0
    for seed in range(20):
        ex = sample_example(d, answer, "train", random.Random(seed), "u")
        cf = make_counterfactual(ex, rng)
        if cf is None:
            continue
        made += 1
        assert cf.answer == (not ex.answer)
        assert (cf.rank_x < cf.rank_y) == cf.answer  # internally consistent
        assert cf.rank_x == ex.rank_x and valid_ranks(cf.rank_x, cf.rank_y)
        assert cf.d == abs(cf.rank_x - cf.rank_y)
        assert cf.x == ex.x and cf.y == ex.y
        assert [p.template_id for p in cf.premises] == [p.template_id for p in ex.premises]
        assert sum(a != b for a, b in zip(ex.premises, cf.premises)) == 4
        q = lambda e: render(e).text.split('"criterion": ')[1]  # question + options, unchanged
        assert q(cf) == q(ex)
    assert made > 0


def test_valid_ranks():
    assert valid_ranks(7, 13) and not valid_ranks(6, 12) and not valid_ranks(17, 11)


def test_build_pairs_meets_quota_balanced_per_answer():
    pairs = build_pairs(20, "test", seed=0)
    counts = Counter((clean.d, clean.answer) for clean, _ in pairs)
    assert counts == {(d, a): 10 for d in range(1, 7) for a in (True, False)}
    assert len({clean.uid for clean, _ in pairs}) == len(pairs)
    assert all(cf.uid == clean.uid + "-cf" for clean, cf in pairs)


def test_build_pairs_respects_alignment_predicate():
    same_len = lambda clean, cf: len(clean.y) == len(clean.chain[cf.rank_y])  # swap partner's name length
    pairs = build_pairs(10, "test", seed=0, aligned=same_len)
    assert len(pairs) == 60 and all(same_len(c, f) for c, f in pairs)
