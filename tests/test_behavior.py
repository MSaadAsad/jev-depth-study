import math
import pytest
from jevprobe.behavior import auroc_ci, bootstrap_ci, summarize


def test_bootstrap_ci_brackets_mean():
    m, lo, hi = bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1])
    assert lo <= m <= hi and m == 0.75


def test_summarize_per_depth():
    rows = [{"d": 1, "answer": True, "score": 2.0}, {"d": 1, "answer": False, "score": -1.0},
            {"d": 2, "answer": True, "score": -0.5}, {"d": 2, "answer": False, "score": 0.5}]
    out = {r["d"]: r for r in summarize(rows)}
    assert out[1]["acc"] == 1.0 and out[1]["auroc"] == 1.0
    assert out[2]["acc"] == 0.0 and out[2]["auroc"] == 0.0
    assert out[1]["n"] == 2


def test_auroc_ci_perfect_separation_and_single_class():
    assert auroc_ci([False, False, True, True], [-2, -1, 1, 2]) == pytest.approx((1, 1, 1))
    assert all(math.isnan(v) for v in auroc_ci([True, True], [1, 2]))


def test_auroc_ci_tied_scores():
    assert auroc_ci([False, True] * 10, [0] * 20) == pytest.approx((.5, .5, .5))
