import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

scripts = Path(__file__).parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("analyze_layers", scripts / "analyze_layers.py")
analysis = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(scripts))
try:
    spec.loader.exec_module(analysis)
finally:
    sys.path.remove(str(scripts))


def test_bootstrap_auc_matches_explicit_resampling_including_ties():
    y = np.array([False, True, False, True, True, False])
    scores = np.array([0., 0., -2., 3., 1., 1.])
    weights = analysis.bootstrap_weights(len(y), n_boot=100, seed=7)
    actual = analysis.bootstrap_auc(y, scores, weights)
    for got, w in zip(actual, weights):
        idx = np.repeat(np.arange(len(y)), w)
        if len(np.unique(y[idx])) < 2:
            assert np.isnan(got)
        else:
            assert got == pytest.approx(roc_auc_score(y[idx], scores[idx]), abs=1e-12)


def test_class_balance_removes_constant_letter_bias():
    y = np.array([True] * 41 + [False] * 34)
    scores = np.full(len(y), 3.)
    assert np.mean(scores * np.where(y, 1, -1)) != 0
    assert analysis.balanced_margin(y, scores) == 0
    weights = analysis.bootstrap_weights(len(y), n_boot=100)
    np.testing.assert_allclose(analysis.balanced_margin(y, scores, weights), 0)


def test_paired_bootstrap_identical_conditions_have_zero_difference():
    weights = analysis.bootstrap_weights(6, n_boot=100)
    correct = np.array([1, 0, 1, 0, 1, 1])
    samples = weights @ correct / 6
    assert analysis.ci(samples - samples) == [0., 0.]
