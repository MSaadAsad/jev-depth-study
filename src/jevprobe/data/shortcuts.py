"""Surface-heuristic baselines, reported per hop distance. Gated ones must sit near 50% at every d."""
from __future__ import annotations

import math

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from jevprobe.data.transitive import N_MAIN, Example
from jevprobe.prompts import render

GATED = ("mention_order", "name_frequency", "bow_cv")


def _acc(preds, examples) -> float:
    return sum(p == e.answer for p, e in zip(preds, examples)) / len(examples)


def mention_order_acc(examples: list[Example]) -> float:
    """Predict yes iff X is first mentioned before Y in the state."""
    preds = []
    for e in examples:
        spans = render(e).entity_spans
        preds.append(spans[e.x][0][0] < spans[e.y][0][0])
    return _acc(preds, examples)


def name_frequency_acc(examples: list[Example]) -> float:
    """Predict yes iff X is mentioned at least as often as Y."""
    preds = []
    for e in examples:
        spans = render(e).entity_spans
        preds.append(len(spans[e.x]) >= len(spans[e.y]))
    return _acc(preds, examples)


def bow_cv_acc(examples: list[Example], folds: int = 5) -> float:
    """Within-split cross-validated bag-of-words accuracy: can surface n-grams predict the answer?"""
    clf = make_pipeline(CountVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=2000))
    scores = cross_val_score(clf, [render(e).text for e in examples], [e.answer for e in examples], cv=folds)
    return float(scores.mean())


def endpoint_cover(examples: list[Example]) -> float:
    """Fraction where X or Y is within d hops of a main-chain endpoint (an endpoint shortcut no costlier than the path)."""
    near = [min(min(r, N_MAIN - 1 - r) for r in (e.rank_x, e.rank_y)) <= e.d for e in examples]
    return sum(near) / len(examples)


def rank_x_auroc(examples: list[Example]) -> float:
    """Informational: how well X's absolute rank predicts the answer (needs >= d+1 hops to compute)."""
    return float(roc_auc_score([e.answer for e in examples], [-e.rank_x for e in examples]))


_METRICS = {
    "mention_order": mention_order_acc,
    "name_frequency": name_frequency_acc,
    "bow_cv": bow_cv_acc,
    "endpoint_cover": endpoint_cover,
    "rank_x_auroc": rank_x_auroc,
}


def shortcut_report(examples: list[Example]) -> dict[str, dict[str, float]]:
    by_d: dict[int, list[Example]] = {}
    for e in examples:
        by_d.setdefault(e.d, []).append(e)
    return {name: {str(d): fn(by_d[d]) for d in sorted(by_d)} for name, fn in _METRICS.items()}


def gate(report: dict[str, dict[str, float]], n_per_d: int) -> list[str]:
    """Failures: a gated metric more than 3 binomial SDs from 0.5, or any endpoint cover."""
    tol = 3 * math.sqrt(0.25 / n_per_d)
    fails = [f"{m} d={d}: {v:.3f}" for m in GATED for d, v in report[m].items() if abs(v - 0.5) > tol]
    fails += [f"endpoint_cover d={d}: {v:.3f}" for d, v in report["endpoint_cover"].items() if v > 0]
    return fails
