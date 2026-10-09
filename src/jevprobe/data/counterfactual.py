"""Counterfactuals: swap Y's name with an entity on the other side of X.

X keeps its rank, Y moves past X so the answer flips, and the question text is unchanged.
Y and the swap partner are separated by X, so they share no premise: exactly 4 premises change.
The counterfactual must itself satisfy the endpoint rule (valid_ranks), and, when an `aligned`
predicate is given (e.g. tokens.token_aligned), every token position outside the swapped names.
"""
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import replace

from jevprobe.data.transitive import MAX_D, N_MAIN, Example, Premise, sample_example, valid_ranks

Aligned = Callable[[Example, Example], bool]


def make_counterfactual(ex: Example, rng: random.Random, aligned: Aligned | None = None) -> Example | None:
    i = ex.rank_x
    side = range(0, i) if ex.answer else range(i + 1, N_MAIN)
    candidates = [m for m in side if valid_ranks(i, m)]
    rng.shuffle(candidates)
    for m in candidates:
        a, b = ex.y, ex.chain[m]
        swap = {a: b, b: a}
        s = lambda n: swap.get(n, n)
        cf = replace(
            ex,
            uid=ex.uid + "-cf",
            answer=not ex.answer,
            d=abs(i - m),
            chain=[s(n) for n in ex.chain],
            premises=[Premise(s(p.greater), s(p.lesser), p.template_id) for p in ex.premises],
        )
        if aligned is None or aligned(ex, cf):
            return cf
    return None


def build_pairs(n_per_d: int, split: str, seed: int, aligned: Aligned | None = None,
                ds=range(1, MAX_D + 1), max_tries: int = 100) -> list[tuple[Example, Example]]:
    """Exactly n_per_d pairs per clean d, half per answer, drawn from a dedicated example pool."""
    if n_per_d % 2:
        raise ValueError("n_per_d must be even for exact answer balance")
    rng = random.Random(f"{seed}-{split}-pairs")
    pairs = []
    for d in ds:
        for answer in (True, False):
            got, tries = 0, 0
            while got < n_per_d // 2:
                tries += 1
                if tries > max_tries * n_per_d:
                    raise RuntimeError(f"could not build {n_per_d // 2} pairs for d={d} answer={answer}")
                ex = sample_example(d, answer, split, rng, f"{split}-pair-d{d}-{answer:d}-{tries}")
                cf = make_counterfactual(ex, rng, aligned)
                if cf is not None:
                    pairs.append((ex, cf))
                    got += 1
    return pairs
