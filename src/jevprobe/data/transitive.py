"""Transitive-chain examples: a 24-entity main chain plus a 3-entity distractor chain."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from jevprobe.data.vocab import PREMISE_TEMPLATES, RELATIONS, name_pool

N_MAIN = 24
N_DISTRACT = 3
N_PREMISES = (N_MAIN - 1) + (N_DISTRACT - 1)
MAX_D = 6


@dataclass(frozen=True)
class Premise:
    greater: str
    lesser: str
    template_id: int


@dataclass(frozen=True)
class Example:
    uid: str
    split: str
    d: int
    answer: bool
    rel: str
    inv: str
    chain: list[str]  # rank 0 = greatest
    distractors: list[str]  # index 0 = greatest
    x: str
    y: str
    premises: list[Premise]  # in displayed (shuffled) order

    @property
    def rank_x(self) -> int:
        return self.chain.index(self.x)

    @property
    def rank_y(self) -> int:
        return self.chain.index(self.y)

    @property
    def bridges(self) -> list[str]:
        """Entities strictly between X and Y, ordered starting from X."""
        i, j = self.rank_x, self.rank_y
        return self.chain[i + 1 : j] if i < j else self.chain[j + 1 : i][::-1]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Example":
        return cls(**{**d, "premises": [Premise(**p) for p in d["premises"]]})


def valid_ranks(rank_x: int, rank_y: int) -> bool:
    """Both query entities lie >= d+1 hops from either main-chain endpoint, d = |rank_x - rank_y| >= 1."""
    d = abs(rank_x - rank_y)
    return d >= 1 and min(min(r, N_MAIN - 1 - r) for r in (rank_x, rank_y)) >= d + 1


def sample_example(d: int, answer: bool, split: str, rng: random.Random, uid: str) -> Example:
    if not 1 <= d <= MAX_D:
        raise ValueError(f"d must be in 1..{MAX_D}, got {d}")
    names = rng.sample(name_pool(split), N_MAIN + N_DISTRACT)
    chain, distractors = names[:N_MAIN], names[N_MAIN:]
    # Endpoints are identifiable by their single mention, so both query entities sit >= d+1 hops
    # from either end: any endpoint-anchored shortcut then costs more hops than the path itself.
    i = rng.randint(d + 1, N_MAIN - 2 - 2 * d)
    x, y = (chain[i], chain[i + d]) if answer else (chain[i + d], chain[i])
    relation = rng.choice(RELATIONS[split])
    pairs = [(chain[k], chain[k + 1]) for k in range(N_MAIN - 1)]
    pairs += [(distractors[k], distractors[k + 1]) for k in range(N_DISTRACT - 1)]
    rng.shuffle(pairs)
    n_templates = len(PREMISE_TEMPLATES[split])
    premises = [Premise(g, l, rng.randrange(n_templates)) for g, l in pairs]
    return Example(uid, split, d, answer, relation.rel, relation.inv, chain, distractors, x, y, premises)


def build_dataset(n_per_d: int, split: str, seed: int, ds=range(1, MAX_D + 1)) -> list[Example]:
    if n_per_d % 2:
        raise ValueError("n_per_d must be even for exact answer balance")
    rng = random.Random(f"{seed}-{split}")
    out = []
    for d in ds:
        answers = [True] * (n_per_d // 2) + [False] * (n_per_d // 2)
        rng.shuffle(answers)
        out += [sample_example(d, a, split, rng, f"{split}-d{d}-{k}") for k, a in enumerate(answers)]
    rng.shuffle(out)
    return out


def save_jsonl(examples: list[Example], path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")


def load_jsonl(path) -> list[Example]:
    with open(path) as f:
        return [Example.from_dict(json.loads(line)) for line in f]
