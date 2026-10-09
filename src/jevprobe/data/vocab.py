"""Split-specific vocabularies. Train and test never share names, relations or templates."""
from __future__ import annotations

import random
from dataclasses import dataclass

_CONSONANTS = "bdfgklmnprstvz"
_VOWELS = "aeiou"
_CODAS = ["", "n", "r", "l", "s"]


@dataclass(frozen=True)
class Relation:
    rel: str  # comparative for greater-than ("taller")
    inv: str  # comparative for less-than ("shorter")


RELATIONS: dict[str, list[Relation]] = {
    "train": [
        Relation("taller", "shorter"),
        Relation("older", "younger"),
        Relation("heavier", "lighter"),
        Relation("faster", "slower"),
    ],
    "test": [
        Relation("richer", "poorer"),
        Relation("stronger", "weaker"),
        Relation("louder", "quieter"),
    ],
}

# {g} = greater entity, {l} = lesser entity. Forward and inverse phrasings are mixed per premise.
PREMISE_TEMPLATES: dict[str, list[str]] = {
    "train": [
        "{g} is {rel} than {l}.",
        "{l} is {inv} than {g}.",
        "Compared to {l}, {g} is {rel}.",
        "Compared to {g}, {l} is {inv}.",
    ],
    "test": [
        "{g} is known to be {rel} than {l}.",
        "{l} is known to be {inv} than {g}.",
    ],
}

QUESTION_TEMPLATE = "Is {x} {rel} than {y}?"

_POOL_SIZES = {"train": 400, "test": 200}


def _all_names() -> list[str]:
    sylls = [c + v for c in _CONSONANTS for v in _VOWELS]
    names = sorted({(a + b + coda).capitalize() for a in sylls for b in sylls for coda in _CODAS})
    random.Random(0).shuffle(names)
    return names


_NAMES = _all_names()


def name_pool(split: str) -> list[str]:
    if split == "train":
        return _NAMES[: _POOL_SIZES["train"]]
    if split == "test":
        return _NAMES[_POOL_SIZES["train"] : _POOL_SIZES["train"] + _POOL_SIZES["test"]]
    raise ValueError(f"unknown split {split!r}")
