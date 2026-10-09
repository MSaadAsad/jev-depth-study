"""Map character spans to token indices using tokenizer offset mappings."""
from __future__ import annotations

from dataclasses import dataclass

from jevprobe.prompts import Rendered, render


def last_token_index(offsets: list[tuple[int, int]], start: int, end: int) -> int:
    """Index of the last non-empty token overlapping [start, end)."""
    hits = [t for t, (s, e) in enumerate(offsets) if e > s and s < end and e > start]
    if not hits:
        raise ValueError(f"no token overlaps span ({start}, {end})")
    return hits[-1]


@dataclass(frozen=True)
class Positions:
    decision: int
    entities: list[list[int]]  # per entity (in entity_order), one index per state mention
    premises: list[int]  # last token of each premise, in displayed order


def positions(r: Rendered, offsets: list[tuple[int, int]], entity_order: list[str]) -> Positions:
    decision = len(offsets) - 1
    if offsets[decision][1] != len(r.text):
        raise ValueError("last token does not end the prompt; tokenizer appended something")
    return Positions(
        decision=decision,
        entities=[[last_token_index(offsets, s, e) for s, e in r.entity_spans[n]] for n in entity_order],
        premises=[last_token_index(offsets, s, e) for s, e in r.premise_spans],
    )


def token_aligned(tokenizer):
    """Predicate: clean and counterfactual share token count, premise positions and decision position."""

    def layout(ex):
        r = render(ex)
        enc = tokenizer(r.text, return_offsets_mapping=True, add_special_tokens=False)
        pos = positions(r, [tuple(o) for o in enc["offset_mapping"]], [])
        return len(enc["input_ids"]), pos.decision, pos.premises

    return lambda clean, cf: layout(clean) == layout(cf)
