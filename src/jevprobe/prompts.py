"""Render examples as JevK5 prompts while tracking exact character spans."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from string import Formatter

from jevk5.prompt import decision_options, prompt_text

from jevprobe.data.transitive import Example
from jevprobe.data.vocab import PREMISE_TEMPLATES, QUESTION_TEMPLATE

OPTION_LETTERS = ("A", "B")  # jevk5 noul order: A = true, B = false
_OPTIONS = [text for _, text in decision_options({"type": "noul"})]


@dataclass(frozen=True)
class Rendered:
    text: str
    entity_spans: dict[str, list[tuple[int, int]]]
    premise_spans: list[tuple[int, int]]
    question_spans: dict[str, tuple[int, int]]


def fill(template: str, values: dict[str, str]) -> tuple[str, list[tuple[str, int, int]]]:
    out, spans, pos = [], [], 0
    for literal, field, _, _ in Formatter().parse(template):
        out.append(literal)
        pos += len(literal)
        if field is not None:
            v = values[field]
            spans.append((field, pos, pos + len(v)))
            out.append(v)
            pos += len(v)
    return "".join(out), spans


def _render_state(ex: Example):
    templates = PREMISE_TEMPLATES[ex.split]
    parts, ent_spans, prem_spans, pos = [], defaultdict(list), [], 0
    for k, p in enumerate(ex.premises):
        if k:
            parts.append(" ")
            pos += 1
        text, spans = fill(templates[p.template_id], {"g": p.greater, "l": p.lesser, "rel": ex.rel, "inv": ex.inv})
        for field, s, e in spans:
            if field in ("g", "l"):
                ent_spans[text[s:e]].append((pos + s, pos + e))
        prem_spans.append((pos, pos + len(text)))
        parts.append(text)
        pos += len(text)
    return "".join(parts), dict(ent_spans), prem_spans


def _offset_of(text: str, piece: str) -> int:
    """Where `piece` sits inside the JSON payload; it must survive json.dumps unescaped and occur once."""
    if json.dumps(piece) != f'"{piece}"':
        raise ValueError(f"text would be escaped by json.dumps: {piece!r}")
    if text.count(piece) != 1:
        raise ValueError(f"expected exactly one occurrence of {piece!r}")
    return text.index(piece)


def render(ex: Example) -> Rendered:
    state, ent, prem = _render_state(ex)
    question, q_spans = fill(QUESTION_TEMPLATE, {"x": ex.x, "rel": ex.rel, "y": ex.y})
    text = prompt_text(state, question, _OPTIONS)
    so, qo = _offset_of(text, state), _offset_of(text, question)
    return Rendered(
        text=text,
        entity_spans={n: [(s + so, e + so) for s, e in v] for n, v in ent.items()},
        premise_spans=[(s + so, e + so) for s, e in prem],
        question_spans={f: (s + qo, e + qo) for f, s, e in q_spans if f in ("x", "y")},
    )
