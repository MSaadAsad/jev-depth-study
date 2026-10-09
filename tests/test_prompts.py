import json
import random
import pytest
from jevk5.prompt import prompt_text
from jevprobe.data.transitive import N_PREMISES, sample_example
from jevprobe.prompts import fill, render


def test_fill_spans():
    text, spans = fill("{a} > {b}.", {"a": "Mara", "b": "Tolen"})
    assert text == "Mara > Tolen."
    assert [(f, text[s:e]) for f, s, e in spans] == [("a", "Mara"), ("b", "Tolen")]


@pytest.mark.parametrize("split", ["train", "test"])
def test_render_spans_point_at_names(split):
    ex = sample_example(3, True, split, random.Random(5), "u")
    r = render(ex)
    assert set(r.entity_spans) == set(ex.chain + ex.distractors)
    for name, spans in r.entity_spans.items():
        assert all(r.text[s:e] == name for s, e in spans)
        endpoint = name in (ex.chain[0], ex.chain[-1], ex.distractors[0], ex.distractors[-1])
        assert len(spans) == (1 if endpoint else 2)
    assert len(r.premise_spans) == N_PREMISES
    assert all(r.text[e - 1] == "." for _, e in r.premise_spans)
    assert r.text[slice(*r.question_spans["x"])] == ex.x
    assert r.text[slice(*r.question_spans["y"])] == ex.y


def test_render_is_the_jevk5_prompt():
    ex = sample_example(2, False, "train", random.Random(0), "u")
    r = render(ex)
    payload = json.loads(r.text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0])
    assert payload["options"][0]["description"].startswith("true")
    assert payload["options"][1]["description"].startswith("false")
    assert r.text == prompt_text(payload["evidence"], payload["criterion"],
                                 [o["description"] for o in payload["options"]])
    assert r.text.endswith("</think>\n\n")
