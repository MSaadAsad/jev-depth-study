import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from jevprobe.data.transitive import build_dataset
from jevprobe.hooks import capture, find_decoder_layers
from jevprobe.prompts import render
from jevprobe.readout import forward_score

spec = importlib.util.spec_from_file_location("extract_matched", Path(__file__).parents[1] / "scripts/extract_matched.py")
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)


@pytest.fixture
def tiny():
    torch.manual_seed(3)
    cfg = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    return AutoModelForCausalLM.from_config(cfg).eval()


def test_selective_capture_matches_full_states_and_readout_without_lm_head(tiny):
    inputs = torch.randint(0, 128, (1, 24))
    full, logits = capture(tiny, inputs)
    indices = [23, 15, 4, 9, 18, 20, -1, -1]

    def forbid_head(*args):
        raise AssertionError("selective capture must not compute full vocabulary logits")

    handle = tiny.lm_head.register_forward_hook(forbid_head)
    try:
        selected = extract.capture_selected(tiny, inputs, indices)
    finally:
        handle.remove()
    assert selected.shape == (3, 8, 32)
    assert torch.allclose(selected[:, :6], full[:, indices[:6]], atol=1e-5)
    assert torch.count_nonzero(selected[:, 6:]) == 0
    projection = extract.project_letters(tiny, selected, (32, 33))
    assert float(projection[-1, 0]) == pytest.approx(float(logits[-1, 32] - logits[-1, 33]), abs=1e-5)


def test_hooks_removed_when_forward_fails(tiny, monkeypatch):
    modules = [tiny.get_input_embeddings(), *find_decoder_layers(tiny)]
    before = [len(m._forward_hooks) for m in modules]

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(tiny.model, "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        extract.capture_selected(tiny, torch.tensor([[1, 2]]), [1, -1])
    assert [len(m._forward_hooks) for m in modules] == before


def test_slots_track_all_query_mentions_and_missing_neighbors():
    long = build_dataset(2, "test", 0, ds=[1])[0]
    p = next(p for p in long.premises if {p.greater, p.lesser} == {long.x, long.y})
    short = replace(long, premises=[p])
    for ex in (short, long):
        rendered = render(ex)
        offsets = [(i, i + 1) for i in range(len(rendered.text))]
        positions = extract.selected_positions(ex, rendered, offsets)
        assert len(positions) == 8
        assert positions[0] == len(rendered.text) - 1
        assert rendered.text[positions[1]] == "."
        for index, name in ((2, ex.x), (3, ex.y), (4, ex.x), (5, ex.y)):
            assert rendered.text[positions[index] - len(name) + 1:positions[index] + 1] == name
        if ex is short:
            assert positions[6:] == [-1, -1]
        else:
            assert all(p >= 0 for p in positions[6:])
            assert positions[6] != positions[2] and positions[7] != positions[3]


def fixture_pair():
    states = np.zeros((2, 3, 8, 32), dtype=np.float16)
    lens = np.zeros((2, 3, 8), dtype=np.float32)
    pos = np.array([[23, 15, 4, 9, 18, 20, -1, -1], [23, 15, 4, 9, 18, 20, 6, 10]], dtype=np.int32)
    case = {"uid": "one", "expected_scores": [0., 0.]}
    meta = {"case": case, "manifest_sha256": "hash", "input_ids": [list(range(24))] * 2}
    return states, lens, pos, case, meta


def test_atomic_storage_keeps_previous_pair_on_interrupted_write(tmp_path, monkeypatch):
    states, lens, pos, case, meta = fixture_pair()
    path = tmp_path / "pair.npz"
    extract.atomic_pair(path, states, lens, pos, meta)
    before = path.read_bytes()

    def fail(file, **kwargs):
        file.write(b"incomplete")
        raise OSError("disk error")

    monkeypatch.setattr(extract.np, "savez", fail)
    with pytest.raises(OSError):
        extract.atomic_pair(path, states, lens, pos, meta)
    assert path.read_bytes() == before
    got = extract.read_pair(path, case, states.shape, "hash")
    assert np.array_equal(got[0], states) and got[3] == meta


@pytest.mark.parametrize("fault", ["manifest", "shape", "nonfinite", "missing", "readout"])
def test_resume_rejects_invalid_saved_pairs(tmp_path, fault):
    states, lens, pos, case, meta = fixture_pair()
    if fault == "manifest":
        meta["manifest_sha256"] = "wrong"
    elif fault == "shape":
        states = states[:, :, :, :4]
    elif fault == "nonfinite":
        states[0, 0, 0, 0] = np.inf
    elif fault == "missing":
        states[0, 0, 6, 0] = 1
    else:
        lens[0, -1, 0] = 10
    path = tmp_path / "pair.npz"
    extract.atomic_pair(path, states, lens, pos, meta)
    with pytest.raises(ValueError):
        extract.read_pair(path, case, (2, 3, 8, 32), "hash")


def test_storage_budget_for_full_experiment(tmp_path):
    budget = extract.disk_budget(tmp_path, 300, (2, 33, 8, 2560))
    assert budget["activation_bytes"] == 811_008_000
    assert budget["required_free_bytes"] > budget["activation_bytes"] + 2 * 1024**3


def test_end_to_end_partial_run_resume_and_summary(tiny, tmp_path, monkeypatch):
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [32 if text == "A" else 33]

        def __call__(self, text, **kwargs):
            offsets = [(i, min(i + 8, len(text))) for i in range(0, len(text), 8)]
            return {"input_ids": (torch.arange(len(offsets)) % 128)[None], "offset_mapping": torch.tensor([offsets])}

    tok = Tokenizer()
    longs = build_dataset(2, "test", 0, ds=[1])
    examples = [{}, {}]
    cases = []
    for long in longs:
        short = replace(long, premises=[next(p for p in long.premises if {p.greater, p.lesser} == {long.x, long.y})])
        scores = []
        for context, ex in enumerate((short, long)):
            examples[context][ex.uid] = ex
            scores.append(forward_score(tiny, tok(render(ex).text)["input_ids"], (32, 33)))
        cases.append({"uid": long.uid, "answer": long.answer, "expected_scores": scores,
                      "long_correct": (scores[1] > 0) == long.answer})
    config = {"model": "fake/model", "revision": "immutable", "device": "cpu", "dtype": "float32"}
    monkeypatch.setattr(extract, "load_sources", lambda *args: (config, examples, cases))
    monkeypatch.setattr(extract, "AutoConfig", SimpleNamespace(from_pretrained=lambda *args, **kw: tiny.config))
    monkeypatch.setattr(extract, "load_model", lambda *args, **kw: (tiny, tok))
    out = tmp_path / "run"
    args = ["--out", str(out)]
    extract.main(args + ["--max-pairs", "1"])
    assert json.loads((out / "status.json").read_text())["state"] == "partial"
    first = (out / "pairs/0000.npz").read_bytes()
    extract.main(args)
    assert (out / "pairs/0000.npz").read_bytes() == first
    assert json.loads((out / "status.json").read_text())["state"] == "complete"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["completed_pairs"] == 2 and summary["prediction_disagreements_with_behavior"] == 0
    assert summary["max_readout_difference"] < 1e-4
    assert (out / "layers.csv").exists()
    monkeypatch.setattr(extract, "load_model", lambda *a, **kw: pytest.fail("loaded an already-completed model"))
    extract.main(args)
    extract.main(args + ["--analyze-only"])
