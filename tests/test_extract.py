import json

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

from jevprobe.data.transitive import N_DISTRACT, N_MAIN, N_PREMISES, build_dataset, save_jsonl
from jevprobe.extract import check_storable
from jevprobe.extract import main as extract_main
from jevprobe.hooks import capture
from jevprobe.models import load_model
from jevprobe.readout import letter_ids, letter_score


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3.5 tokenizer not cached")
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048)
    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("tiny")
    AutoModelForCausalLM.from_config(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return d


def test_load_model_goes_straight_to_device(tiny_dir, monkeypatch):
    import transformers
    seen, original = {}, transformers.AutoModelForCausalLM.from_pretrained

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", spy)
    model, _ = load_model(str(tiny_dir), device="cpu")
    assert seen["device_map"] == {"": "cpu"}
    assert next(model.parameters()).device.type == "cpu"


def test_letter_score_matches_fp32_logits(tiny_dir):
    model, tok = load_model(str(tiny_dir), device="cpu")
    ids = torch.tensor([tok.encode("Is A above B?", add_special_tokens=False)])
    resid, logits = capture(model, ids)
    a, b = letter_ids(tok)
    assert letter_score(model, resid[-1, -1], (a, b)) == pytest.approx(float(logits[-1, a] - logits[-1, b]), abs=1e-4)


def test_check_storable_rejects_fp16_overflow_and_nonfinite():
    check_storable(torch.tensor([1.0, -6e4]))
    with pytest.raises(ValueError):
        check_storable(torch.tensor([1.0, 7e4]))
    with pytest.raises(ValueError):
        check_storable(torch.tensor([float("nan")]))


def test_extract_end_to_end_shapes(tiny_dir, tmp_path):
    data = tmp_path / "d.jsonl"
    save_jsonl(build_dataset(2, "test", seed=0)[:3], data)
    out = tmp_path / "run"
    extract_main(["--data", str(data), "--model", str(tiny_dir), "--out", str(out), "--device", "cpu"])
    assert np.load(out / "decision.npy").shape == (3, 3, 32)
    assert np.load(out / "entities.npy").shape == (3, 3, N_MAIN + N_DISTRACT, 32)
    assert np.load(out / "premises.npy").shape == (3, 3, N_PREMISES, 32)
    rows = [json.loads(l) for l in open(out / "meta.jsonl")]
    assert len(rows) == 3 and all(np.isfinite(r["score"]) and r["max_abs"] > 0 for r in rows)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple GPU")
def test_load_model_on_mps_avoids_device_map(tiny_dir, monkeypatch):
    """transformers' device_map={'': 'mps'} load segfaults (exit 139) on this stack; MPS loads then moves."""
    import transformers
    seen, original = {}, transformers.AutoModelForCausalLM.from_pretrained

    def spy(*args, **kwargs):
        seen.update(kwargs)
        kwargs.pop("device_map", None)  # never trigger the crash inside the test process
        return original(*args, **kwargs)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", spy)
    model, _ = load_model(str(tiny_dir), device="mps")
    assert "device_map" not in seen
    assert next(model.parameters()).device.type == "mps"
