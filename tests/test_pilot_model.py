import random
import pytest
import torch
from jevprobe.data.transitive import sample_example
from jevprobe.hooks import capture, find_decoder_layers, find_final_norm
from jevprobe.models import load_model
from jevprobe.prompts import render
from jevprobe.readout import forward_score, letter_ids, letter_score, true_minus_false
from jevprobe.tokens import positions

pytestmark = pytest.mark.model
MODEL = "Qwen/Qwen3.5-0.8B"


@pytest.fixture(scope="module")
def pilot():
    return load_model(MODEL, device="cpu", dtype=torch.float32)


def test_pilot_layers_and_capture(pilot):
    model, tok = pilot
    assert len(find_decoder_layers(model)) == 24
    ex = sample_example(3, True, "test", random.Random(0), "u")
    r = render(ex)
    enc = tok(r.text, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
    pos = positions(r, enc["offset_mapping"][0].tolist(), ex.chain + ex.distractors)
    resid, logits = capture(model, enc["input_ids"])
    assert resid.shape[:2] == (25, enc["input_ids"].shape[1])
    recon = model.get_output_embeddings()(find_final_norm(model)(resid[-1, pos.decision]))
    assert torch.allclose(recon, logits[pos.decision], atol=1e-3)
    assert isinstance(true_minus_false(logits[pos.decision], letter_ids(tok)), float)
    direct = forward_score(model, enc["input_ids"], letter_ids(tok))
    assert direct == pytest.approx(letter_score(model, resid[-1, pos.decision], letter_ids(tok)), abs=1e-4)


def test_tokenization_matches_jevk5_runtime(pilot):
    """Our tokenization of the pinned template equals jevk5's apply_chat_template path."""
    from jevk5.prompt import messages
    _, tok = pilot
    ex = sample_example(2, False, "test", random.Random(1), "u")
    ours = tok(render(ex).text, add_special_tokens=False)["input_ids"]
    import json
    payload = json.loads(render(ex).text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0])
    theirs_text = tok.apply_chat_template(
        messages(payload["evidence"], payload["criterion"], [o["description"] for o in payload["options"]]),
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert ours == tok.encode(theirs_text, add_special_tokens=False)
