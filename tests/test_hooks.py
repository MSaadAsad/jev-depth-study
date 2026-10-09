import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen2Config
from jevprobe.hooks import capture, find_decoder_layers, find_final_norm

L, H, V = 3, 32, 128


@pytest.fixture(scope="module")
def tiny():
    cfg = Qwen2Config(vocab_size=V, hidden_size=H, intermediate_size=64, num_hidden_layers=L,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg).eval()


def test_finds_text_decoder(tiny):
    assert len(find_decoder_layers(tiny)) == L
    assert find_final_norm(tiny) is tiny.model.norm


def test_capture_matches_model(tiny):
    ids = torch.randint(0, V, (1, 10))
    resid, logits = capture(tiny, ids)
    out = tiny(input_ids=ids, output_hidden_states=True)
    assert resid.shape == (L + 1, 10, H)
    for k in range(L):  # embedding + first L-1 layers match regardless of final-norm convention
        assert torch.allclose(resid[k], out.hidden_states[k][0], atol=1e-5)
    head = tiny.get_output_embeddings()
    assert torch.allclose(head(find_final_norm(tiny)(resid[L])), out.logits[0], atol=1e-4)
    assert torch.allclose(logits, out.logits[0], atol=1e-5)


def test_hooks_removed(tiny):
    # transformers 5 installs its own persistent output-capturing hooks; ours must leave no trace
    count = lambda: [len(layer._forward_hooks) for layer in find_decoder_layers(tiny)] + [
        len(tiny.get_input_embeddings()._forward_hooks)]
    before = count()
    capture(tiny, torch.randint(0, V, (1, 5)))
    assert count() == before
