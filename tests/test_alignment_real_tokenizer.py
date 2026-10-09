import pytest
from transformers import AutoTokenizer

from jevprobe.data.counterfactual import build_pairs
from jevprobe.prompts import render
from jevprobe.tokens import token_aligned

pytestmark = pytest.mark.model


def test_pairs_token_aligned_under_qwen_tokenizer():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    pairs = build_pairs(10, "test", seed=0, aligned=token_aligned(tok))
    for clean, cf in pairs:
        a = tok(render(clean).text, add_special_tokens=False)["input_ids"]
        b = tok(render(cf).text, add_special_tokens=False)["input_ids"]
        assert len(a) == len(b)
        assert sum(x != y for x, y in zip(a, b)) > 0  # the swap is real
