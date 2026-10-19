import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from patch_runtime import PrefixPatcher
from patch_matched import patched_score
from jevprobe.readout import forward_score


def tiny_model():
    torch.manual_seed(73)
    config = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                              num_hidden_layers=4, num_attention_heads=4,
                              num_key_value_heads=2, head_dim=8,
                              linear_key_head_dim=8, linear_value_head_dim=8,
                              linear_num_key_heads=2, linear_num_value_heads=2,
                              rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                                               'partial_rotary_factor': .5, 'mrope_section': [1, 1, 0]})
    return Qwen3_5ForCausalLM(config).eval()


def test_prefix_scores_match_native_in_hybrid_decoder_and_restore_model():
    model, ids = tiny_model(), (10, 11)
    engine = PrefixPatcher(model, ids)
    source, target = torch.randint(0, 64, (1, 12)), torch.randint(0, 64, (1, 12))
    _, donor = engine.capture(source)
    baseline, cached = engine.capture(target)
    for layer in range(1, 5):
        for positions in ([0], [4, 8], [11]):
            values = donor[layer][0, positions]
            native = patched_score(model, target, ids, layer, positions, values)
            fast = engine.patch(target, cached, layer, positions, values)
            assert fast == pytest.approx(native, abs=1e-7)
            assert forward_score(model, target, ids) == baseline
    assert not any('forward' in vars(block) for block in engine.blocks)


def test_prefix_restores_block_methods_after_patch_error():
    model, ids = tiny_model(), (10, 11)
    engine = PrefixPatcher(model, ids)
    inputs = torch.randint(0, 64, (1, 12))
    baseline, cached = engine.capture(inputs)
    with pytest.raises(ValueError, match='invalid patch positions'):
        engine.patch(inputs, cached, 3, [999], torch.zeros(1, 32))
    assert not any('forward' in vars(block) for block in engine.blocks)
    assert forward_score(model, inputs, ids) == baseline


def test_training_mode_is_rejected():
    with pytest.raises(ValueError, match='evaluation mode'):
        PrefixPatcher(tiny_model().train(), (10, 11))
