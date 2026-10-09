import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

scripts = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('localization', scripts / 'localize_query_patch.py')
    local = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(local)
finally:
    sys.path.remove(str(scripts))


def test_multilayer_capture_single_name_patch_and_cleanup(monkeypatch):
    monkeypatch.setattr(local, 'LAYERS', (1, 2))
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)).eval()
    inputs = torch.tensor([[1, 2, 3, 4]])
    baseline, states = local.capture(model, inputs, (10, 11), [1, 2])
    assert set(states) == {1, 2}
    for layer in local.LAYERS:
        for slots in local.GROUPS.values():
            score = local.patched_score(model, inputs, (10, 11), layer, [[1, 2][s] for s in slots], states[layer][slots])
            assert score == pytest.approx(baseline)
    with pytest.raises(IndexError):
        local.capture(model, inputs, (10, 11), [99])
    assert all(not block._forward_hooks for block in local.find_decoder_layers(model))


def test_summary_and_reproduction_validation():
    cases, rows = [], []
    for answer in (False, True):
        sign = 1 if answer else -1
        case = {'example': {'uid': str(answer), 'answer': answer}}
        row = {'uid': str(answer), 'plan_sha256': 'hash', 'donor_scores': dict(long=-sign, same=sign, opposite=-sign, neutral=0),
               'scores': {f'{layer}/{group}/{kind}': sign if kind == 'same' else -sign
                          for layer in local.LAYERS for group in local.GROUPS for kind in ('same', 'opposite', 'neutral')}}
        row['scores'].update({f'{layer}/both/long': -sign for layer in local.LAYERS})
        original = {'donor_scores': row['donor_scores'], 'patched_scores': dict(same=sign, opposite=-sign, neutral=-sign)}
        local.validate(row, case, 'hash', original)
        cases.append(case)
        rows.append(row)
    result = local.summarize(rows, cases)
    assert result['passes'] == 68
    assert len(result['conditions']) == 9
    assert all(r['effects']['same']['rescued'] == 2 for r in result['conditions'].values())
    rows[-1]['scores']['16/both/same'] = -1
    with pytest.raises(ValueError, match='control failed'):
        local.validate(rows[-1], cases[-1], 'hash', original)
