import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from jevprobe.data.transitive import build_dataset
from jevprobe.prompts import render

scripts = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('confirmation', scripts / 'confirm_query_patch.py')
    confirmation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(confirmation)
finally:
    sys.path.remove(str(scripts))


def test_donors_keep_question_but_change_fact():
    for ex in build_dataset(20, 'test', 123, ds=[1]):
        variants = confirmation.variants(ex)
        p = variants['same'].premises[0]
        opposite = variants['opposite']
        assert opposite.answer != ex.answer
        assert opposite.premises[0].greater == p.lesser
        assert opposite.premises[0].lesser == p.greater
        assert (opposite.rank_x < opposite.rank_y) == opposite.answer
        neutral = variants['neutral'].premises[0]
        assert not {neutral.greater, neutral.lesser} & {ex.x, ex.y}
        for donor in variants.values():
            assert (donor.x, donor.y, donor.rel) == (ex.x, ex.y, ex.rel)
            assert render(donor).text


def test_capture_self_and_cleanup(monkeypatch):
    monkeypatch.setattr(confirmation, 'LAYER', 1)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)).eval()
    inputs = torch.tensor([[1, 2, 3, 4]])
    base, states = confirmation.capture(model, inputs, (10, 11), [1, 2])
    assert states.shape == (2, 16)
    score = confirmation.patched_score(model, inputs, (10, 11), 1, [1, 2], states)
    assert score == pytest.approx(base)
    with pytest.raises(IndexError):
        confirmation.capture(model, inputs, (10, 11), [99])
    assert not confirmation.find_decoder_layers(model)[0]._forward_hooks


def test_summary_counts_rescues_regressions_and_signed_contrasts():
    cases = [{'example': {'answer': a}} for a in (True, True, False, False)]
    rows = []
    for i, case in enumerate(cases):
        sign = 1 if case['example']['answer'] else -1
        base = sign if i % 2 else -sign
        rows.append({'donor_scores': dict(long=base, same=sign, opposite=-sign, neutral=0),
                     'patched_scores': dict(long=base, same=sign, opposite=-sign, neutral=base)})
    summary = confirmation.summarize(rows, cases)
    assert summary['baseline_correct'] == 2
    assert summary['effects']['same']['rescued'] == 2
    assert summary['effects']['same']['regressed'] == 0
    assert summary['effects']['opposite']['regressed'] == 2
    assert summary['paired_contrasts']['opposite']['same_minus_control_accuracy'] == 1
    assert summary['paired_contrasts']['neutral']['same_minus_control_signed_score'] == 1


def test_resume_validation_rejects_bad_provenance_and_controls():
    row = {'uid': 'one', 'plan_sha256': 'hash', 'donor_scores': dict.fromkeys(confirmation.KINDS, 1.),
           'patched_scores': dict.fromkeys(confirmation.KINDS, 1.)}
    case = {'example': {'uid': 'one'}}
    confirmation.validate(row, case, 'hash')
    with pytest.raises(ValueError, match='provenance'):
        confirmation.validate(row, case, 'other')
    row['patched_scores']['long'] = -1
    with pytest.raises(ValueError, match='self control'):
        confirmation.validate(row, case, 'hash')
