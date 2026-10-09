import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location("patch_matched", scripts / "patch_matched.py")
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
finally:
    sys.path.remove(str(scripts))


def test_replace_preserves_source_and_tuple():
    original = torch.zeros(1, 5, 3)
    marker = object()
    result = patch.replace_positions((original, marker), [1, 3], torch.ones(2, 3))
    assert result[1] is marker
    assert original.sum() == 0
    assert result[0].sum() == 6
    with pytest.raises(ValueError):
        patch.replace_positions(original, [1, 1], torch.ones(2, 3))


def test_real_decoder_self_positive_and_cleanup():
    torch.manual_seed(3)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                                        num_hidden_layers=2, num_attention_heads=4,
                                        num_key_value_heads=2)).eval()
    short = torch.tensor([[1, 2, 3]])
    long = torch.tensor([[5, 6, 1, 2, 3]])
    layers = patch.find_decoder_layers(model)
    captured = []
    handle = layers[1].register_forward_hook(lambda m, i, o: captured.append((o[0] if isinstance(o, tuple) else o).detach().clone()))
    short_score = patch.forward_score(model, short, (10, 11))
    baseline = patch.forward_score(model, long, (10, 11))
    handle.remove()
    assert patch.patched_score(model, long, (10, 11), 2, [4], captured[1][0, -1:]) == pytest.approx(baseline)
    assert patch.patched_score(model, long, (10, 11), 2, [4], captured[0][0, -1:]) == pytest.approx(short_score)
    with pytest.raises(ValueError):
        patch.patched_score(model, long, (10, 11), 1, [99], torch.zeros(1, 32))
    assert not layers[0]._forward_hooks
    assert not layers[1]._forward_hooks


def test_balanced_matching_and_job_controls():
    catalog = [{"uid": f"{correct}-{answer}-{i}", "long_correct": correct, "answer": answer,
                "tokens": [[1]*8, [1]*8], "expected_scores": [1 if answer else -1, 0]}
               for correct in (False, True) for answer in (False, True) for i in range(7)]
    catalog[0]["tokens"][0][2] = 2
    selected, eligible = patch.choose_cohort(catalog, 6, 0)
    assert eligible == 27
    assert len(selected) == 24
    assert selected == patch.choose_cohort(catalog, 6, 0)[0]
    jobs = patch.make_jobs(selected, catalog, 0, 32)
    assert len(jobs) == len({j['id'] for j in jobs}) == 660
    assert all(j['kind'] in ('baseline', 'self', 'positive') for j in jobs[:40])
    for j in jobs:
        if j['kind'] == 'unrelated_same_answer':
            assert j['donor_uid'] != j['uid']
            assert j['group'] == 'decision'


def test_resume_and_control_failure(tmp_path):
    job = {'id': 'one', 'kind': 'baseline'}
    row = {'job': job, 'plan_sha256': 'hash', 'score': 1.0}
    path = tmp_path / 'results.jsonl'
    complete = json.dumps(row) + '\n'
    path.write_text(complete + '{"partial')
    assert patch.read_results(path, [job], 'hash') == {'one': row}
    assert path.read_text() == complete
    path.write_text(complete * 2)
    with pytest.raises(ValueError):
        patch.read_results(path, [job], 'hash')
    with pytest.raises(ValueError, match='control failed'):
        patch.check_control(job, -1, {'expected_scores': [1, 1]}, None)


def test_summary_uses_paired_signed_effects(tmp_path, monkeypatch):
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location('summarize_patching', scripts / 'summarize_patching.py')
        summary = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(summary)
    finally:
        sys.path.remove(str(scripts))
    catalog = [{'uid': f'{correct}-{answer}', 'long_correct': correct, 'answer': answer,
                'tokens': [[1]*8, [1]*8],
                'expected_scores': [1 if answer else -1, (1 if answer else -1) * (1 if correct else -1)]}
               for correct in (False, True) for answer in (False, True)]
    jobs = patch.make_jobs(catalog, catalog, 0, 32)
    plan = {'cohort': catalog, 'jobs': jobs, 'layers': list(patch.LAYERS), 'groups': patch.GROUPS}
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    plan_hash = patch.digest(json.dumps(plan, sort_keys=True).encode())
    by_uid = {c['uid']: c for c in catalog}
    with (tmp_path / 'results.jsonl').open('w') as f:
        for job in jobs:
            c = by_uid[job['uid']]
            score = c['expected_scores'][1] if job['kind'] in ('baseline', 'self', 'unrelated_same_answer') else c['expected_scores'][0]
            f.write(json.dumps({'job': job, 'plan_sha256': plan_hash, 'score': score}) + '\n')
    monkeypatch.setattr(sys, 'argv', ['summary', '--run', str(tmp_path)])
    summary.main()
    result = json.loads((tmp_path / 'summary.json').read_text())
    assert result['complete']
    assert result['max_control_abs_error'] == 0
    for row in result['paired_decision_comparisons']:
        assert row['same_question_minus_unrelated'] == (0 if row['originally_correct'] else 2)
