import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from jevprobe.data.counterfactual import build_pairs
from prepare_layer_followup import audit_pair
from screen_layers import check, prepare, read_rows, workload
from compose_patch import GROUPS


def test_equal_depth_pairs_are_balanced_guarded_and_opposite():
    pairs = build_pairs(10, 'test', 20261016, ds=range(1, 5), aligned=lambda a, b: a.d == b.d)
    assert len(pairs) == 40
    for depth in range(1, 5):
        selected = [(a, b) for a, b in pairs if a.d == depth]
        assert sum(a.answer for a, b in selected) == 5
        for a, b in selected:
            audit_pair(a, b)
    a, b = pairs[0]
    with pytest.raises(ValueError, match='preserve depth'):
        audit_pair(a, replace(b, d=b.d + 1))
    with pytest.raises(ValueError, match='endpoint guards'):
        audit_pair(replace(a, premises=a.premises[:-1]), b)


def test_screen_uses_supplied_depths_and_deterministic_sentinels(tmp_path):
    cases = [{'uid': f'{d}-{a}-{i}', 'd': d, 'answer': a} for d in (1, 2, 3, 4)
             for a in (False, True) for i in range(2)]
    path = tmp_path / 'source.json'
    path.write_text(json.dumps({'cases': cases, 'fixed_counterfactual_depth': True}))
    args = SimpleNamespace(source_plan=path, n_per_depth=4, seed=20261016,
                           model='jev', device='mps', layers=[4, 16, 32])
    plan = prepare(args)
    assert plan == prepare(args)
    assert len(plan['cases']) == 16
    assert plan['fixed_counterfactual_depth']
    assert sum(c['self_control'] for c in plan['cases']) == 8
    assert [c['d'] for c in plan['cases'] if c['native_control']] == [1, 4]


def test_resume_checks_controls_and_repairs_only_incomplete_tail(tmp_path):
    positions = {g: [[], []] for g in GROUPS}
    positions['decision'] = [[1], [1]]
    case = {'uid': 'a', 'positions': positions,
            'self_control': True, 'native_control': True}
    plan = {'layers': [16], 'cases': [case]}
    row = {'uid': 'a', 'plan_sha256': 'frozen', 'source_score': 1., 'cf_score': -1.,
           'patch_scores': {'16/decision': .5}, 'native_scores': {'16/decision': .5},
           'self_scores': {'16/decision/source': 1., '16/decision/counterfactual': -1.}}
    check(row, case, plan, 'frozen')
    assert workload(plan) == {'interventions': 1, 'self_patches': 2, 'native_controls': 1,
                              'baselines': 2, 'total_scored_forwards': 6}
    path = tmp_path / 'results.jsonl'
    valid = json.dumps(row) + '\n'
    path.write_text(valid + '{"partial":')
    assert read_rows(path, plan, 'frozen') == {'a': row}
    assert path.read_text() == valid
    row['native_scores']['16/decision'] = .9
    with pytest.raises(ValueError, match='Prefix/native mismatch'):
        check(row, case, plan, 'frozen')
    row['native_scores']['16/decision'] = .5
    row['self_scores']['16/decision/source'] = .9
    with pytest.raises(ValueError, match='Self patch failed'):
        check(row, case, plan, 'frozen')
