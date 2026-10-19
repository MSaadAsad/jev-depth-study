import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from prepare_remaining_layers import remaining_layers
from combine_layer_screens import combine_row


def test_complement_covers_all_blocks_once_and_includes_both_boundaries():
    covered = [4, 8, 12, 16, 20, 24, 28, 32]
    missing = remaining_layers(covered)
    assert len(missing) == 24
    assert 1 in missing and 31 in missing
    assert not set(missing) & set(covered)
    assert sorted(missing + covered) == list(range(1, 33))
    assert all(i % 4 != 0 for i in missing)


@pytest.mark.parametrize('covered', [[4, 4], [0], [33], list(range(1, 33))])
def test_invalid_or_exhausted_coverage_is_rejected(covered):
    with pytest.raises(ValueError):
        remaining_layers(covered)


def test_combination_preserves_scores_and_rejects_drift_and_overlap():
    a = {'uid': 'pair', 'source_score': 1., 'cf_score': -1., 'plan_sha256': 'a',
         'patch_scores': {'4/query_names': .4}, 'self_scores': {}, 'native_scores': {}}
    b = {**a, 'plan_sha256': 'b', 'patch_scores': {'3/query_names': .3}}
    combined = combine_row(a, b, 'combined')
    assert combined['patch_scores'] == {'4/query_names': .4, '3/query_names': .3}
    assert a['plan_sha256'] == 'a'
    with pytest.raises(ValueError, match='baseline changed'):
        combine_row(a, {**b, 'source_score': 1.1}, 'combined')
    with pytest.raises(ValueError, match='overlap'):
        combine_row(a, a, 'combined')
    with pytest.raises(ValueError, match='identity'):
        combine_row(a, {**b, 'uid': 'different'}, 'combined')
