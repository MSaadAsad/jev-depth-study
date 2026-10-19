"""Validate and combine the complementary screens without changing either raw run."""
import json
from pathlib import Path

import compose_patch as compose
from jevprobe.evaluate import atomic_json, digest
from prepare_remaining_layers import validate_completed
from screen_layers import check


def combine_row(a, b, plan_hash):
    if a['uid'] != b['uid']:
        raise ValueError('Pair identity differs')
    for key in ('source_score', 'cf_score'):
        if abs(a[key] - b[key]) > 1e-5:
            raise ValueError('Repeated baseline changed; do not pool layer effects')
    merged = dict(a, plan_sha256=plan_hash)
    for key in ('patch_scores', 'self_scores', 'native_scores'):
        if set(a[key]) & set(b[key]):
            raise ValueError('Layer results overlap')
        merged[key] = {**a[key], **b[key]}
    return merged


def main():
    summaries = {}
    for model in ('jev', 'qwen'):
        dirs = [Path('runs') / run / model for run in ('layer-followup', 'layer-remainder')]
        a, b = [validate_completed(d) for d in dirs]
        for key in ('cases', 'model', 'revision', 'source_plan_sha256', 'source_sha256',
                    'scripts_sha256', 'versions', 'device', 'dtype'):
            if a[key] != b[key]:
                raise ValueError(f'Cross-run provenance mismatch: {model}/{key}')
        if set(a['layers']) & set(b['layers']) or sorted(a['layers'] + b['layers']) != list(range(1, 33)):
            raise ValueError('Expected complementary coverage of all 32 layers')
        plan = dict(a, layers=list(range(1, 33)))
        plan_hash = digest(json.dumps(plan, sort_keys=True).encode())
        rows = [{r['uid']: r for r in map(json.loads, (d / 'results.jsonl').read_text().splitlines())} for d in dirs]
        merged = []
        for case in plan['cases']:
            row = combine_row(rows[0][case['uid']], rows[1][case['uid']], plan_hash)
            check(row, case, plan, plan_hash)
            merged.append(row)
        old_layers = compose.LAYERS
        try:
            compose.LAYERS = tuple(range(1, 33))
            summary = compose.summarize(merged, plan['cases'])
        finally:
            compose.LAYERS = old_layers
        summary['controls'] = 'Eight self-patch sentinel pairs and two native-equivalence sentinel pairs per checkpoint, covering all 32 layers across the two runs.'
        summary['limitations'] = 'Same exploratory 40-pair/depth cohort; four edited premise records; fixed donor/target hop depth. Pointwise intervals, no multiplicity adjustment or independent confirmation.'
        summary['provenance'] = {str(d): {file: digest((d / file).read_bytes())
                                        for file in ('plan.json', 'results.jsonl')} for d in dirs}
        summary['repeated_baseline_max_error'] = max(abs(rows[0][uid][key] - rows[1][uid][key])
                                                   for uid in rows[0] for key in ('source_score', 'cf_score'))
        summaries[model] = summary
    destination = Path('runs/layer-remainder/combined-summary.json')
    atomic_json(destination, summaries)
    print(f'Validated all 32 layers on the paired cohort; combined summary: {destination}')


if __name__ == '__main__':
    main()
