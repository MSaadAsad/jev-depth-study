"""Freeze the complementary 24-layer run; inspect saved results/configs, never weights."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoConfig
from jevprobe.evaluate import atomic_json, digest
from screen_layers import MODELS, check, prepare, workload

SOURCE = Path('data/layer-followup/plan.json')
PREVIOUS = Path('runs/layer-followup')
OUT = Path('data/layer-remainder/manifest.json')


def remaining_layers(covered, total=32):
    if len(set(covered)) != len(covered) or any(x < 1 or x > total for x in covered):
        raise ValueError('Invalid or duplicate covered layers')
    missing = sorted(set(range(1, total + 1)) - set(covered))
    if not missing:
        raise ValueError('No remaining layers')
    return missing


def validate_completed(directory):
    plan = json.loads((directory / 'plan.json').read_text())
    plan_hash = digest(json.dumps(plan, sort_keys=True).encode())
    rows = [json.loads(line) for line in (directory / 'results.jsonl').read_text().splitlines()]
    cases = {c['uid']: c for c in plan['cases']}
    if len(rows) != len(cases) or {r['uid'] for r in rows} != set(cases):
        raise ValueError('Prior run is incomplete or contains duplicate pairs')
    for row in rows:
        check(row, cases[row['uid']], plan, plan_hash)
    return plan


def build_manifest():
    previous = {name: validate_completed(PREVIOUS / name) for name in MODELS}
    covered = previous['jev']['layers']
    if previous['qwen']['layers'] != covered:
        raise ValueError('Checkpoints have different prior layer coverage')
    layers = remaining_layers(covered)
    plans = {}
    for name, (repo, revision) in MODELS.items():
        old = previous[name]
        config = AutoConfig.from_pretrained(repo, revision=revision, local_files_only=True).get_text_config()
        if config.num_hidden_layers != 32 or any(config.layer_types[i - 1] != 'linear_attention' for i in layers):
            raise ValueError('Remaining layers do not match the pinned hybrid architecture')
        args = SimpleNamespace(source_plan=SOURCE, n_per_depth=old['n_per_depth'],
                               seed=old['seed'], model=name, device=old['device'], layers=layers)
        new = prepare(args)
        for key in ('model', 'revision', 'source_plan_sha256', 'source_sha256',
                    'scripts_sha256', 'versions', 'cases', 'device', 'dtype'):
            if new[key] != old[key]:
                raise ValueError(f'{name}: comparison provenance changed at {key}')
        plans[name] = {
            'runner_plan_sha256': digest(json.dumps(new, sort_keys=True).encode()),
            'previous_plan_sha256': digest((PREVIOUS / name / 'plan.json').read_bytes()),
            'previous_results_sha256': digest((PREVIOUS / name / 'results.jsonl').read_bytes()),
            'model': repo, 'revision': revision, 'pairs': len(new['cases']),
            'workload': workload(new),
        }
    return {'schema': 1, 'source_plan': str(SOURCE),
            'covered_layers': covered, 'layers': layers, 'block_type': 'Gated DeltaNet',
            'models': plans, 'preparation_script_sha256': digest(Path(__file__).read_bytes()),
            'scope': 'Complementary localization on the same cohort, not independent confirmation.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    manifest = build_manifest()
    if OUT.exists() and json.loads(OUT.read_text()) != manifest:
        raise ValueError('Frozen preparation differs; review changes before creating another manifest')
    if not OUT.exists():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(OUT, manifest)
    total = {k: sum(p['workload'][k] for p in manifest['models'].values())
             for k in next(iter(manifest['models'].values()))['workload']}
    print(json.dumps({'manifest': str(OUT), 'layers': manifest['layers'],
                      'totals_both_models': total, 'model_weights_loaded': False}), flush=True)


if __name__ == '__main__':
    main()
