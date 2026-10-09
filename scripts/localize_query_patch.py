"""Localize the confirmed query-state effect on the same 100 questions.

Exploratory layers 15/16/17 x first/second/both names, three factual donors.
Each case: four multi-layer captures, 27 interventions, three self controls.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch

import confirm_query_patch as confirm
from jevprobe.evaluate import atomic_json, digest, run_lock
from jevprobe.hooks import find_decoder_layers
from jevprobe.models import load_model
from jevprobe.readout import forward_score, letter_ids
from patch_matched import patched_score

LAYERS = (15, 16, 17)
GROUPS = {'first': [0], 'second': [1], 'both': [0, 1]}


@torch.inference_mode()
def capture(model, inputs, ids, positions):
    states = {}
    handles = []
    def save(layer):
        def hook(_m, _i, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if layer in states:
                raise ValueError('duplicate hook call')
            states[layer] = tensor[0, positions].detach().clone()
        return hook
    try:
        blocks = find_decoder_layers(model)
        for layer in LAYERS:
            handles.append(blocks[layer - 1].register_forward_hook(save(layer)))
        score = forward_score(model, inputs, ids)
    finally:
        for handle in handles:
            handle.remove()
    if set(states) != set(LAYERS):
        raise ValueError('missing capture')
    return score, states


def close(actual, expected):
    if not math.isfinite(actual) or not math.isclose(actual, expected, abs_tol=.05, rel_tol=.01) or (actual > 0) != (expected > 0):
        raise ValueError(f'reproduction/self control failed: {actual} versus {expected}')


def validate(row, case, plan_hash, original):
    if row['uid'] != case['example']['uid'] or row['plan_sha256'] != plan_hash:
        raise ValueError('case provenance mismatch')
    if set(row['donor_scores']) != set(confirm.KINDS):
        raise ValueError('missing donors')
    for kind, score in row['donor_scores'].items():
        close(score, original['donor_scores'][kind])
    expected = {f'{layer}/{group}/{kind}' for layer in LAYERS for group in GROUPS
                for kind in ('same', 'opposite', 'neutral')}
    expected |= {f'{layer}/both/long' for layer in LAYERS}
    if set(row['scores']) != expected or not all(math.isfinite(s) for s in row['scores'].values()):
        raise ValueError('missing or invalid interventions')
    for layer in LAYERS:
        close(row['scores'][f'{layer}/both/long'], row['donor_scores']['long'])
    for kind in ('same', 'opposite', 'neutral'):
        close(row['scores'][f'16/both/{kind}'], original['patched_scores'][kind])


def summarize(rows, cases):
    results = {}
    for layer in LAYERS:
        for group in GROUPS:
            adapted = [{'donor_scores': row['donor_scores'], 'patched_scores': {
                kind: row['scores'][f'{layer}/{group}/{kind}'] if kind != 'long' else row['scores'][f'{layer}/both/long']
                for kind in confirm.KINDS}} for row in rows]
            results[f'{layer}/{group}'] = confirm.summarize(adapted, cases)
            # The reused summarizer's per-cell pass count is not the actual job count.
            del results[f'{layer}/{group}']['passes']
    return {'questions': len(rows), 'passes': 34 * len(rows), 'conditions': results,
            'limitations': 'Exploratory localization on the same confirmation cohort; pointwise intervals, no multiple-comparison adjustment. Individual patches need not add to the joint effect. Self controls patch both names.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('runs/query-localization'))
    ap.add_argument('--source', type=Path, default=Path('runs/query-confirmation'))
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    source = json.loads((args.source / 'plan.json').read_text())
    if source != confirm.prepare(len(source['cases']), source['seed']):
        raise ValueError('confirmation provenance changed')
    source_hash = digest(json.dumps(source, sort_keys=True).encode())
    originals = [json.loads((args.source / 'cases' / f'{i:04d}.json').read_text()) for i in range(len(source['cases']))]
    for row, case in zip(originals, source['cases']):
        confirm.validate(row, case, source_hash)
    plan = {'source_plan_sha256': source_hash, 'source_results_sha256': digest(json.dumps(originals, sort_keys=True).encode()),
            'layers': list(LAYERS), 'groups': GROUPS, 'script_sha256': digest(Path(__file__).read_bytes()),
            'confirmation_script_sha256': digest(Path(confirm.__file__).read_bytes()),
            'questions': len(originals), 'passes': len(originals) * 34,
            'scope': 'Exploratory localization; same confirmation questions, no outcome selection'}
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    plan_hash = digest(json.dumps(plan, sort_keys=True).encode())
    with run_lock(args.out):
        path = args.out / 'plan.json'
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError('plan changed; use new directory')
        if not path.exists() and (args.out / 'cases').exists():
            raise ValueError('cases without plan')
        atomic_json(path, plan)
        (args.out / 'cases').mkdir(exist_ok=True)
        rows = {}
        for i, case in enumerate(source['cases']):
            path = args.out / 'cases' / f'{i:04d}.json'
            if path.exists():
                row = json.loads(path.read_text())
                validate(row, case, plan_hash, originals[i])
                rows[i] = row
        def status(state):
            atomic_json(args.out / 'status.json', {'state': state, 'completed_questions': len(rows),
                'total_questions': len(originals), 'saved_passes': len(rows) * 34,
                'updated_at': datetime.now(timezone.utc).isoformat()})
        status('running')
        try:
            if len(rows) < len(originals):
                if not torch.backends.mps.is_available():
                    raise RuntimeError('Apple GPU unavailable')
                model, tok = load_model(confirm.MODEL, 'mps', torch.bfloat16, revision=confirm.REVISION)
                ids = letter_ids(tok)
                for i, case in enumerate(source['cases']):
                    if i in rows:
                        continue
                    prompts = case['prompts']
                    inputs = {k: torch.tensor([p['input_ids']], device='mps') for k, p in prompts.items()}
                    captures = {k: capture(model, inputs[k], ids, p['positions']) for k, p in prompts.items()}
                    scores = {}
                    for layer in LAYERS:
                        scores[f'{layer}/both/long'] = patched_score(model, inputs['long'], ids, layer,
                            prompts['long']['positions'], captures['long'][1][layer])
                        close(scores[f'{layer}/both/long'], captures['long'][0])
                        for group, slots in GROUPS.items():
                            for kind in ('same', 'opposite', 'neutral'):
                                scores[f'{layer}/{group}/{kind}'] = patched_score(model, inputs['long'], ids, layer,
                                    [prompts['long']['positions'][s] for s in slots], captures[kind][1][layer][slots])
                    row = {'uid': case['example']['uid'], 'plan_sha256': plan_hash,
                           'donor_scores': {k: v[0] for k, v in captures.items()}, 'scores': scores}
                    validate(row, case, plan_hash, originals[i])
                    atomic_json(args.out / 'cases' / f'{i:04d}.json', row)
                    rows[i] = row
                    status('running')
                    print(f'{len(rows)}/{len(originals)} questions saved; reproduction and self controls passed', flush=True)
                del model
            atomic_json(args.out / 'summary.json', summarize([rows[i] for i in range(len(originals))], source['cases']))
            status('complete')
            print('Localization and summary complete', flush=True)
        except BaseException:
            status('interrupted')
            raise


if __name__ == '__main__':
    main()
