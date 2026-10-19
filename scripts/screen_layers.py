"""Resumable, paired eight-layer screening; separate from mechanistic confirmation.

One model and one pair in memory. Original completed runs remain untouched.
Cached-prefix scores must reproduce native patches on sentinel pairs.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')

import torch
import compose_patch as compose
from jevprobe.data.transitive import Example
from jevprobe.evaluate import atomic_json, digest, run_lock, source_digest
from jevprobe.models import load_model
from jevprobe.readout import letter_ids
from patch_matched import patched_score
from patch_runtime import PrefixPatcher

MODELS = {'jev': (compose.MODEL, compose.REVISION),
          'qwen': ('Qwen/Qwen3.5-4B', '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')}
DEFAULT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)


def workload(plan):
    primary = self_patches = native = 0
    for case in plan['cases']:
        count = len(plan['layers']) * sum(bool(case['positions'][g][0]) for g in compose.GROUPS)
        primary += count
        self_patches += 2 * count if case['self_control'] else 0
        native += count if case['native_control'] else 0
    baseline = 2 * len(plan['cases'])
    return {'interventions': primary, 'self_patches': self_patches,
            'native_controls': native, 'baselines': baseline,
            'total_scored_forwards': primary + self_patches + native + baseline}


def prepare(args):
    source_path = args.source_plan
    original = json.loads(source_path.read_text())
    chosen = []
    depths = sorted({c['d'] for c in original['cases']})
    for depth in depths:
        for answer in (False, True):
            pool = [c for c in original['cases'] if c['d'] == depth and c['answer'] == answer]
            random.Random(f'layer-screen-{args.seed}-{depth}-{answer}').shuffle(pool)
            if len(pool) < args.n_per_depth // 2:
                raise ValueError('Screen cohort exceeds available source pairs')
            for i, item in enumerate(pool[:args.n_per_depth // 2]):
                case = dict(item)
                case['self_control'] = i == 0
                case['native_control'] = i == 0 and not answer and depth in (depths[0], depths[-1])
                chosen.append(case)
    random.Random(f'layer-screen-order-{args.seed}').shuffle(chosen)
    # Validate both a one-hop and multi-hop sentinel before normal screening.
    chosen.sort(key=lambda c: not c['native_control'])
    repo, revision = MODELS[args.model]
    files = ('screen_layers.py', 'patch_runtime.py', 'patch_matched.py', 'compose_patch.py')
    return {'schema': 1, 'model': repo, 'revision': revision, 'device': args.device,
            'dtype': 'bfloat16', 'layers': args.layers, 'n_per_depth': args.n_per_depth,
            'seed': args.seed, 'source_plan_sha256': digest(source_path.read_bytes()),
            'source_sha256': source_digest(),
            'scripts_sha256': {f: digest(Path(__file__).with_name(f).read_bytes()) for f in files},
            'versions': {p: version(p) for p in ('torch', 'transformers', 'numpy', 'jevk5')},
            'fixed_counterfactual_depth': original.get('fixed_counterfactual_depth', False),
            'cases': chosen, 'purpose': 'Exploratory layer/position screening; requires independent confirmation.'}


def check(row, case, plan, plan_hash):
    if row['uid'] != case['uid'] or row['plan_sha256'] != plan_hash:
        raise ValueError('Result provenance mismatch')
    expected = {f'{l}/{g}' for l in plan['layers'] for g in compose.GROUPS if case['positions'][g][0]}
    if set(row['patch_scores']) != expected:
        raise ValueError('Missing patch scores')
    scores = [row['source_score'], row['cf_score'], *row['patch_scores'].values(),
              *row['self_scores'].values(), *row['native_scores'].values()]
    if not all(math.isfinite(s) for s in scores):
        raise ValueError('Nonfinite score')
    expected_self = {f'{key}/{kind}' for key in expected for kind in ('source', 'counterfactual')} if case['self_control'] else set()
    if set(row['self_scores']) != expected_self:
        raise ValueError('Missing self-patch controls')
    for key, score in row['self_scores'].items():
        base = row['source_score'] if key.endswith('/source') else row['cf_score']
        if abs(score - base) > 1e-5:
            raise ValueError(f'Self patch failed: {key}')
    if set(row['native_scores']) != (expected if case['native_control'] else set()):
        raise ValueError('Missing native-equivalence controls')
    for key, score in row['native_scores'].items():
        if abs(score - row['patch_scores'][key]) > 1e-5:
            raise ValueError(f'Prefix/native mismatch: {key}')
    if 32 in plan['layers'] and abs(row['patch_scores']['32/decision'] - row['source_score']) > .05:
        raise ValueError('Final-decision positive control failed')


def read_rows(path, plan, plan_hash):
    rows, cases = {}, {c['uid']: c for c in plan['cases']}
    if path.exists():
        with path.open('r+b') as f:
            while True:
                start, raw = f.tell(), f.readline()
                if not raw:
                    break
                if not raw.endswith(b'\n'):
                    f.truncate(start)
                    break
                row = json.loads(raw)
                if row['uid'] not in cases or row['uid'] in rows:
                    raise ValueError('Unknown/duplicate result')
                check(row, cases[row['uid']], plan, plan_hash)
                rows[row['uid']] = row
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=MODELS, required=True)
    ap.add_argument('--source-plan', type=Path, default=Path('runs/composition-patching/plan.json'))
    ap.add_argument('--device', choices=('mps', 'cuda'), default='mps')
    ap.add_argument('--layers', type=int, nargs='+', default=list(DEFAULT_LAYERS))
    ap.add_argument('--n-per-depth', type=int, default=40)
    ap.add_argument('--seed', type=int, default=20261015)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-pairs', type=int)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.n_per_depth < 2 or args.n_per_depth % 2:
        ap.error('n-per-depth must be positive and even')
    if len(set(args.layers)) != len(args.layers) or any(l < 1 or l > 32 for l in args.layers):
        ap.error('layers must be distinct 1-based block indices in 1..32')
    plan = prepare(args)
    counts = Counter(c['d'] for c in plan['cases'])
    print(json.dumps({'model': args.model, 'pairs': len(plan['cases']), 'pairs_per_depth': counts,
                      'layers': args.layers, **workload(plan),
                      'native_sentinels': sum(c['native_control'] for c in plan['cases']),
                      'self_sentinels': sum(c['self_control'] for c in plan['cases'])}), flush=True)
    if args.dry_run:
        return
    plan_hash = digest(json.dumps(plan, sort_keys=True).encode())
    with run_lock(args.out):
        pp = args.out / 'plan.json'
        if pp.exists() and json.loads(pp.read_text()) != plan:
            raise ValueError('Resume plan/provenance differs; choose a new output directory')
        if not pp.exists():
            if (args.out / 'results.jsonl').exists():
                raise ValueError('Results exist without a plan')
            atomic_json(pp, plan)
        rows = read_rows(args.out / 'results.jsonl', plan, plan_hash)
        pending = [c for c in plan['cases'] if c['uid'] not in rows]
        started = time.monotonic()
        processed = 0
        def status(state):
            atomic_json(args.out / 'status.json', {'state': state, 'completed_pairs': len(rows),
                        'total_pairs': len(plan['cases']), 'new_pairs_this_process': processed,
                        'elapsed_seconds_this_process': time.monotonic() - started,
                        'updated_at': datetime.now(timezone.utc).isoformat()})
        status('running')
        try:
            if pending:
                if args.device == 'mps':
                    if not torch.backends.mps.is_available():
                        raise RuntimeError('MPS unavailable')
                    torch.mps.set_per_process_memory_fraction(.9)
                elif not torch.cuda.is_available():
                    raise RuntimeError('CUDA unavailable')
                model, tok = load_model(plan['model'], args.device, torch.bfloat16, revision=plan['revision'])
                ids = letter_ids(tok)
                engine = PrefixPatcher(model, ids)
                started = time.monotonic()
                with (args.out / 'results.jsonl').open('a') as f:
                    for case in pending[:args.max_pairs]:
                        tick = time.monotonic()
                        layouts = [compose.layout(Example.from_dict(case[k]), tok) for k in ('source', 'counterfactual')]
                        if [x['prompt_sha256'] for x in layouts] != case['prompt_hashes']:
                            raise ValueError('Rendered prompt changed')
                        inputs = [torch.tensor([x['input_ids']], device=args.device) for x in layouts]
                        source_score, source = engine.capture(inputs[0])
                        cf_score, cf = engine.capture(inputs[1])
                        row = {'uid': case['uid'], 'plan_sha256': plan_hash, 'source_score': source_score,
                               'cf_score': cf_score, 'patch_scores': {}, 'self_scores': {}, 'native_scores': {}}
                        for layer in args.layers:
                            for group in compose.GROUPS:
                                src, dst = case['positions'][group]
                                if not src:
                                    continue
                                key = f'{layer}/{group}'
                                values = source[layer][0, src]
                                row['patch_scores'][key] = engine.patch(inputs[1], cf, layer, dst, values)
                                if case['native_control']:
                                    row['native_scores'][key] = patched_score(model, inputs[1], ids, layer, dst, values)
                                if case['self_control']:
                                    for kind, inp, states, pos in (('source', inputs[0], source, src),
                                                                   ('counterfactual', inputs[1], cf, dst)):
                                        row['self_scores'][f'{key}/{kind}'] = engine.patch(inp, states, layer, pos, states[layer][0, pos])
                        row['seconds'] = time.monotonic() - tick
                        row['tokens'] = len(layouts[0]['input_ids'])
                        check(row, case, plan, plan_hash)
                        f.write(json.dumps(row, allow_nan=False) + '\n')
                        f.flush(); os.fsync(f.fileno())
                        rows[case['uid']] = row
                        processed += 1
                        status('running')
                        print(f"{len(rows)}/{len(plan['cases'])} pairs; d={case['d']}; {row['seconds']:.2f} s/pair; native_control={case['native_control']}; self_control={case['self_control']}", flush=True)
                        del source, cf, inputs
                del engine, model
            if len(rows) == len(plan['cases']):
                old_layers = compose.LAYERS
                try:
                    compose.LAYERS = tuple(args.layers)
                    summary = compose.summarize(list(rows.values()), plan['cases'])
                finally:
                    compose.LAYERS = old_layers
                summary['scope'] = plan['purpose']
                summary['method'] = 'Batch-one counterfactual prefix reuse; native equivalence and self patches on prespecified sentinel pairs.'
                if plan['fixed_counterfactual_depth']:
                    summary['limitations'] = ('Exploratory pointwise intervals, no multiplicity adjustment; '
                        'counterfactual swaps change four premises but preserve hop depth. '
                        'Patch effects locate answer-sensitive states, not a unique composition algorithm.')
                atomic_json(args.out / 'summary.json', summary)
            status('complete' if len(rows) == len(plan['cases']) else 'partial')
        except BaseException:
            status('interrupted')
            raise


if __name__ == '__main__':
    main()
