"""Fresh, outcome-independent confirmation of layer-16 question-name patching.

100 balanced one-hop questions, fixed seed; same test vocabulary/templates as
the pilot. Donors retain the exact question but give the correct, reversed, or
unrelated single fact. The unrelated donor does not determine a correct answer.
Eight passes per case: four captures and four patches, including self control.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from make_context_sweep import context_variant
from patch_matched import patched_score
from jevprobe.data.transitive import build_dataset, load_jsonl, Premise
from jevprobe.evaluate import atomic_json, digest, run_lock, source_digest
from jevprobe.hooks import find_decoder_layers
from jevprobe.models import load_model
from jevprobe.prompts import render
from jevprobe.readout import forward_score, letter_ids
from jevprobe.tokens import last_token_index

MODEL = 'alibiserikbay/JevK5'
REVISION = 'c4f7fdb3aeab5582336406e78d3bef11bf98833d'
LAYER = 16
KINDS = ('long', 'same', 'opposite', 'neutral')


def variants(ex):
    short, _, _ = context_variant(ex, 1)
    p = short.premises[0]
    opposite = replace(short, answer=not ex.answer, chain=list(reversed(ex.chain)),
                       premises=[Premise(p.lesser, p.greater, p.template_id)])
    # Use the same template and relation; change only the fact's participants.
    neutral = replace(short, premises=[Premise(ex.distractors[0], ex.distractors[1], p.template_id)])
    return dict(long=ex, same=short, opposite=opposite, neutral=neutral)


def encode(tok, ex):
    rendered = render(ex)
    encoded = tok(rendered.text, return_offsets_mapping=True, add_special_tokens=False)
    positions = [last_token_index(encoded['offset_mapping'], *rendered.question_spans[k]) for k in ('x', 'y')]
    return {'input_ids': encoded['input_ids'], 'positions': positions,
            'tokens': [encoded['input_ids'][p] for p in positions],
            'prompt_sha256': digest(rendered.text.encode())}


def prepare(n, seed):
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    examples = [replace(e, uid=f'confirm-{seed}-{e.uid}') for e in build_dataset(n, 'test', seed, ds=[1])]
    old = {digest(render(e).text.encode()) for e in load_jsonl('data/e1/test.jsonl')}
    cases = []
    for ex in examples:
        prompts = {k: encode(tok, v) for k, v in variants(ex).items()}
        if prompts['long']['prompt_sha256'] in old:
            raise ValueError('fresh dataset overlaps original prompts')
        if any(p['tokens'] != prompts['long']['tokens'] for p in prompts.values()):
            raise ValueError('question token mismatch')
        cases.append({'example': ex.to_dict(), 'prompts': prompts})
    if len({c['prompts']['long']['prompt_sha256'] for c in cases}) != n:
        raise ValueError('duplicate fresh prompt')
    return {'model': MODEL, 'revision': REVISION, 'layer': LAYER, 'seed': seed,
            'script_sha256': digest(Path(__file__).read_bytes()), 'source_sha256': source_digest(),
            'helper_sha256': {p: digest(Path(__file__).with_name(p).read_bytes()) for p in
                              ('patch_matched.py', 'make_context_sweep.py')},
            'versions': {p: version(p) for p in ('torch', 'transformers', 'numpy', 'jevk5')},
            'device': 'mps', 'dtype': 'bfloat16', 'cases': cases,
            'primary': 'paired accuracy change, same-fact patch minus long baseline, on all 100 fresh cases',
            'secondary': 'same minus opposite and neutral patches; rescue and regression counts; answer-signed score changes',
            'limitations': 'One seed and model; same vocabulary/templates; neutral donor lacks evidence; donor lengths/positions can differ; reversal changes fact participant order.'}


@torch.inference_mode()
def capture(model, inputs, ids, positions):
    states = []
    def save(_m, _i, output):
        tensor = output[0] if isinstance(output, tuple) else output
        states.append(tensor[0, positions].detach().clone())
    handle = find_decoder_layers(model)[LAYER - 1].register_forward_hook(save)
    try:
        score = forward_score(model, inputs, ids)
    finally:
        handle.remove()
    if len(states) != 1:
        raise ValueError('capture hook count mismatch')
    return score, states[0]


def validate(row, case, plan_hash):
    if row['uid'] != case['example']['uid'] or row['plan_sha256'] != plan_hash:
        raise ValueError('case provenance mismatch')
    if set(row['donor_scores']) != set(KINDS) or set(row['patched_scores']) != set(KINDS):
        raise ValueError('missing conditions')
    if not all(math.isfinite(s) for group in ('donor_scores', 'patched_scores') for s in row[group].values()):
        raise ValueError('non-finite score')
    base, self_score = row['donor_scores']['long'], row['patched_scores']['long']
    if not math.isclose(base, self_score, abs_tol=.05, rel_tol=.01) or (base > 0) != (self_score > 0):
        raise ValueError('self control failed')


def summarize(rows, cases):
    answers = np.array([c['example']['answer'] for c in cases])
    sign = np.where(answers, 1, -1)
    base = np.array([r['donor_scores']['long'] for r in rows])
    correct = (base > 0) == answers
    rng = np.random.default_rng(0)
    draws = np.concatenate([rng.choice(np.flatnonzero(answers == a), (5000, int(sum(answers == a))), replace=True)
                            for a in (False, True)], axis=1)
    effects = {}
    for kind in KINDS:
        scores = np.array([r['patched_scores'][kind] for r in rows])
        patched = (scores > 0) == answers
        delta = patched.astype(float) - correct
        effects[kind] = {'correct': int(sum(patched)), 'rescued': int(sum(patched & ~correct)),
                         'regressed': int(sum(~patched & correct)), 'accuracy_change': float(delta.mean()),
                         'accuracy_change_ci95': np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist(),
                         'signed_score_change': float(np.mean(sign * (scores - base)))}
    contrasts = {}
    same = np.array([r['patched_scores']['same'] for r in rows])
    for kind in ('opposite', 'neutral'):
        other = np.array([r['patched_scores'][kind] for r in rows])
        delta = (((same > 0) == answers).astype(float) - ((other > 0) == answers))
        contrasts[kind] = {'same_minus_control_accuracy': float(delta.mean()),
                          'ci95': np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist(),
                          'same_minus_control_signed_score': float(np.mean(sign * (same - other)))}
    return {'questions': len(rows), 'passes': 8 * len(rows), 'baseline_correct': int(sum(correct)),
            'short_same_correct': sum((r['donor_scores']['same'] > 0) == bool(a) for r, a in zip(rows, answers)),
            'short_opposite_correct_for_reversed_fact': sum((r['donor_scores']['opposite'] > 0) != bool(a) for r, a in zip(rows, answers)),
            'max_self_abs_error': max(abs(r['donor_scores']['long'] - r['patched_scores']['long']) for r in rows),
            'effects': effects, 'paired_contrasts': contrasts,
            'uncertainty': 'Pointwise 95% answer-stratified paired bootstrap, 5000 draws; secondary contrasts exploratory.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('runs/query-confirmation'))
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=20261007)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.n < 2 or args.n % 2:
        ap.error('n must be positive and even')
    plan = prepare(args.n, args.seed)
    print(f"Validated {args.n} new questions, {args.n * 8} passes, layer {LAYER}, exact question token matches", flush=True)
    if args.dry_run:
        return
    plan_hash = digest(json.dumps(plan, sort_keys=True).encode())
    with run_lock(args.out):
        path = args.out / 'plan.json'
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError('plan changed; use a new output directory')
        if not path.exists() and (args.out / 'cases').exists():
            raise ValueError('cannot adopt cases without plan')
        atomic_json(path, plan)
        (args.out / 'cases').mkdir(exist_ok=True)
        rows = {}
        for i, case in enumerate(plan['cases']):
            path = args.out / 'cases' / f'{i:04d}.json'
            if path.exists():
                row = json.loads(path.read_text())
                validate(row, case, plan_hash)
                rows[i] = row
        def status(state):
            atomic_json(args.out / 'status.json', {'state': state, 'completed_questions': len(rows),
                        'total_questions': args.n, 'saved_passes': len(rows) * 8,
                        'updated_at': datetime.now(timezone.utc).isoformat()})
        status('running')
        try:
            if len(rows) < args.n:
                if not torch.backends.mps.is_available():
                    raise RuntimeError('Apple GPU unavailable')
                model, tok = load_model(MODEL, 'mps', torch.bfloat16, revision=REVISION)
                ids = letter_ids(tok)
                for i, case in enumerate(plan['cases']):
                    if i in rows:
                        continue
                    inputs = {k: torch.tensor([p['input_ids']], device='mps') for k, p in case['prompts'].items()}
                    captures = {k: capture(model, inputs[k], ids, p['positions']) for k, p in case['prompts'].items()}
                    patched = {k: patched_score(model, inputs['long'], ids, LAYER,
                               case['prompts']['long']['positions'], captures[k][1]) for k in KINDS}
                    row = {'uid': case['example']['uid'], 'plan_sha256': plan_hash,
                           'donor_scores': {k: v[0] for k, v in captures.items()}, 'patched_scores': patched}
                    validate(row, case, plan_hash)
                    atomic_json(args.out / 'cases' / f'{i:04d}.json', row)
                    rows[i] = row
                    status('running')
                    print(f'{len(rows)}/{args.n} questions saved; self control passed', flush=True)
                del model
            summary = summarize([rows[i] for i in range(args.n)], plan['cases'])
            summary['limitations'] = plan['limitations']
            atomic_json(args.out / 'summary.json', summary)
            status('complete')
            print('Confirmation complete; summary.json saved', flush=True)
        except BaseException:
            status('interrupted')
            raise


if __name__ == '__main__':
    main()
