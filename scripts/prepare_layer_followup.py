"""Prepare a fresh, same-hop layer screen using cached tokenizers only; no model weights."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')

from transformers import AutoTokenizer
import compose_patch as compose
from jevprobe.data.transitive import Example, valid_ranks
from jevprobe.evaluate import atomic_json, digest
from screen_layers import DEFAULT_LAYERS, MODELS


def audit_pair(source, target):
    if source.d != target.d or source.answer == target.answer:
        raise ValueError('Counterfactual must preserve depth and flip the answer')
    if (source.x, source.y, source.rel, source.inv) != (target.x, target.y, target.rel, target.inv):
        raise ValueError('Question changed')
    for ex in (source, target):
        if len(ex.premises) != 25 or not valid_ranks(ex.rank_x, ex.rank_y):
            raise ValueError('Full context and endpoint guards required')
        if ex.answer != (ex.rank_x < ex.rank_y) or abs(ex.rank_x - ex.rank_y) != ex.d:
            raise ValueError('Incorrect label or depth')
        for name in (ex.x, ex.y):
            greater = sum(p.greater == name for p in ex.premises)
            lesser = sum(p.lesser == name for p in ex.premises)
            if (greater, lesser) != (1, 1):
                raise ValueError('Queried-entity endpoint shortcut')
    if sum(a != b for a, b in zip(source.premises, target.premises)) != 4:
        raise ValueError('Expected exactly four edited premise records')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('data/layer-followup/plan.json'))
    ap.add_argument('--n-per-depth', type=int, default=40)
    ap.add_argument('--seed', type=int, default=20261016)
    args = ap.parse_args()
    if args.n_per_depth < 2 or args.n_per_depth % 2:
        ap.error('n-per-depth must be positive and even')
    plan = compose.prepare(args.n_per_depth, args.seed, ds=range(1, 5), fixed_depth=True)
    qrepo, qrev = MODELS['qwen']
    qtok = AutoTokenizer.from_pretrained(qrepo, revision=qrev, local_files_only=True)
    jtok = AutoTokenizer.from_pretrained(compose.MODEL, revision=compose.REVISION, local_files_only=True)
    seen = set()
    prior = set()
    for path in (Path('runs/composition-patching/plan.json'), Path('runs/query-confirmation/plan.json')):
        if path.exists():
            for case in json.loads(path.read_text()).get('cases', []):
                prior.update(case.get('prompt_hashes', []))
    for case in plan['cases']:
        for key in ('source', 'counterfactual'):
            ex = Example.from_dict(case[key])
            a, b = compose.layout(ex, jtok), compose.layout(ex, qtok)
            if a['input_ids'] != b['input_ids'] or a['groups'] != b['groups']:
                raise ValueError('Checkpoint tokenizers do not share the prepared token mapping')
            sha = a['prompt_sha256']
            if sha in seen or sha in prior:
                raise ValueError('Duplicate or previously patched prompt')
            seen.add(sha)
        audit_pair(Example.from_dict(case['source']), Example.from_dict(case['counterfactual']))
        case['uid'] = f'layer-followup-{args.seed}-{case["uid"]}'
    plan.update(layers=list(DEFAULT_LAYERS),
                design='Fresh balanced 1–4-hop pairs, 25 premises, equal donor/target depth; eight-layer exploratory screen.',
                tokenizer_checkpoints=MODELS,
                limitations='Four premise records change. Shared task vocabulary/templates; no new-domain generalization claim. Independent confirmation required.',
                preparation_script_sha256=digest(Path(__file__).read_bytes()))
    if args.out.exists() and json.loads(args.out.read_text()) != json.loads(json.dumps(plan)):
        raise ValueError('Existing plan differs; choose a new path')
    if not args.out.exists():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.out, plan)
    print(json.dumps({'prepared': str(args.out), 'pairs': len(plan['cases']),
                      'per_depth': dict(Counter(c['d'] for c in plan['cases'])),
                      'layers': plan['layers'], 'checkpoint_tokenizers_verified': True,
                      'model_weights_loaded': False}))


if __name__ == '__main__':
    main()
