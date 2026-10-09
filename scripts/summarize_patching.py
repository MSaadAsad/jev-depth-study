"""Summarize the selected patch pilot; estimates are exploratory, not population accuracy."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from patch_matched import read_results, check_control
from jevprobe.evaluate import atomic_json, digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, default=Path('runs/patch-pilot'))
    args = ap.parse_args()
    plan = json.loads((args.run / 'plan.json').read_text())
    rows = read_results(args.run / 'results.jsonl', plan['jobs'], digest(json.dumps(plan, sort_keys=True).encode()))
    if len(rows) != len(plan['jobs']):
        raise ValueError('pilot incomplete')
    cohort = {c['uid']: c for c in plan['cohort']}
    control_errors = []
    for row in rows.values():
        j = row['job']
        base = rows[f"{j['uid']}|baseline|0|none"]['score']
        check_control(j, row['score'], cohort[j['uid']], base)
        if j['kind'] in ('self', 'positive'):
            expected = base if j['kind'] == 'self' else cohort[j['uid']]['expected_scores'][0]
            control_errors.append(abs(row['score'] - expected))
    output = []
    rng = np.random.default_rng(0)
    for kind in ('same_question', 'unrelated_same_answer'):
        for layer in plan['layers']:
            for group in plan['groups']:
                for correct in (False, True):
                    selected = [r for r in rows.values() if r['job']['kind'] == kind and r['job']['layer'] == layer
                                and r['job']['group'] == group and cohort[r['job']['uid']]['long_correct'] == correct]
                    if not selected:
                        continue
                    changes, outcomes, labels = [], [], []
                    for r in selected:
                        uid = r['job']['uid']
                        sign = 1 if cohort[uid]['answer'] else -1
                        changes.append(sign * (r['score'] - rows[f'{uid}|baseline|0|none']['score']))
                        outcomes.append((r['score'] > 0) == cohort[uid]['answer'])
                        labels.append(cohort[uid]['answer'])
                    changes = np.array(changes)
                    # Resample questions separately within the two answer strata.
                    draws = np.concatenate([rng.choice(np.flatnonzero(np.array(labels) == answer),
                                                       (2000, labels.count(answer)), replace=True)
                                            for answer in (False, True)], axis=1)
                    lo, hi = np.quantile(changes[draws].mean(axis=1), [.025, .975])
                    output.append(dict(kind=kind, layer=layer, group=group, originally_correct=correct,
                                       n=len(selected), patched_correct=sum(outcomes),
                                       mean_signed_change=float(changes.mean()), ci_low=float(lo), ci_high=float(hi)))
    paired = []
    for layer in plan['layers']:
        for correct in (False, True):
            selected = [c for c in cohort.values() if c['long_correct'] == correct]
            differences = np.array([(1 if c['answer'] else -1) *
                (rows[f"{c['uid']}|same_question|{layer}|decision"]['score'] -
                 rows[f"{c['uid']}|unrelated_same_answer|{layer}|decision"]['score']) for c in selected])
            labels = np.array([c['answer'] for c in selected])
            draws = np.concatenate([rng.choice(np.flatnonzero(labels == answer),
                                    (2000, int(sum(labels == answer))), replace=True)
                                    for answer in (False, True)], axis=1)
            lo, hi = np.quantile(differences[draws].mean(axis=1), [.025, .975])
            paired.append(dict(layer=layer, originally_correct=correct, n=len(selected),
                               same_question_minus_unrelated=float(differences.mean()),
                               ci_low=float(lo), ci_high=float(hi)))
    with (args.run / 'effects.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    atomic_json(args.run / 'summary.json', {'complete': True, 'passes': len(rows), 'questions': len(cohort),
                'max_control_abs_error': max(control_errors), 'effects': output, 'paired_decision_comparisons': paired,
                'limitations': 'Selected 24-question pilot; pointwise stratified bootstrap intervals; no multiplicity correction. Same-answer donors can transfer answer information. Cross-context interventions do not establish a reasoning-depth ceiling.'})
    print('Summary saved: runs/patch-pilot/summary.json and effects.csv', flush=True)


if __name__ == '__main__':
    main()
