"""Write a concise report and two figures from the completed paired analysis.

Also audit an endpoint-role shortcut in the frozen prompts. This post-hoc
structural audit is separate from the pre-specified behavioral comparisons.
"""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/jev-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/context-factorial'
NAMES = {'jevk5': 'JevK5 v0.3', 'qwen35-4b': 'Qwen3.5-4B'}
COLORS = {'jevk5': '#0000ee', 'qwen35-4b': '#4b5563'}


def endpoint_prediction(example):
    """Use local greater/lesser roles only; abstain if both endpoints are ambiguous."""
    votes = []
    for name, is_x in ((example['x'], True), (example['y'], False)):
        greater = any(p['greater'] == name for p in example['premises'])
        lesser = any(p['lesser'] == name for p in example['premises'])
        if greater != lesser:
            votes.append(greater if is_x else lesser)
    return votes[0] if votes and all(v == votes[0] for v in votes) else None


def shortcut_audit():
    examples = [json.loads(line) for line in (ROOT / 'data/context-factorial/test.jsonl').read_text().splitlines()]
    rows = []
    for depth in range(1, 5):
        for dose in (0, 4, 12, 24):
            cell = [e for e in examples if e['d'] == depth and e['uid'].endswith(f'-x{dose:02d}')]
            predictions = [(endpoint_prediction(e), e['answer']) for e in cell]
            covered = [(p, y) for p, y in predictions if p is not None]
            rows.append({'depth': depth, 'off_path_premises': dose, 'n': len(cell),
                         'covered': len(covered), 'correct_when_covered': sum(p == y for p, y in covered)})
    return rows


def save(fig, name):
    for suffix in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_conditions(report):
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7), sharex=True, sharey=True)
    for depth, ax in enumerate(axes.flat, 1):
        for model in NAMES:
            rows = sorted((r for r in report['conditions'] if r['model'] == model and r['depth'] == depth),
                          key=lambda r: r['off_path_premises'])
            x = np.array([r['off_path_premises'] for r in rows])
            y = np.array([r['accuracy'] * 100 for r in rows])
            ci = np.array([r['wilson95'] for r in rows]).T * 100
            ax.errorbar(x, y, yerr=np.maximum(0, [y - ci[0], ci[1] - y]), color=COLORS[model],
                        marker='o' if model == 'jevk5' else 's', linestyle='-' if model == 'jevk5' else '--',
                        lw=1.9, markersize=4.5, capsize=3, label=NAMES[model])
        ax.axhline(50, color='#9ca3af', ls=':', lw=1)
        ax.set(title=f'{depth} hop' + ('s' if depth > 1 else ''), xticks=(0, 4, 12, 24), ylim=(35, 103))
        ax.grid(axis='y', alpha=.15)
    for ax in axes[:, 0]:
        ax.set_ylabel('Accuracy (%)')
    for ax in axes[-1]:
        ax.set_xlabel('Off-path premises added')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, .98))
    fig.suptitle('Relational accuracy under added context', x=.08, ha='left', fontsize=15, y=1.02)
    fig.text(.08, .01, '300 paired questions per condition; bars: pointwise 95% Wilson intervals. Dotted line: 50% chance.', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .91))
    save(fig, 'accuracy-by-context')


def plot_effects(report):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    for ax, model in zip(axes[:2], NAMES):
        rows = sorted((r for r in report['paired_context_effects'] if r['model'] == model and r['contrast'] == '24-minus-0'), key=lambda r: r['depth'])
        x = np.array([r['estimate'] * 100 for r in rows])
        ci = np.array([r['ci95'] for r in rows]).T * 100
        ax.errorbar(x, [r['depth'] for r in rows], xerr=np.maximum(0, [x - ci[0], ci[1] - x]),
                    fmt='o', color=COLORS[model], capsize=3)
        ax.set_title(NAMES[model], fontsize=11)
        ax.set_xlabel('Accuracy change (pp)')
    rows = sorted((r for r in report['difference_in_context_effect_by_model'] if r['contrast'] == '24-minus-0'), key=lambda r: r['depth'])
    x = np.array([r['estimate'] * 100 for r in rows])
    ci = np.array([r['ci95'] for r in rows]).T * 100
    axes[2].errorbar(x, [r['depth'] for r in rows], xerr=np.maximum(0, [x - ci[0], ci[1] - x]), fmt='o', color='#0000ee', capsize=3)
    axes[2].set(title='Difference in context effect', xlabel='JevK5 minus Qwen (pp)')
    effect_intervals = [r['ci95'] for r in report['paired_context_effects'] if r['contrast'] == '24-minus-0']
    common_left = min(ci[0] for ci in effect_intervals) * 100 - 3
    for ax in axes[:2]:
        ax.set_xlim(common_left, 3)
    for ax in axes:
        ax.axvline(0, color='#9ca3af', ls=':', lw=1)
        ax.set_yticks(range(1, 5))
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Hops')
    axes[0].invert_yaxis()
    fig.suptitle('Effect of adding 24 off-path premises', x=.08, ha='left', fontsize=15, y=1.02)
    fig.text(.08, .01, '10,000 paired graph/question bootstrap resamples; pointwise 95% intervals. Positive difference = smaller loss for JevK5.', fontsize=8.5)
    fig.tight_layout(rect=(0, .055, 1, .95))
    save(fig, 'paired-context-effects')


def main():
    report = json.loads((OUT / 'summary.json').read_text())
    if len(report['conditions']) != 32 or any(r['n'] != 300 for r in report['conditions']):
        raise ValueError('Expected a completed 32-cell paired analysis')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#9ca3af', 'text.color': '#1f2937',
                         'axes.labelcolor': '#374151', 'savefig.facecolor': 'white'})
    plot_conditions(report)
    plot_effects(report)
    audit = shortcut_audit()
    with (OUT / 'endpoint-shortcut-audit.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(audit[0])); w.writeheader(); w.writerows(audit)
    cells = {(r['model'], r['depth'], r['off_path_premises']): r for r in report['conditions']}
    pct = lambda v: f'{v * 100:.1f}%'
    lines = ['# Paired depth × context experiment', '',
             'Both models completed 4,800 evaluations: 300 balanced questions at each of four depths and four context doses. Both checkpoints received identical prompts. All comparisons below use the frozen cohort.', '',
             '## Main observations', '']
    lines.append(f"- Both models scored **100%** on clean one-hop questions. With 24 off-path facts, accuracy fell to **{pct(cells['jevk5', 1, 24]['accuracy'])}** for JevK5 and **{pct(cells['qwen35-4b', 1, 24]['accuracy'])}** for Qwen. The relationship is explicitly stated in these one-hop prompts.")
    for model in NAMES:
        clean, long = cells[model, 4, 0]['accuracy'], cells[model, 4, 24]['accuracy']
        lines.append(f"- At four hops, **{NAMES[model]}** scored **{pct(clean)}** with only path facts and **{pct(long)}** with 24 off-path facts added.")
    lines += ['- These are context-dependent behavioral differences. The clean chains expose a local endpoint cue, so high clean accuracy does not establish a composition mechanism; see the structural audit below.', '',
             '## Accuracy', '', '| Model | Hops | 0 off-path facts | 4 | 12 | 24 |', '|---|---:|---:|---:|---:|---:|']
    for model in NAMES:
        for depth in range(1, 5):
            values = [pct(cells[model, depth, dose]['accuracy']) for dose in (0, 4, 12, 24)]
            lines.append(f"| {NAMES[model]} | {depth} | " + ' | '.join(values) + ' |')
    lines += ['', '![Accuracy across context doses and hop depths](accuracy-by-context.png)', '',
              '## Primary paired contrasts', '',
              'Change in accuracy from 0 to 24 off-path facts, in percentage points. Negative values mean accuracy fell.', '',
              '| Model | Hops | Change (pp) | Paired 95% interval |', '|---|---:|---:|---:|']
    for row in report['paired_context_effects']:
        if row['contrast'] == '24-minus-0':
            lo, hi = row['ci95']
            lines.append(f"| {NAMES[row['model']]} | {row['depth']} | {row['estimate'] * 100:+.1f} | [{lo * 100:+.1f}, {hi * 100:+.1f}] |")
    lines += ['', 'Positive between-model differences below mean JevK5 lost less accuracy under added context; negative differences mean it lost more.', '',
              '| Hops | JevK5 context effect − Qwen context effect (pp) | Paired 95% interval |', '|---:|---:|---:|']
    for row in report['difference_in_context_effect_by_model']:
        if row['contrast'] == '24-minus-0':
            lo, hi = row['ci95']
            lines.append(f"| {row['depth']} | {row['estimate'] * 100:+.1f} | [{lo * 100:+.1f}, {hi * 100:+.1f}] |")
    lines += ['', '![Paired context effects and differences between checkpoints](paired-context-effects.png)', '',
              '## Interpretation and a structural limitation', '',
              'The contrasts measure sensitivity to added off-path facts under these prompts. They do not isolate semantic interference from token length, evidence position, or changes in available structural cues. This is factual context interference, not an adversarial instruction-poisoning experiment. The dose response is not strictly monotonic: some four-fact conditions improve descriptively over the zero-fact baseline.', '',
              'A post-hoc audit checks an important shortcut: if a queried entity appears only on the greater side or only on the lesser side of its premises, its endpoint role can reveal the answer in this generated chain family. With no off-path facts, the queried entities are endpoints of the presented path. This local cue does not require composing the whole path. The audit uses parsed premise directions and abstains when neither queried entity has an unambiguous role; it does not establish that either model uses the cue.', '',
              'For example, in `A > B, B > C`, A appears only as greater and C only as lesser. Adding `X > A` and `C > Y` removes those local cues while preserving the answer to A > C. The experiment therefore changes available shortcuts along with context length and fact position.', '',
              '| Hops | 0-fact cue coverage | 4-fact coverage | 12-fact coverage | 24-fact coverage |', '|---:|---:|---:|---:|---:|']
    for depth in range(1, 5):
        values = [next(r for r in audit if r['depth'] == depth and r['off_path_premises'] == dose) for dose in (0, 4, 12, 24)]
        lines.append(f'| {depth} | ' + ' | '.join(pct(r['covered'] / r['n']) for r in values) + ' |')
    lines += ['', 'Consequently, high clean-context accuracy does not establish a multi-hop composition mechanism or a pure measure of reasoning depth. The original distracted-task failures also cannot establish a fixed reasoning-depth ceiling. Checkpoint differences cannot by themselves identify which training change caused them.', '',
              '## Methods and validation', '',
              '- 4 depths × 4 off-path-fact doses × 300 items × 2 models = 9,600 evaluations.',
              '- Same base question and nested off-path-fact sets across context doses; same prompts across models.',
              '- Total premises = hop depth + off-path dose. The 24-fact condition therefore contains 25–28 premises; only its one-hop cell exactly reproduces the original 25-premise setup.',
              '- Each depth contains 300 distinct source graphs. Graph/base question is the resampling unit; context and checkpoint comparisons preserve pairing.',
              '- Answers are forced-choice A/B decisions from the sign of the final-position logit difference, using the same prompt/readout convention for both checkpoints. No reasoning tokens are generated.',
              '- Accuracy intervals are Wilson intervals. Paired contrasts use 10,000 bootstrap resamples. All intervals are pointwise; no multiple-comparison correction was applied.',
              '- Off-path facts include main-chain edges, a separate distractor chain, and disconnected fillers where needed. Counts and positions are retained in the layout manifest.',
              '- The endpoint-cue audit was added after inspecting the completed JevK5 results while Qwen evaluation was still in progress; it is exploratory and does not alter the planned primary contrasts.',
              '- Final manifest, prompt hashes and complete unique item coverage were checked by the paired summarizer.', '']
    diagnostic = report['qwen_ab_readout_diagnostic']
    lines += [f"Qwen A/B diagnostic: **{diagnostic['correct']}/{diagnostic['n']}** on the preselected balanced clean one-hop subset. This diagnoses the combined prompt/readout/direct-retrieval setup rather than isolating answer-format following.", '', 'Pinned revisions:', '']
    for model, revision in report['model_revisions'].items():
        lines.append(f'- {NAMES[model]}: `{revision}`')
    lines += ['', f"Context dataset SHA-256: `{report['design']['dataset_sha256']}`.",
              f"Source E1 dataset SHA-256: `{report['design']['source_sha256']}`. Context sampling seed: `{report['design']['seed']}`."]
    lines += ['', 'Reproduce the completed analysis and figures:', '', '```sh',
              '.venv/bin/python scripts/summarize_context_factorial.py',
              '.venv/bin/python scripts/report_context_factorial.py', '```', '',
              '## Project status', '',
              'The depth × context behavioral follow-up is complete. The subsequent all-32-layer localization screen is also complete; see [its report](../2026-10-18-all-layer-patching.md). Independent confirmation and controlled premise-composition tests remain possible follow-ups, not completed findings. No recurrence has been implemented or validated by this experiment.', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    print(OUT / 'REPORT.md')


if __name__ == '__main__':
    main()
