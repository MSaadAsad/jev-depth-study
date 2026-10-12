# JevK5 depth and composition study

A preliminary mechanistic-interpretability study of JevK5 v0.3, an open decision-tuned model based on Qwen3.5-4B, and its base-model comparison. We measure performance on synthetic transitive questions with one to six linked facts, then use activation patching to test whether selected internal states causally move answers between matched prompts.

## Current results

On 300 balanced test questions per depth, both models are near 75% on one-hop questions and close to chance by three hops on a fixed 25-premise task. JevK5 scores 72.7% at two hops and Qwen3.5-4B scores 60.3%; the point estimates are close to chance at depths three through six.

![Accuracy by relation-chain length.](reports/figures/03-depth-performance.png)

In a 1,200-pair counterfactual patch study, layer-16 states at the two question-name positions shifted the answer toward the donor prompt at one and two hops. Layer-20 decision-position patches also shifted answers. Patching source-path bridge mentions or premise ends had little effect in the tested setup.

![Causal patching effects by layer and token position.](reports/figures/04-patch-effects.png)

These results show a behavioral drop on this task and answer-sensitive internal states. They do not identify a complete multi-hop computation, establish a general reasoning-depth ceiling, or show that a recurrent architecture would help. The counterfactual pairs change four premises and can differ in hop distance; patch results are position- and layer-specific, and intervals are pointwise without a multiplicity correction.

The full report, including uncertainty intervals, methods, and interpretation, is [here](reports/2026-10-11-depth-composition-results.md).

## Earlier one-hop observations

On the same 300 one-hop questions, JevK5 scored 100% with the answer-giving fact alone, 99.3% with five premises, 92.7% with thirteen, and 75% with all 25 premises. Premise count, total context length, and relevant-fact position vary together, so this is evidence of context sensitivity, not an isolated context-length effect.

On 100 fresh one-hop questions with 25 premises, a layer-16 patch at both question-name positions raised accuracy from 71% to 87% when the donor contained the correct fact. A reversed-fact donor yielded 50%; an unrelated-fact donor yielded 72%. This suggests fact-sensitive information at those positions affects answers; it does not identify the underlying computation.

## Reproduction

The package contains task generation, prompt rendering, model loading, scoring, hooks, and evaluation utilities. `scripts/compose_patch.py` runs the streaming six-depth patch study; `scripts/run_depth_composition.sh` resumes the behavior curves and patch study in tmux. `scripts/plot_depth_composition.py` renders charts from local aggregate run summaries.

Run data, datasets, activations, model weights, and local environment files are intentionally excluded. Reproducing the run requires the 25-premise test dataset and locally available pinned model revisions; GPU workflows are resource intensive. The committed report and figures contain aggregate results only. The local plotting script expects saved summaries under `runs/`.

## Scope

Results cover these model revisions and one synthetic task family. They do not establish what closed Jev models do internally, prove a symbolic intermediate representation, or support claims about recurrence.
