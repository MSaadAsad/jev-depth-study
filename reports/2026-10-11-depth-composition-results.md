# JevK5 depth and causal patching results

**Status:** completed preliminary study · **Run date:** 2026-10-11 · **Scope:** JevK5 v0.3 and base Qwen3.5-4B on a synthetic transitive comparison task

## Research question

Does answer accuracy fall as the shortest relation chain grows, and do selected internal states causally move answers between matched prompts that require opposite answers? The study is intended to characterize candidate computation sites before considering recurrence. It does not implement recurrence or identify a complete reasoning circuit.

## Behavioral depth curve

Each prompt contains 25 shuffled premises: 23 edges in a 24-entity main chain and two edges in a three-entity distractor chain. The queried entities are separated by the specified shortest path length, *d*. We evaluated 300 balanced test questions per depth for each model, using the same prompts and answer readout. Accuracy intervals are ordinary example-bootstrap 95% intervals from 1,000 resamples.

| Shortest path (*d*) | JevK5 v0.3 | Base Qwen3.5-4B |
|---:|---:|---:|
| 1 | 75.0% [70.0, 80.0] | 74.0% [69.0, 79.0] |
| 2 | 72.7% [67.7, 77.3] | 60.3% [54.7, 65.7] |
| 3 | 55.7% [50.3, 61.0] | 54.0% [48.7, 59.7] |
| 4 | 52.0% [46.3, 58.0] | 53.0% [47.3, 58.7] |
| 5 | 49.3% [43.3, 55.0] | 53.0% [47.3, 58.7] |
| 6 | 51.7% [46.3, 57.7] | 48.3% [43.0, 54.0] |

![Accuracy by shortest relation-chain length for JevK5 and base Qwen3.5-4B.](figures/03-depth-performance.png)

Both models are close on one-hop questions and approach the 50% reference by three hops. JevK5 is higher on this sample at two hops; from three through six hops the point estimates are close to chance. This establishes a depth-related behavioral decline on this fixed 25-premise task. It does not isolate a general reasoning-depth ceiling: model, task, and prompt distribution are limited, and aggregate accuracy alone does not identify the failed computation.

## Counterfactual activation patching

We evaluated 1,200 token-aligned source/counterfactual pairs: 200 per source depth, balanced by answer. The counterfactual prompt requires the opposite answer and differs by a four-premise entity swap; its shortest path can differ from the source path. For each pair, source-run states were patched into the counterfactual run at layers 16 and 20 at five position groups: the final decision position, the two queried names, source-path bridge-entity mentions, source-path premise ends, and distractor-premise ends. Twelve sentinel pairs received self-patch checks; the run completed all validation checks.

The plotted outcome is the change in answer-signed logit margin from the unpatched counterfactual toward the source answer. Positive values mean the intervention moved the output toward the source answer. Intervals are pointwise, answer-stratified bootstrap intervals with 2,000 resamples; no multiplicity adjustment was applied.

![Answer-signed logit change under activation patching by layer and token position.](figures/04-patch-effects.png)

At layer 16, patching the queried-name positions produced the clearest answer-directed effect: mean changes were +0.983 logits at *d*=1 (95% interval [0.830, 1.151]) and +0.633 at *d*=2 ([0.496, 0.778]). The donor-answer rate rose descriptively from 34.0% to 60.0% at *d*=1 and from 41.5% to 58.0% at *d*=2. Effects were smaller at *d*=3–6. At layer 20, patching the final decision position shifted logits toward the source answer at each depth, with mean changes from +1.490 at *d*=1 to approximately +0.48–0.54 at *d*=3–6.

Patching bridge-entity and path-premise positions had changes near zero across depths; distractor-premise positions also served as a near-zero location control. This pattern localizes answer-sensitive information to selected positions in this intervention setup, especially the query-name positions at layer 16 and the decision position at layer 20. It does not establish that the model explicitly represents or composes an intermediate relation at those sites.

## Interpretation and limits

- The behavioral result is evidence of a sharp performance drop after two hops on this task, not a universal ceiling or proof of a particular reasoning failure.
- The patch result is causal evidence that interventions at the tested positions can alter answer logits. It does not show that the complete relation-composition algorithm has been found.
- Each counterfactual changes four premises and can have a different shortest path. Therefore these interventions do not isolate one premise or one computation step.
- Only layers 16 and 20 were tested in this study. The pointwise intervals are not corrected for the many layers, positions, and depths compared.
- Results cover one fine-tuned model and its base-model comparison on one synthetic task family; they do not establish generalization to other models or natural-language reasoning.

## What this means for the project

The first behavioral milestone is complete, and we now have evidence that this task becomes difficult beyond two hops. The patch study gives candidate answer-sensitive positions, but it did not demonstrate a causal role for the patched bridge-entity or path-premise states. Before introducing recurrence, the next useful milestone is a tighter two-hop composition experiment: preserve the queried relation while varying one premise at a time, compare individual and joint premise interventions, and test whether candidate states generalize across held-out names and templates. That would directly test integration rather than only answer sensitivity.

## Reproducibility and artifact scope

The local plotting command is `python scripts/plot_depth_composition.py`; it reads the aggregate run summaries under `runs/` and writes the figures in `reports/figures/`. The experiment code is in `scripts/compose_patch.py`, with the resumable launcher in `scripts/run_depth_composition.sh`.

The public repository contains this report, the aggregate charts, and the experiment/plotting code. Per-example predictions, activation vectors, run JSON/JSONL files, datasets, and model weights remain local and are not published.
