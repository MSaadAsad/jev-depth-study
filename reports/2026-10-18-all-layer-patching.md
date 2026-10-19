# All-32-layer activation-patching screen

Completed 18 October 2026. This extends the initial selected-layer study to every decoder block of JevK5 v0.3 and base Qwen3.5-4B, using a fresh, matched-depth cohort. It is an exploratory causal localization study, not a recurrence evaluation.

## Main finding

Layer 12 has the largest mean question-name intervention effect at one and two hops in both checkpoints. The effect measures a change in the counterfactual run's source-answer logit margin. It is not an accuracy gain, a probability-point change, or necessarily a flipped prediction.

| Checkpoint | Hops | Layer with largest mean question-name effect | Mean margin change | Pointwise 95% interval |
|---|---:|---:|---:|---:|
| JevK5 v0.3 | 1 | 12 | +1.495 | [+0.917, +2.081] |
| JevK5 v0.3 | 2 | 12 | +0.825 | [+0.450, +1.218] |
| JevK5 v0.3 | 3 | 12 | +0.179 | [-0.053, +0.422] |
| JevK5 v0.3 | 4 | 3 | +0.015 | [-0.002, +0.035] |
| Qwen3.5-4B | 1 | 12 | +1.304 | [+0.840, +1.786] |
| Qwen3.5-4B | 2 | 12 | +0.328 | [+0.039, +0.595] |
| Qwen3.5-4B | 3 | 8 | +0.159 | [+0.009, +0.314] |
| Qwen3.5-4B | 4 | 6 | +0.018 | [-0.008, +0.045] |

The ranking is descriptive on the screening cohort. The intervals above do not test whether one layer is significantly stronger than another. Effects at three and four hops are less consistent. Neither checkpoint's four-hop peak excludes zero.

## Design and intervention

- 40 answer-balanced source/counterfactual pairs at each of depths 1, 2, 3 and 4, for 160 pairs per checkpoint. Both models use identical formatted prompts, A/B choice labels and next-token scoring.
- Every prompt contains 25 shuffled premises. Paired prompts have opposite correct answers and equal hop distance. Queried entities are interior to the full chain.
- Swapping an entity's chain position changes four premise records. This is not a single-fact causal intervention.
- One-based decoder-block numbering. First screen: full-attention blocks 4, 8, 12, 16, 20, 24, 28, 32. Complementary screen: the other 24 Gated DeltaNet blocks, on the same paired cohort.
- The question-name intervention replaces both queried names together at their question occurrences, patching only the last overlapping token per name. A multi-token name is not patched as a whole span.
- Other groups: final decision position, source-path bridge mentions, source-path premise endings, disconnected distractor premise endings. The bridge group is absent at one hop. Bridge mentions map to the same entity names; path-premise endings map to corresponding premise slots.
- Replace residual-stream states at block outputs, then recompute downstream computation. Positive signed effects mean movement toward the source prompt's correct answer.
- Effects average all 40 pairs per depth, regardless of baseline correctness. Answer-stratified bootstrap intervals are pointwise 95% intervals, without a multiple-comparison correction.

Across both models, all layers and all applicable position groups, the screen contains **48,640 primary patches**: 40 pairs per depth × (4 + 5 + 5 + 5 applicable groups across the four depths) × 32 layers × 2 checkpoints. This count excludes baselines and control interventions.

## Implementation checks

Each checkpoint has eight self-patch sentinel pairs and two native/full-forward equivalence sentinel pairs spanning the complementary screens. Self patches reproduced their original scores; cached-prefix interventions matched native full-forward interventions on the sentinels. The layer-32 decision-position positive control passed. Repeated source and counterfactual baselines agreed exactly between the two screens.

The prefix implementation holds one model and one pair at a time, reuses the unaffected prefix, and writes resumable scalar scores rather than activation dumps. These checks support the intervention implementation, not a particular relational-composition mechanism.

## Interpretation

The sweep identifies answer-sensitive states and a candidate region for a recurrence prototype. It does not establish that relations are stored or combined at layer 12. Four premise records differ across each pair, the layer peak was selected on this cohort, and the short-chain effect does not establish useful multi-hop composition. The original layer-16/20 probe used a different, larger cohort and did not search for the network-wide maximum.

Coverage of all 32 layers is complete. Independent confirmation, single-premise interventions and recurrence are distinct possible next studies. None is reported as completed here.

## Revisions and reproducibility

- JevK5: `alibiserikbay/JevK5@c4f7fdb3aeab5582336406e78d3bef11bf98833d`.
- Qwen: `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Runtime: torch 2.14.1, transformers 5.18.0, numpy 2.4.6, jevk5 0.3.3; Apple MPS, bfloat16.
- Fresh pair-generation seed: `20261016`. Preparation verifies balanced labels, four edited records, equal hop distances, token mapping and interior endpoint guards before model evaluation.
- `scripts/prepare_layer_followup.py` prepares the fresh cohort using locally cached tokenizers. `scripts/screen_layers.py` evaluates a specified layer set. `scripts/prepare_remaining_layers.py` validates the completed first screen before preparing the complement. `scripts/combine_layer_screens.py` checks matching provenance, repeated baselines and disjoint full coverage before combining effects.

Plans, JSON/JSONL files, prompt-level results, model weights and local run outputs are deliberately excluded from this repository. The scripts produce these locally when reproducing the study.
