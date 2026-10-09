# Jev style single pass reasoning study

A preliminary study of one hop factual questions in JevK5 v0.3 4B, an open decision tuned model based on Qwen3.5. The project asks whether performance declines as transitive reasoning requires more linked facts, and where answer relevant information appears inside the model.

## Early findings

On the same 300 one hop questions, JevK5 accuracy was 100% with the answer giving fact alone, 99.3% with five premises, 92.7% with thirteen, and 75% with all 25 premises. The questions and relevant fact were preserved while subsets of distractor premises changed. Premise count, total context length, and relevant fact position vary together, so this is evidence of context sensitivity, not a measured multi hop depth ceiling.

On 100 fresh one hop questions with 25 premises, baseline accuracy was 71%. Replacing the layer 16 states at both question name positions with states from a short prompt containing the correct fact raised accuracy to 87% (17 mistakes fixed, one correct answer changed). A reversed fact donor yielded 50%; an unrelated fact donor yielded 72%. Self patch controls reproduced baseline scores. This suggests fact sensitive information at these positions affects answers. It does not identify the underlying computation.

A follow up localized this effect on the same 100 questions: layer 16 was strongest among layers 15 to 17; patching the first name alone raised accuracy to 81%, the second name alone to 75%, and both to 87%. Because this reuses the same cohort, it is localization, not an independent replication.

## One hop gate and protocol note

The original plan set an 85% one hop gate and presumed a lower score meant the prompt format was wrong. The measured score was 75% (95% interval 70 to 80%). Native prompt tokenization was checked; on 12 native runtime examples predictions matched. On a separate 48 example control sample, accuracy was 48/48 with the answer giving fact alone and 35/48 with the full premise set. The protocol amendment retains the failed gate and revises its diagnosis: proceed to the planned one to six hop behavioral curve for JevK5 and base Qwen3.5 4B with the same 25 premise prompts, report the low one hop baseline, and interpret the curve as performance on this distractor heavy task. No multi hop depth ceiling has been established yet.

## Reproducibility

The Python package contains the task generator, prompt renderer, model loading, scoring, hooks, and evaluation utilities. `scripts/` contains the selected extraction, context comparison, and activation patching workflows. `tests/` contains unit and integration checks.

Generated datasets, model activations, run outputs, figures, JSON records, model weights, and local environment files are intentionally excluded. To run the model workflows, install dependencies from `pyproject.toml`, ensure the pinned JevK5 model is available locally, then consult each script's `--help`. GPU workflows are resource intensive.

## Scope

These are results for one open model and one synthetic task family. They do not establish what the closed Jev model does internally. They do not yet show a multi hop reasoning mechanism or support claims about recurrence.
