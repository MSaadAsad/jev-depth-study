"""CPU-only exploratory analysis of the saved matched activations.

Use the final answer-letter head as a fixed logit lens, never as a trained probe.
Audit token identities before comparing representations. Bootstrap intervals are
pointwise/exploratory, not corrected for selecting layers after seeing the data.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from extract_matched import SLOTS, read_pair
from jevprobe.evaluate import atomic_json, digest
from summarize_context_sweep import wilson_interval


def bootstrap_weights(n, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, (n_boot, n))
    return np.array([np.bincount(row, minlength=n) for row in draws])


def bootstrap_auc(y, scores, weights):
    """Exact weighted AUROC for example-bootstrap samples, including score ties."""
    y, scores = np.asarray(y, dtype=bool), np.asarray(scores)
    order = np.argsort(scores, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]
    positive = np.add.reduceat(weights[:, order] * y[order], starts, axis=1)
    negative = np.add.reduceat(weights[:, order] * ~y[order], starts, axis=1)
    numerator = (positive * (np.cumsum(negative, axis=1) - .5 * negative)).sum(axis=1)
    denominator = positive.sum(axis=1) * negative.sum(axis=1)
    return np.divide(numerator, denominator, out=np.full(len(weights), np.nan), where=denominator > 0)


def balanced_margin(y, scores, weights=None):
    """Equal class weighting removes spurious group differences from letter bias."""
    y, scores = np.asarray(y, dtype=bool), np.asarray(scores)
    if weights is None:
        return .5 * (scores[y].mean(axis=0) - scores[~y].mean(axis=0))
    pos, neg = weights @ y, weights @ ~y
    return .5 * ((weights @ (scores * y)) / pos - (weights @ (scores * ~y)) / neg)


def ci(values):
    return [float(v) for v in np.nanquantile(values, [.025, .975])]


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=Path("runs/activations-matched"))
    args = ap.parse_args(argv)
    root = args.run
    manifest = json.loads((root / "manifest.json").read_text())
    if digest(Path(__file__).with_name("extract_matched.py").read_bytes()) != manifest["script_sha256"]:
        raise ValueError("extractor changed since this run")
    manifest_hash = digest(json.dumps(manifest, sort_keys=True).encode())
    shape = manifest["shape_per_pair"]
    projections, labels, long_correct, cosine, token_matches, valid_slots = [], [], [], [], [], []
    token_pairs = [Counter() for _ in SLOTS]
    for i, case in enumerate(manifest["cases"]):
        states, lens, positions, meta = read_pair(root / "pairs" / f"{i:04d}.npz", case, shape, manifest_hash)
        valid = np.all(positions >= 0, axis=0)
        matched = np.zeros(len(SLOTS), dtype=bool)
        for slot in np.flatnonzero(valid):
            ids = tuple(meta["input_ids"][c][positions[c, slot]] for c in range(2))
            matched[slot] = ids[0] == ids[1]
            token_pairs[slot][ids] += 1
        a, b = states[0].astype(np.float32), states[1].astype(np.float32)
        denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
        cos = np.clip(np.sum(a * b, axis=-1) / np.maximum(denom, 1e-12), -1, 1)
        cos[:, ~matched] = np.nan
        cosine.append(cos)
        projections.append(lens.astype(np.float64))
        labels.append(case["answer"])
        long_correct.append(case["long_correct"])
        token_matches.append(matched)
        valid_slots.append(valid)
        if (i + 1) % 100 == 0:
            print(f"validated {i+1}/{len(manifest['cases'])} saved pairs", flush=True)
    projection = np.stack(projections)
    y, correct, cosine, matches, valid = map(np.asarray, (labels, long_correct, cosine, token_matches, valid_slots))
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    groups = {"all": np.ones(len(y), dtype=bool), "long_correct": correct, "long_wrong": ~correct}
    audit = []
    for slot, name in enumerate(SLOTS):
        audit.append({"slot": name, "valid_pairs": int(valid[:, slot].sum()), "same_token_pairs": int(matches[:, slot].sum()),
                      "same_token_long_correct": int(np.sum(matches[:, slot] & correct)),
                      "same_token_long_wrong": int(np.sum(matches[:, slot] & ~correct)),
                      "token_pairs": [{"short_id": int(ids[0]), "long_id": int(ids[1]), "n": n} for ids, n in token_pairs[slot].most_common()]})
    atomic_json(out / "token_audit.json", {"slots": audit})
    decision_rows, geometry_rows, paired_rows = [], [], []
    for group_id, (group, mask) in enumerate(groups.items()):
        group_y = y[mask]
        weights = bootstrap_weights(int(mask.sum()), seed=group_id)
        group_projection = projection[mask, :, :, 0]
        final_prediction = group_projection[:, 1, -1] > 0
        for layer in range(shape[1]):
            boot_accuracy, boot_auc = [], []
            for context in range(2):
                scores = group_projection[:, context, layer]
                prediction = scores > 0
                is_correct = prediction == group_y
                acc_samples = (weights @ is_correct) / len(scores)
                auc_samples = bootstrap_auc(group_y, scores, weights)
                margins = balanced_margin(group_y, scores, weights)
                acc_ci, auc_ci, margin_ci = ci(acc_samples), ci(auc_samples), ci(margins)
                wilson = wilson_interval(int(is_correct.sum()), len(scores))
                agreement = prediction == final_prediction
                decision_rows.append({"group": group, "context_premises": (1, 25)[context], "layer": layer,
                                      "n": len(scores), "n_true": int(group_y.sum()), "accuracy": float(is_correct.mean()),
                                      "accuracy_boot_lo": acc_ci[0], "accuracy_boot_hi": acc_ci[1],
                                      "accuracy_wilson_lo": wilson[0], "accuracy_wilson_hi": wilson[1],
                                      "auroc": float(roc_auc_score(group_y, scores)), "auroc_lo": auc_ci[0], "auroc_hi": auc_ci[1],
                                      "class_balanced_margin": float(balanced_margin(group_y, scores)),
                                      "margin_lo": margin_ci[0], "margin_hi": margin_ci[1],
                                      "agreement_with_final_long_answer": float(agreement.mean())})
                boot_accuracy.append(acc_samples)
                boot_auc.append(auc_samples)
            diff_acc = ci(boot_accuracy[0] - boot_accuracy[1])
            diff_auc = ci(boot_auc[0] - boot_auc[1])
            paired_rows.append({"group": group, "layer": layer,
                                "accuracy_difference_short_minus_long": decision_rows[-2]["accuracy"] - decision_rows[-1]["accuracy"],
                                "accuracy_difference_lo": diff_acc[0], "accuracy_difference_hi": diff_acc[1],
                                "auroc_difference_short_minus_long": decision_rows[-2]["auroc"] - decision_rows[-1]["auroc"],
                                "auroc_difference_lo": diff_auc[0], "auroc_difference_hi": diff_auc[1]})
        for slot, name in enumerate(SLOTS):
            selected = mask & matches[:, slot]
            n = int(selected.sum())
            if n < 2:
                continue
            values = cosine[selected, :, slot]
            boot_cos = bootstrap_weights(n, seed=group_id) @ values / n
            for layer in range(shape[1]):
                lo, hi = ci(boot_cos[:, layer])
                geometry_rows.append({"group": group, "slot": name, "layer": layer, "n_same_token": n,
                                      "cosine_mean": float(values[:, layer].mean()), "cosine_lo": lo, "cosine_hi": hi})
    write_csv(out / "decision_layers.csv", decision_rows)
    write_csv(out / "paired_layer_differences.csv", paired_rows)
    write_csv(out / "geometry_same_token.csv", geometry_rows)
    highlighted = (16, 18, 19, 20, 24, 32)
    error_mask = ~correct
    layer20 = projection[:, 1, 20, 0] > 0
    final = projection[:, 1, -1, 0] > 0
    report = {
        "pairs": len(y), "long_wrong": int(error_mask.sum()), "long_correct": int(correct.sum()),
        "analysis_script_sha256": digest(Path(__file__).read_bytes()), "activation_manifest_sha256": manifest_hash,
        "highlighted_layers": [r for r in decision_rows if r["group"] == "all" and r["layer"] in highlighted],
        "layer20_vs_final": {"same_predictions": int(np.sum(layer20 == final)), "total": len(y),
                             "final_errors_already_wrong": int(np.sum((layer20 != y) & error_mask)),
                             "final_errors_total": int(error_mask.sum()),
                             "final_successes_wrong_at_20": int(np.sum((layer20 != y) & correct))},
        "token_audit_counts": [{k: v for k, v in r.items() if k != "token_pairs"} for r in audit],
        "candidate_patching_experiment": {
            "status": "proposal only; not executed",
            "primary_saved_layers": [16, 17, 18, 19, 20], "later_reference": [24],
            "positions": ["decision", "question_x", "question_y", "relevant_x", "relevant_y"],
            "source": "short-context state from the same question", "target": "long-context forward pass",
            "requirements": ["Use only same-token matches for entity-position interventions.",
                             "Include self-patch controls, matched-label unrelated-example patches, successful-example controls and a final-decision positive control.",
                             "Report both accuracy and signed-score changes; avoid division by tiny baseline score differences.",
                             "Treat layer selection as exploratory and confirm on a new matched dataset before making a general claim."]},
        "limitations": ["The fixed final norm/head is a logit lens, not an independently trained or calibrated decoder at intermediate layers.",
                        "Earlier information may be present but unreadable by this head. There is no established emergence layer.",
                        "Long-correct/wrong groups are defined by the final answer; final group differences are guaranteed by selection.",
                        "Equal class weighting is used for signed margins to avoid class-imbalance artifacts.",
                        "Matching token IDs does not match absolute token positions or separate position effects from surrounding evidence.",
                        "Confidence intervals are pointwise example bootstraps (1000 samples), not simultaneous or selection-adjusted intervals.",
                        "A one-hop context experiment cannot establish multi-hop reasoning or causal localization."]}
    atomic_json(out / "summary.json", report)
    print(json.dumps(report["layer20_vs_final"], indent=2))
    print(f"Saved analysis to {out}", flush=True)


if __name__ == "__main__":
    main()
