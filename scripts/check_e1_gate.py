"""Diagnose a failed depth-1 gate without changing the preregistered dataset.

Usage: .venv/bin/python scripts/check_e1_gate.py --run runs/e1-jevk5-v03
Controls are selected by relation, answer and premise phrasing, independently of errors.
The native-runtime comparisons use its bf16 projection; the study uses fp32.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch
from jevk5.runtime import JevK5

from jevprobe.data.transitive import load_jsonl
from jevprobe.evaluate import atomic_json, digest, source_digest
from jevprobe.models import load_model, parse_model_ref
from jevprobe.prompts import render
from jevprobe.readout import forward_score, letter_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    manifest = json.loads((run / "manifest.json").read_text())
    config = manifest["config"]
    data = Path(manifest["data_path"])
    assert digest(data.read_bytes()) == config["data_sha256"]
    assert source_digest() == config["source_sha256"]
    examples = {e.uid: e for e in load_jsonl(data) if e.d == 1}
    rows = {r["uid"]: r for r in map(json.loads, (run / "meta.jsonl").read_text().splitlines()) if r["d"] == 1}
    assert rows.keys() == examples.keys(), "finish the depth-1 gate first"
    groups = defaultdict(list)
    for ex in sorted(examples.values(), key=lambda e: e.uid):
        premise = next(p for p in ex.premises if {p.greater, p.lesser} == {ex.x, ex.y})
        groups[ex.rel, ex.answer, premise.template_id].append((ex, premise))

    repo, _ = parse_model_ref(config["model"])
    model, tok = load_model(repo, config["device"], getattr(torch, config["dtype"]), revision=config["revision"])
    ids = letter_ids(tok)
    # Reuse the loaded weights in the package's own readout implementation; its
    # constructor's device_map='mps' is known to crash on this machine.
    native = JevK5.__new__(JevK5)
    native.model, native.tok, native.device = model, tok, config["device"]
    native.slot_weight = model.get_output_embeddings().weight[list(ids)].detach().contiguous()
    native.graphs = {}
    controls, comparisons = [], []
    for key, items in sorted(groups.items()):
        for i, (ex, premise) in enumerate(items[:4]):
            short = render(replace(ex, premises=[premise])).text
            inputs = tok(short, add_special_tokens=False, return_tensors="pt")["input_ids"].to(config["device"])
            score = forward_score(model, inputs, ids)
            controls.append({"uid": ex.uid, "relation": ex.rel, "answer": ex.answer,
                             "template_id": premise.template_id, "full_score": rows[ex.uid]["score"],
                             "single_premise_score": score})
            if i == 0:
                full = render(ex).text
                payload = json.loads(full.split("<|im_start|>user\n")[1].split("<|im_end|>")[0])
                encoded = native.encode(payload["evidence"], payload["criterion"],
                                        [o["description"] for o in payload["options"]])
                assert encoded == tok.encode(full, add_special_tokens=False), "native prompt mismatch"
                logits = native.letter_logits(encoded, 2)
                native_score = float(logits[0] - logits[1])
                comparisons.append({"uid": ex.uid, "native_bf16_score": native_score,
                                    "study_fp32_score": rows[ex.uid]["score"]})
        print(f"checked {key}: {len(controls)} controls", flush=True)
    result = {
        "model": config["model"], "revision": config["revision"],
        "selection": "first four uids per (relation, answer, relevant premise template), independent of correctness",
        "n_controls": len(controls),
        "single_premise_accuracy": float(np.mean([(r["single_premise_score"] > 0) == r["answer"] for r in controls])),
        "matched_full_prompt_accuracy": float(np.mean([(r["full_score"] > 0) == r["answer"] for r in controls])),
        "native_tokenization_matches": True, "n_native_comparisons": len(comparisons),
        "native_prediction_agreement": float(np.mean([(r["native_bf16_score"] > 0) == (r["study_fp32_score"] > 0) for r in comparisons])),
        "max_native_score_difference": max(abs(r["native_bf16_score"] - r["study_fp32_score"]) for r in comparisons),
        "controls": controls, "native_comparisons": comparisons,
    }
    atomic_json(run / "gate_diagnostics.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("controls", "native_comparisons")}, indent=2))


if __name__ == "__main__":
    main()
