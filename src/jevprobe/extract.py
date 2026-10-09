"""Run a model over a dataset; save per-layer residuals at decision/entity/premise positions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from jevprobe.data.transitive import N_DISTRACT, N_MAIN, N_PREMISES, load_jsonl
from jevprobe.hooks import capture, find_decoder_layers
from jevprobe.models import load_model, parse_model_ref
from jevprobe.prompts import render
from jevprobe.readout import letter_ids, letter_score
from jevprobe.tokens import positions

FP16_SAFE = 6e4  # float16 max is 65504; larger residual values would silently become inf


def check_storable(x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        raise ValueError("non-finite activation")
    if x.abs().max() > FP16_SAFE:
        raise ValueError(f"activation {float(x.abs().max()):.3g} overflows float16 storage")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    examples = load_jsonl(args.data)[: args.limit]
    repo, rev = parse_model_ref(args.model)
    model, tok = load_model(repo, args.device, getattr(torch, args.dtype), revision=rev)
    ids = letter_ids(tok)
    n_layers = len(find_decoder_layers(model)) + 1
    hidden = model.config.get_text_config().hidden_size
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mm = lambda name, *shape: np.lib.format.open_memmap(out / name, "w+", np.float16, (len(examples), n_layers, *shape))
    dec, ent, prem = mm("decision.npy", hidden), mm("entities.npy", N_MAIN + N_DISTRACT, hidden), mm("premises.npy", N_PREMISES, hidden)

    with open(out / "meta.jsonl", "w") as meta:
        for i, ex in enumerate(examples):
            r = render(ex)
            enc = tok(r.text, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
            order = ex.chain + ex.distractors
            pos = positions(r, enc["offset_mapping"][0].tolist(), order)
            resid, _ = capture(model, enc["input_ids"].to(args.device))
            check_storable(resid)
            dec[i] = resid[:, pos.decision].cpu().numpy()
            ent[i] = torch.stack([resid[:, idx].mean(1) for idx in pos.entities], dim=1).cpu().numpy()
            prem[i] = resid[:, pos.premises].cpu().numpy()
            meta.write(json.dumps({
                "uid": ex.uid, "split": ex.split, "d": ex.d, "answer": ex.answer,
                "model": args.model, "score": letter_score(model, resid[-1, pos.decision], ids),
                "max_abs": float(resid.abs().max()),
                "n_tokens": int(enc["input_ids"].shape[1]), "entity_order": order,
            }) + "\n")
            if i % 50 == 0:
                print(f"{i}/{len(examples)}", flush=True)
    for arr in (dec, ent, prem):
        arr.flush()


if __name__ == "__main__":
    main()
