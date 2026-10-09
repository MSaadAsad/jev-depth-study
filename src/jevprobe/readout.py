"""JevK5 letter readout: logit(A = true) minus logit(B = false) at the decision position."""
from __future__ import annotations

import torch

from jevprobe.hooks import find_final_norm, find_text_decoder

from jevprobe.prompts import OPTION_LETTERS


def letter_ids(tokenizer) -> tuple[int, int]:
    ids = []
    for letter in OPTION_LETTERS:
        toks = tokenizer.encode(letter, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"{letter!r} is not a single token: {toks}")
        ids.append(toks[0])
    return ids[0], ids[1]


def true_minus_false(logits_last: torch.Tensor, ids: tuple[int, int]) -> float:
    return float(logits_last[ids[0]] - logits_last[ids[1]])


@torch.no_grad()
def letter_score(model, h_last: torch.Tensor, ids: tuple[int, int]) -> float:
    """logit(A) - logit(B) from the last layer's residual, in fp32 (bf16 logits are quantised ~0.1 at |logit| ~ 20)."""
    norm = find_final_norm(model)
    w = model.get_output_embeddings().weight
    normed = norm(h_last.to(device=w.device, dtype=w.dtype)).float()
    a, b = w[list(ids)].float()
    return float(normed @ (a - b))


@torch.inference_mode()
def forward_score(model, input_ids: torch.Tensor, ids: tuple[int, int]) -> float:
    """One forward pass, retaining no layer history and projecting only the answer letters.

    The text decoder already applies final normalization. Project its last-token
    state in fp32, matching letter_score without constructing [tokens, vocab] logits.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("forward_score requires a batch of one")
    output = find_text_decoder(model)(input_ids=input_ids, use_cache=False, return_dict=True)
    h = output.last_hidden_state[0, -1].float()
    head = model.get_output_embeddings()
    a, b = head.weight[list(ids)].float()
    score = h @ (a - b)
    if head.bias is not None:
        score = score + head.bias[ids[0]].float() - head.bias[ids[1]].float()
    if not torch.isfinite(score):
        raise ValueError("non-finite answer score")
    return float(score)
