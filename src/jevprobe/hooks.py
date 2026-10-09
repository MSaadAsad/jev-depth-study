"""Residual-stream capture that works across HF decoder layouts (incl. multimodal wrappers)."""
from __future__ import annotations

import torch
from torch import nn


def _decoder_layers_with_name(model) -> tuple[str, nn.ModuleList]:
    n = model.config.get_text_config().num_hidden_layers
    for name, mod in model.named_modules():
        if not (isinstance(mod, nn.ModuleList) and len(mod) == n and name.split(".")[-1] == "layers"):
            continue
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        if hasattr(parent, "embed_tokens") and hasattr(parent, "norm"):  # the text decoder, not a vision tower
            return name, mod
    raise RuntimeError("could not locate text decoder layers")


def find_decoder_layers(model) -> nn.ModuleList:
    return _decoder_layers_with_name(model)[1]


def find_final_norm(model) -> nn.Module:
    return find_text_decoder(model).norm


def find_text_decoder(model) -> nn.Module:
    """The text decoder, whose output includes the model's final normalization."""
    name, _ = _decoder_layers_with_name(model)
    parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
    return parent


@torch.no_grad()
def capture(model, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch of one. Returns resid [L+1, T, H] (0 = embeddings, k = decoder layer k output) and logits [T, V]."""
    layers = find_decoder_layers(model)
    store: dict[int, torch.Tensor] = {}

    def save(key: int):
        def hook(_module, _inputs, out):
            store[key] = (out[0] if isinstance(out, tuple) else out)[0].detach()
        return hook

    handles = [model.get_input_embeddings().register_forward_hook(save(0))]
    handles += [layer.register_forward_hook(save(k + 1)) for k, layer in enumerate(layers)]
    try:
        out = model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    resid = torch.stack([store[k] for k in range(len(layers) + 1)]).float()
    return resid, out.logits[0].float()
