"""Load Qwen3.5-family checkpoints (base or JevK5-style merged) as text-only causal LMs."""
from __future__ import annotations

import torch
import transformers


def parse_model_ref(ref: str) -> tuple[str, str | None]:
    repo, _, rev = ref.partition("@")
    return repo, rev or None


def load_model(model_id: str, device: str = "mps", dtype: torch.dtype = torch.float32, revision: str | None = None):
    config = transformers.AutoConfig.from_pretrained(model_id, revision=revision)
    tok = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)
    cls = transformers.AutoModelForCausalLM
    if config.model_type in {"qwen3_5", "qwen3_5_text"}:  # same rule as jevk5.runtime
        cls, config = transformers.Qwen3_5ForCausalLM, config.get_text_config()
    # device_map loads straight onto the device, but segfaults for "mps" on this stack (exit 139),
    # so MPS loads on CPU and moves; .to() swaps tensors one parameter at a time.
    placement = {} if device == "mps" else {"device_map": {"": device}}
    model, info = cls.from_pretrained(model_id, revision=revision, config=config, dtype=dtype,
                                      output_loading_info=True, **placement)
    if info["missing_keys"]:
        raise RuntimeError(f"{model_id}: {len(info['missing_keys'])} weights missing, e.g. {info['missing_keys'][:3]}")
    return model.to(device).eval(), tok
