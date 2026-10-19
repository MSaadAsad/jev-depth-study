"""Batch-one residual patching with reusable, unchanged counterfactual prefixes."""
from contextlib import contextmanager

import torch

from jevprobe.hooks import find_decoder_layers
from jevprobe.readout import forward_score
from patch_matched import patched_score


@contextmanager
def reuse_prefix(blocks, cached, layer):
    originals, missing = [], object()
    try:
        for i in range(layer):
            block = blocks[i]
            originals.append((block, vars(block).get('forward', missing)))
            def saved_forward(*_args, state=cached[i + 1], **_kwargs):
                return state
            block.forward = saved_forward
        yield
    finally:
        for block, original in originals:
            if original is missing:
                delattr(block, 'forward')
            else:
                block.forward = original


class PrefixPatcher:
    def __init__(self, model, ids):
        if model.training:
            raise ValueError('Prefix reuse requires evaluation mode')
        self.model, self.ids = model, ids
        self.blocks = find_decoder_layers(model)

    @torch.inference_mode()
    def capture(self, inputs):
        states, handles = {}, []
        def save(layer):
            def hook(_m, _i, output):
                if not isinstance(output, torch.Tensor):
                    raise TypeError('Expected tensor decoder-block output')
                states[layer] = output.detach().clone()
            return hook
        try:
            for i, block in enumerate(self.blocks, 1):
                handles.append(block.register_forward_hook(save(i)))
            score = forward_score(self.model, inputs, self.ids)
        finally:
            for handle in handles:
                handle.remove()
        return score, states

    def patch(self, inputs, cached, layer, positions, values):
        if inputs.shape[0] != 1:
            raise ValueError('Only batch-one prefix reuse is validated')
        with reuse_prefix(self.blocks, cached, layer):
            return patched_score(self.model, inputs, self.ids, layer, positions, values)
