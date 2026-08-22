"""Pure-PyTorch padding helpers for the no-sequence-parallel path.

The G1 trainer audit fixes Ulysses/sequence parallelism to 1, so these helpers
must not be reached during the actual forward/backward smoke. They are kept
semantically useful to fail less mysteriously if an accidental call occurs.
"""
from __future__ import annotations

import torch


def rearrange(x, pattern, **sizes):
    if pattern == "b s ... -> (b s) ...":
        return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
    raise NotImplementedError(f"compat shim does not implement rearrange pattern: {pattern}")


def index_first_axis(x, indices):
    return x.index_select(0, indices)


def unpad_input(hidden_states, attention_mask):
    mask = attention_mask.bool()
    indices = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
    flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
    values = flat.index_select(0, indices)
    lengths = mask.sum(dim=1, dtype=torch.int32)
    cu = torch.zeros(mask.shape[0] + 1, dtype=torch.int32, device=mask.device)
    cu[1:] = torch.cumsum(lengths, dim=0)
    return values, indices, cu, int(lengths.max().item()) if len(lengths) else 0


def pad_input(hidden_states, indices, batch, seqlen):
    out = hidden_states.new_zeros((batch * seqlen,) + tuple(hidden_states.shape[1:]))
    out.index_copy_(0, indices, hidden_states)
    return out.reshape(batch, seqlen, *hidden_states.shape[1:])
