from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable

import torch


def get_decoder_layers(model: torch.nn.Module):
    """Return decoder layers for common HF causal-LM layouts."""
    candidate_paths = (
        ("model", "layers"),
        ("model", "model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    )
    for path in candidate_paths:
        obj: Any = model
        ok = True
        for name in path:
            if not hasattr(obj, name):
                ok = False
                break
            obj = getattr(obj, name)
        if ok and isinstance(obj, (torch.nn.ModuleList, list, tuple)):
            return obj
    raise RuntimeError("Could not locate decoder layers")


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        if not output or not torch.is_tensor(output[0]):
            raise TypeError("Unsupported tuple output from decoder layer")
        return (hidden, *output[1:])
    if isinstance(output, list):
        if not output or not torch.is_tensor(output[0]):
            raise TypeError("Unsupported list output from decoder layer")
        out = list(output)
        out[0] = hidden
        return out
    raise TypeError(f"Unsupported decoder-layer output type: {type(output)!r}")


def _scaled_output(hidden_in: torch.Tensor, output: Any, scale: float) -> Any:
    """Return h_in + scale * (h_out-h_in), preserving auxiliary outputs.

    scale=0 and scale=1 are special-cased. This matters in bf16/fp16: computing
    h + 1*(out-h) is not guaranteed bit-identical to `out`, and h+0*(out-h)
    can propagate NaNs from a discarded update. The endpoint interventions are
    therefore exact rather than algebraically equivalent only in real arithmetic.
    """
    if scale == 1.0:
        return output
    if scale == 0.0:
        return _replace_hidden(output, hidden_in)

    if torch.is_tensor(output):
        if output.shape != hidden_in.shape:
            raise RuntimeError(
                f"Layer output shape {tuple(output.shape)} != input shape {tuple(hidden_in.shape)}"
            )
        return hidden_in + scale * (output - hidden_in)

    if isinstance(output, tuple):
        if not output or not torch.is_tensor(output[0]):
            raise TypeError("Unsupported tuple output from decoder layer")
        first = hidden_in + scale * (output[0] - hidden_in)
        return (first, *output[1:])

    if isinstance(output, list):
        if not output or not torch.is_tensor(output[0]):
            raise TypeError("Unsupported list output from decoder layer")
        out = list(output)
        out[0] = hidden_in + scale * (out[0] - hidden_in)
        return out

    raise TypeError(f"Unsupported decoder-layer output type: {type(output)!r}")


@contextmanager
def residual_scale_layer(model: torch.nn.Module, layer_index: int, scale: float = 0.0):
    """Temporarily scale one decoder block's visible residual-stream update."""
    layers = get_decoder_layers(model)
    if not 0 <= layer_index < len(layers):
        raise IndexError(f"layer_index={layer_index} outside [0, {len(layers)-1}]")
    if not 0.0 <= scale <= 1.0:
        raise ValueError("scale must be in [0, 1]")
    layer = layers[layer_index]

    def hook_with_kwargs(module, args, kwargs, output):
        hidden = args[0] if args else kwargs.get("hidden_states")
        if hidden is None or not torch.is_tensor(hidden):
            raise RuntimeError("Could not recover hidden_states from decoder-layer call")
        return _scaled_output(hidden, output, scale)

    def hook_legacy(module, args, output):
        if not args or not torch.is_tensor(args[0]):
            raise RuntimeError("Could not recover positional hidden_states")
        return _scaled_output(args[0], output, scale)

    try:
        handle = layer.register_forward_hook(hook_with_kwargs, with_kwargs=True)
    except TypeError:
        handle = layer.register_forward_hook(hook_legacy)
    try:
        yield
    finally:
        handle.remove()


def parse_layer_spec(spec: str, num_layers: int) -> list[int]:
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(num_layers))
    if spec in {"none", ""}:
        return []
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a_s, b_s = part.split("-", 1)
            a, b = int(a_s), int(b_s)
            if b < a:
                raise ValueError(f"Invalid descending range: {part}")
            result.update(range(a, b + 1))
        else:
            result.add(int(part))
    ordered = sorted(result)
    bad = [i for i in ordered if i < 0 or i >= num_layers]
    if bad:
        raise ValueError(f"Layer indices out of range for {num_layers} layers: {bad}")
    return ordered


def shard_layers(layers: Iterable[int], shard_index: int, shard_count: int) -> list[int]:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= index < count")
    return [layer for layer in layers if layer % shard_count == shard_index]
