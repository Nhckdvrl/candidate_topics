from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable

import torch


def get_decoder_layers(model: torch.nn.Module):
    """Return the decoder-layer ModuleList for common HF causal-LM layouts.

    Topic 12 is locked to Qwen3 for G-0, but keeping this resolver generic makes
    the confirmation run reusable without hiding architecture-specific choices.
    """
    candidate_paths = (
        ("model", "layers"),          # Qwen2/3, Llama, Mistral HF CausalLM
        ("model", "model", "layers"), # wrapped models
        ("transformer", "h"),         # GPT-style
        ("gpt_neox", "layers"),       # GPT-NeoX/Pythia
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
    raise RuntimeError(
        "Could not locate decoder layers. Expected one of: "
        "model.layers, model.model.layers, transformer.h, gpt_neox.layers."
    )


def _scaled_output(hidden_in: torch.Tensor, output: Any, scale: float) -> Any:
    """Replace a block's residual update with `scale * update`.

    For a residual decoder block with input h and output h + delta, this returns
    h + scale * delta. scale=0 is exact block bypass; scale=1 is unchanged.

    If a layer returns auxiliary values (e.g. a cache tuple), only the hidden
    state is replaced. This is deliberate: the original layer still executes so
    generation can keep an internally valid KV cache. Because the hidden output
    is replaced before downstream layers see it, the layer's contribution to the
    residual stream is removed while cache bookkeeping remains intact.
    """
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
def residual_scale_layer(
    model: torch.nn.Module,
    layer_index: int,
    scale: float = 0.0,
):
    """Temporarily scale one decoder block's residual update.

    Primary G-0 uses scale=0.0 (full layer bypass).
    Confirmation uses scale=0.5 to test whether the rank ordering survives a
    milder intervention instead of relying on catastrophic deletion.
    """
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
            raise RuntimeError("Could not recover positional hidden_states from decoder-layer call")
        return _scaled_output(args[0], output, scale)

    try:
        handle = layer.register_forward_hook(hook_with_kwargs, with_kwargs=True)
    except TypeError:  # old PyTorch fallback
        handle = layer.register_forward_hook(hook_legacy)

    try:
        yield
    finally:
        handle.remove()


def parse_layer_spec(spec: str, num_layers: int) -> list[int]:
    """Parse `all`, `none`, comma lists, and inclusive ranges such as 8-12."""
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
