from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    device: torch.device
    max_context: Optional[int]


def _torch_dtype(name: str):
    table = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "auto": "auto",
    }
    if name not in table:
        raise ValueError(f"unknown dtype: {name}")
    return table[name]


def load_model(spec: dict) -> LoadedModel:
    backend = spec.get("backend", "hf")
    if backend != "hf":
        raise ValueError(f"unsupported backend: {backend}")
    model_id = spec["model_id"]
    device = torch.device(spec.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype = _torch_dtype(spec.get("dtype", "bf16" if device.type == "cuda" else "fp32"))

    try:
        import fla  # noqa: F401
    except ImportError:
        if spec.get("requires_fla", False):
            raise RuntimeError(
                "This checkpoint requires the full flash-linear-attention package "
                "(for CUDA: `pip install 'flash-linear-attention[cuda]'`)."
            )

    tokenizer_id = spec.get("tokenizer_id", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id, trust_remote_code=spec.get("trust_remote_code", False)
    )
    kwargs = {
        "trust_remote_code": spec.get("trust_remote_code", False),
        "torch_dtype": dtype,
    }
    if device.type == "cuda":
        kwargs["device_map"] = {"": str(device)}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device.type != "cuda":
        model = model.to(device)
    model.eval()

    max_context = spec.get("max_context")
    if max_context is None:
        cfg = getattr(model, "config", None)
        candidate = getattr(cfg, "max_position_embeddings", None) if cfg is not None else None
        if isinstance(candidate, int) and candidate < 10_000_000:
            max_context = candidate
    return LoadedModel(model=model, tokenizer=tokenizer, device=device, max_context=max_context)
