"""Controlled-noise OpenPI server with zero-intervention pi0.5 feature capture.

Run this inside an official openpi checkout. It wraps OpenPI's own Policy object and
WebsocketPolicyServer; the Topic-09 additions are only explicit inference noise and an
optional observational forward hook.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np

TOPIC09_NOISE_SEED = "__topic09_noise_seed"
TOPIC09_CAPTURE = "__topic09_capture_feature"


class ActionExpertLayerCapture:
    """Capture the complete decoder-layer residual output at every denoising step."""

    def __init__(self, model, layer_index: int, expected_denoise_steps: int = 10):
        self.model = model
        self.layer_index = int(layer_index)
        self.expected_denoise_steps = int(expected_denoise_steps)
        self._handle = None
        self._steps: list[np.ndarray] = []

    def __enter__(self):
        expert = self.model.paligemma_with_expert.gemma_expert.model
        if not getattr(self.model, "pi05", False):
            raise RuntimeError("Topic 09 primary feature contract is defined for pi0.5")
        if self.layer_index < 0 or self.layer_index >= len(expert.layers):
            raise IndexError(f"action-expert layer {self.layer_index} out of range")

        def hook(_module, _inputs, output):
            # During pi0.5 denoising these are exactly the action-token residual states.
            # We observe only; the hook returns nothing and therefore cannot alter output.
            h = output[0] if isinstance(output, (tuple, list)) else output
            h = h.detach().float().cpu().numpy()
            if h.ndim != 3 or h.shape[0] != 1:
                raise RuntimeError(f"unexpected hooked activation shape {h.shape}")
            self._steps.append(h[0].mean(axis=0))

        self._handle = expert.layers[self.layer_index].register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def pooled(self) -> np.ndarray:
        if len(self._steps) != self.expected_denoise_steps:
            raise RuntimeError(
                f"captured {len(self._steps)} denoise activations, expected "
                f"{self.expected_denoise_steps}; inference path changed"
            )
        return np.stack(self._steps, axis=0).mean(axis=0).astype(np.float32)


class ControlledInstrumentedPolicy:
    def __init__(
        self,
        base_policy,
        *,
        action_horizon: int,
        action_dim: int,
        layer_index: int = 11,
        denoise_steps: int = 10,
    ):
        self.base = base_policy
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.layer_index = int(layer_index)
        self.denoise_steps = int(denoise_steps)

    @property
    def metadata(self):
        return self.base.metadata

    def _noise(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(int(seed))
        return rng.standard_normal((self.action_horizon, self.action_dim), dtype=np.float32)

    def infer(self, request: dict) -> dict:
        req = dict(request)
        if TOPIC09_NOISE_SEED not in req:
            raise ValueError(f"missing required {TOPIC09_NOISE_SEED}")
        seed = int(req.pop(TOPIC09_NOISE_SEED))
        capture = bool(req.pop(TOPIC09_CAPTURE, False))
        noise = self._noise(seed)

        if not capture:
            return self.base.infer(req, noise=noise)

        if not getattr(self.base, "_is_pytorch_model", False):
            raise RuntimeError("feature capture requires the official PyTorch-converted checkpoint")
        model = self.base._model
        with ActionExpertLayerCapture(model, self.layer_index, self.denoise_steps) as cap:
            out = self.base.infer(req, noise=noise)
        out["topic09_feature"] = cap.pooled()
        out["topic09_feature_meta"] = {
            "layer": self.layer_index,
            "pool": "mean_action_tokens_then_mean_denoise_steps",
            "n_denoise_steps": self.denoise_steps,
            "noise_seed": seed,
        }
        return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="pi05_libero")
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--layer", type=int, default=11)
    p.add_argument("--denoise-steps", type=int, default=10)
    args = p.parse_args()

    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as openpi_config

    cfg = openpi_config.get_config(args.config)
    base = policy_config.create_trained_policy(cfg, args.checkpoint_dir, pytorch_device=args.device)
    wrapped = ControlledInstrumentedPolicy(
        base,
        action_horizon=cfg.model.action_horizon,
        action_dim=cfg.model.action_dim,
        layer_index=args.layer,
        denoise_steps=args.denoise_steps,
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped,
        host="0.0.0.0",
        port=args.port,
        metadata=wrapped.metadata,
    )
    logging.info("Topic09 controlled server on port %d", args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
