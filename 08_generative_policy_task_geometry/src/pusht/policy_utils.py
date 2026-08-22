"""Load the pretrained Diffusion Policy for PushT and sample action chunks from a state.

Policy: `lerobot/diffusion_pusht` -- the LeRobot port of Diffusion Policy (Chi et al.,
RSS 2023) trained on the original PushT demonstrations. Released config: horizon 16,
`n_action_steps` 8, `n_obs_steps` 2, ResNet18 + spatial softmax, DDPM. Nothing is trained
here; the point is to measure a mature published policy.

Two things this module exists to get right.

**Normalisation.** The released checkpoint stores its normalisation statistics as module
buffers (`normalize_inputs.*`, `unnormalize_outputs.*`). LeRobot 0.4.x moved normalisation
out of the policy into processor pipelines, so `DiffusionPolicy.from_pretrained` loads
this checkpoint while *silently discarding* those buffers -- it only logs
"Unexpected key(s)". A policy run that way sees unnormalised pixels and coordinates and
emits garbage, which would look like "the policy is uncertain everywhere": exactly the
artefact that could fake a positive result for this topic. We therefore read the buffers
straight out of `model.safetensors` and apply the transforms ourselves, and
`assert_normalization_loaded` fails loudly if they are ever absent.

**Independent samples at one state.** `select_action` is stateful: it caches an action
queue and returns one action at a time. We need B *independent* draws from p(A | s) at one
fixed observation, so we drive the observation queue ourselves and call the diffusion
model directly on a batch of B replicated observations. Conditioning is then bit-identical
across the B draws and only the diffusion noise differs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

PRETRAINED_ID = "lerobot/diffusion_pusht"


@dataclass
class Normalizer:
    """The released checkpoint's own statistics, applied explicitly.

    Conventions copied from LeRobot's `normalize_processor.py`:
      MIN_MAX  : 2 * (x - min) / (max - min) - 1
      MEAN_STD : (x - mean) / std
    """

    state_min: torch.Tensor
    state_max: torch.Tensor
    image_mean: torch.Tensor
    image_std: torch.Tensor
    action_min: torch.Tensor
    action_max: torch.Tensor

    def to(self, device) -> "Normalizer":
        return Normalizer(*[t.to(device) for t in (
            self.state_min, self.state_max, self.image_mean,
            self.image_std, self.action_min, self.action_max)])

    def norm_state(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * (x - self.state_min) / (self.state_max - self.state_min) - 1.0

    def norm_image(self, x: torch.Tensor) -> torch.Tensor:
        """`x` is float in [0, 1], channel-first."""
        shape = (1,) * (x.ndim - 3) + (3, 1, 1)
        return (x - self.image_mean.view(shape)) / self.image_std.view(shape)

    def denorm_action(self, a: torch.Tensor) -> torch.Tensor:
        return (a + 1.0) / 2.0 * (self.action_max - self.action_min) + self.action_min


@dataclass
class PolicyBundle:
    policy: object
    norm: Normalizer
    device: torch.device
    obs_type: str = "pixels_agent_pos"
    n_obs_steps: int = 2
    n_action_steps: int = 8
    horizon: int = 16
    queue: deque = field(default_factory=lambda: deque(maxlen=2))


def _load_normalizer(pretrained: str) -> Normalizer:
    from pathlib import Path

    from safetensors.torch import load_file

    p = Path(pretrained) / "model.safetensors"
    if not p.exists():
        from huggingface_hub import hf_hub_download

        p = Path(hf_hub_download(pretrained, "model.safetensors"))
    sd = load_file(str(p))
    need = [
        "normalize_inputs.buffer_observation_state.min",
        "normalize_inputs.buffer_observation_state.max",
        "normalize_inputs.buffer_observation_image.mean",
        "normalize_inputs.buffer_observation_image.std",
        "unnormalize_outputs.buffer_action.min",
        "unnormalize_outputs.buffer_action.max",
    ]
    missing = [k for k in need if k not in sd]
    if missing:
        raise RuntimeError(
            "checkpoint is missing normalisation buffers; refusing to run an "
            f"unnormalised policy. missing={missing}"
        )
    return Normalizer(
        state_min=sd[need[0]].float(),
        state_max=sd[need[1]].float(),
        image_mean=sd[need[2]].float(),
        image_std=sd[need[3]].float(),
        action_min=sd[need[4]].float(),
        action_max=sd[need[5]].float(),
    )


def load_pusht_policy(pretrained: str = PRETRAINED_ID, device: str = "cuda") -> PolicyBundle:
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    policy = DiffusionPolicy.from_pretrained(pretrained)
    policy.to(dev)
    policy.eval()

    cfg = policy.config
    return PolicyBundle(
        policy=policy,
        norm=_load_normalizer(pretrained).to(dev),
        device=dev,
        n_obs_steps=cfg.n_obs_steps,
        n_action_steps=cfg.n_action_steps,
        horizon=cfg.horizon,
        queue=deque(maxlen=cfg.n_obs_steps),
    )


def assert_normalization_loaded(bundle: PolicyBundle) -> None:
    n = bundle.norm
    assert torch.all(n.state_max > n.state_min), "degenerate state normalisation"
    assert torch.all(n.action_max > n.action_min), "degenerate action normalisation"
    assert torch.all(n.image_std > 0), "degenerate image normalisation"


def reset_queue(bundle: PolicyBundle) -> None:
    bundle.queue.clear()


def make_frame(bundle: PolicyBundle, obs: dict) -> dict:
    """Normalise one env observation into the model's input frame."""
    dev = bundle.device
    img = torch.from_numpy(np.asarray(obs["pixels"], dtype=np.float32) / 255.0)
    img = img.permute(2, 0, 1).to(dev)  # HWC -> CHW
    state = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32)).to(dev)
    return {
        "observation.images": bundle.norm.norm_image(img).unsqueeze(0),  # [n_cam=1, C,H,W]
        "observation.state": bundle.norm.norm_state(state),
    }


@torch.no_grad()
def sample_chunks_for_queues(bundle: PolicyBundle, queues, seed: int | None = None) -> np.ndarray:
    """One action chunk for each of K independent observation histories, in one batch.

    Used by the long-horizon branch experiment, where K counterfactual copies of the same
    state are advanced closed-loop in lockstep. Batching keeps the cost of K branches
    close to the cost of one.

    Returns [K, n_action_steps, 2] in raw env action units.
    """
    batch = {}
    for key in ("observation.images", "observation.state"):
        per_env = [torch.stack([f[key] for f in q], dim=0) for q in queues]  # each [T, ...]
        batch[key] = torch.stack(per_env, dim=0).contiguous()                # [K, T, ...]

    if seed is not None:
        torch.manual_seed(seed)
        if bundle.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    cfg = bundle.policy.config
    global_cond = bundle.policy.diffusion._prepare_global_conditioning(batch)
    full = bundle.policy.diffusion.conditional_sample(len(queues), global_cond=global_cond)
    start = cfg.n_obs_steps - 1
    executed = full[:, start : start + cfg.n_action_steps]
    return bundle.norm.denorm_action(executed).float().cpu().numpy()


def push_observation(bundle: PolicyBundle, obs: dict) -> None:
    """Normalise one env observation and append it to the observation history.

    LeRobot pads the first steps by repeating the first observation; we do the same by
    filling the queue on its first push.
    """
    frame = make_frame(bundle, obs)
    if not bundle.queue:
        for _ in range(bundle.queue.maxlen):
            bundle.queue.append(frame)
    else:
        bundle.queue.append(frame)


@torch.no_grad()
def sample_action_chunks(
    bundle: PolicyBundle, n_samples: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """B independent action chunks from the current observation history.

    Returns `(predicted, executed)` in raw env action units (pixels in [0, 512]):

      * `predicted` [B, horizon=16, 2] -- the whole predicted chunk. FIPER scores ACE over
        the full `action_prediction_horizon` (16 for push_t in `configs/task/push_t.yaml`),
        so this is what the entropy baseline must see.
      * `executed`  [B, n_action_steps=8, 2] -- the slice the policy would actually run,
        which is what the counterfactual rollouts execute.

    Keeping them separate matters: scoring the baseline on a shorter chunk than it was
    designed for would weaken it artificially.
    """
    if not bundle.queue:
        raise RuntimeError("observation queue is empty; call push_observation first")

    batch = {}
    for key in ("observation.images", "observation.state"):
        stacked = torch.stack([f[key] for f in bundle.queue], dim=0)  # [T, ...]
        batch[key] = stacked.unsqueeze(0).expand(n_samples, *stacked.shape).contiguous()

    if seed is not None:
        torch.manual_seed(seed)
        if bundle.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    cfg = bundle.policy.config
    global_cond = bundle.policy.diffusion._prepare_global_conditioning(batch)
    full = bundle.policy.diffusion.conditional_sample(n_samples, global_cond=global_cond)
    # generate_actions() slices exactly this window for execution
    start = cfg.n_obs_steps - 1
    executed = full[:, start : start + cfg.n_action_steps]

    denorm = bundle.norm.denorm_action
    return (
        denorm(full).float().cpu().numpy(),
        denorm(executed).float().cpu().numpy(),
    )
