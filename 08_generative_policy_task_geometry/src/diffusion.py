from __future__ import annotations

import math
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        x = t.float()[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class Denoiser(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, horizon: int, hidden: int = 512, time_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.time = SinusoidalTimeEmbedding(time_dim)
        in_dim = obs_dim + horizon * action_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, horizon * action_dim),
        )

    def forward(self, noisy_action: torch.Tensor, obs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b = noisy_action.shape[0]
        x = torch.cat([noisy_action.reshape(b, -1), obs, self.time(t)], dim=-1)
        return self.net(x).reshape_as(noisy_action)


@dataclass
class DiffusionConfig:
    obs_dim: int
    action_dim: int
    horizon: int = 8
    diffusion_steps: int = 50
    hidden: int = 512
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class DiffusionPolicy(nn.Module):
    """Small low-dimensional conditional DDPM used only for the existence pilot."""

    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.model = Denoiser(cfg.obs_dim, cfg.action_dim, cfg.horizon, hidden=cfg.hidden)
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.diffusion_steps)
        alphas = 1.0 - betas
        ab = torch.cumprod(alphas, dim=0)
        ab_prev = torch.cat([torch.ones(1), ab[:-1]], dim=0)
        posterior_var = betas * (1.0 - ab_prev) / (1.0 - ab)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", ab)
        self.register_buffer("posterior_var", posterior_var.clamp_min(1e-12))
        self.register_buffer("obs_mean", torch.zeros(cfg.obs_dim))
        self.register_buffer("obs_std", torch.ones(cfg.obs_dim))
        self.register_buffer("act_mean", torch.zeros(cfg.action_dim))
        self.register_buffer("act_std", torch.ones(cfg.action_dim))

    def set_normalizer(self, obs: torch.Tensor, actions: torch.Tensor) -> None:
        self.obs_mean.copy_(obs.mean(dim=0))
        self.obs_std.copy_(obs.std(dim=0).clamp_min(1e-4))
        flat = actions.reshape(-1, actions.shape[-1])
        self.act_mean.copy_(flat.mean(dim=0))
        self.act_std.copy_(flat.std(dim=0).clamp_min(1e-4))

    def norm_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.obs_mean) / self.obs_std

    def norm_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.act_mean) / self.act_std

    def denorm_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.act_std + self.act_mean

    def training_loss(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs = self.norm_obs(obs)
        x0 = self.norm_action(action)
        b = x0.shape[0]
        t = torch.randint(0, self.cfg.diffusion_steps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        ab = self.alpha_bar[t].view(b, 1, 1)
        xt = ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise
        pred = self.model(xt, obs, t)
        return torch.mean((pred - noise) ** 2)

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Return [B,n_samples,H,D] action chunks."""
        if obs.ndim == 1:
            obs = obs[None, :]
        b = obs.shape[0]
        obs_n = self.norm_obs(obs)
        obs_rep = obs_n[:, None, :].expand(b, n_samples, -1).reshape(b * n_samples, -1)
        x = torch.randn(
            b * n_samples, self.cfg.horizon, self.cfg.action_dim,
            device=obs.device, dtype=obs.dtype,
        )
        for ti in reversed(range(self.cfg.diffusion_steps)):
            t = torch.full((b * n_samples,), ti, device=obs.device, dtype=torch.long)
            eps = self.model(x, obs_rep, t)
            beta = self.betas[ti]
            alpha = self.alphas[ti]
            ab = self.alpha_bar[ti]
            mean = (x - beta / torch.sqrt(1.0 - ab) * eps) / torch.sqrt(alpha)
            if ti > 0:
                x = mean + torch.sqrt(self.posterior_var[ti]) * torch.randn_like(x)
            else:
                x = mean
        x = self.denorm_action(x)
        return x.reshape(b, n_samples, self.cfg.horizon, self.cfg.action_dim)

    def checkpoint_payload(self) -> dict:
        return {"config": asdict(self.cfg), "state_dict": self.state_dict()}

    @classmethod
    def from_payload(cls, payload: dict, map_location: str | torch.device = "cpu") -> "DiffusionPolicy":
        cfg = DiffusionConfig(**payload["config"])
        model = cls(cfg)
        model.load_state_dict(payload["state_dict"])
        return model.to(map_location)
