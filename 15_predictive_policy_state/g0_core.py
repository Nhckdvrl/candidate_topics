from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class ProbeResult:
    r2: float
    mse: float
    baseline_mse: float
    per_sample_mse: torch.Tensor
    per_sample_baseline_mse: torch.Tensor


def future_change_target(video_latents: torch.Tensor) -> torch.Tensor:
    """Return the full future-latent change target for a clean clip.

    Args:
        video_latents: [B, C, T, H, W] cached Wan VAE latents. T must be > 1.

    Returns:
        [B, P] float32 tensor containing every future latent position after
        subtracting the first latent frame. No PCA or learned target compression
        is used; the probe must predict the actual cached future-latent change.
    """
    if video_latents.ndim != 5:
        raise ValueError(
            f"video_latents must be [B,C,T,H,W], got {tuple(video_latents.shape)}"
        )
    if video_latents.shape[2] <= 1:
        raise ValueError(
            f"video_latents must contain at least one future latent frame, got T={video_latents.shape[2]}"
        )
    x = video_latents.detach().to(dtype=torch.float32, device="cpu")
    delta = x[:, :, 1:, :, :] - x[:, :, :1, :, :]
    return delta.reshape(delta.shape[0], -1).contiguous()


def _standardize_x(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError("x_train and x_test must both be 2D")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("x_train/x_test feature dimensions differ")
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    return (x_train - mean) / std, (x_test - mean) / std


def linear_ridge_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    ridge: float = 1e-2,
    target_chunk_size: int = 4096,
) -> ProbeResult:
    """Evaluate one fixed linear ridge probe without hyperparameter search.

    The solve is performed in sample space, so it remains practical when the
    future-latent target has tens of thousands of dimensions. Target dimensions
    are processed in chunks and never compressed.
    """
    if ridge <= 0:
        raise ValueError(f"ridge must be > 0, got {ridge}")
    if target_chunk_size <= 0:
        raise ValueError("target_chunk_size must be > 0")

    x_train = x_train.detach().to(dtype=torch.float32, device="cpu")
    x_test = x_test.detach().to(dtype=torch.float32, device="cpu")
    y_train = y_train.detach().to(dtype=torch.float32, device="cpu")
    y_test = y_test.detach().to(dtype=torch.float32, device="cpu")

    if x_train.ndim != 2 or x_test.ndim != 2 or y_train.ndim != 2 or y_test.ndim != 2:
        raise ValueError("all probe tensors must be 2D")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train/y_train sample counts differ")
    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError("x_test/y_test sample counts differ")
    if y_train.shape[1] != y_test.shape[1]:
        raise ValueError("y_train/y_test target dimensions differ")
    if x_train.shape[0] < 2 or x_test.shape[0] < 1:
        raise ValueError("probe requires at least 2 train samples and 1 test sample")

    x_train, x_test = _standardize_x(x_train, x_test)
    feature_dim = float(x_train.shape[1])
    # Normalized linear kernel keeps the fixed ridge value comparable across
    # feature widths and avoids a layer-count-dependent regularization scale.
    k_train = (x_train @ x_train.T) / feature_dim
    k_test = (x_test @ x_train.T) / feature_dim
    eye = torch.eye(k_train.shape[0], dtype=k_train.dtype)

    system = k_train + float(ridge) * eye
    try:
        chol = torch.linalg.cholesky(system)
    except RuntimeError:
        # Numerical fallback only; this is not a tunable scientific parameter.
        jitter = 1e-5 * max(1.0, float(k_train.diag().mean().item()))
        chol = torch.linalg.cholesky(system + jitter * eye)

    test_sse = torch.zeros(x_test.shape[0], dtype=torch.float64)
    baseline_sse = torch.zeros(x_test.shape[0], dtype=torch.float64)
    target_dim = int(y_train.shape[1])

    for start in range(0, target_dim, target_chunk_size):
        end = min(start + target_chunk_size, target_dim)
        ytr = y_train[:, start:end]
        yte = y_test[:, start:end]
        y_mean = ytr.mean(dim=0, keepdim=True)
        rhs = ytr - y_mean
        alpha = torch.cholesky_solve(rhs, chol)
        pred = k_test @ alpha + y_mean

        test_sse += ((pred - yte) ** 2).sum(dim=1, dtype=torch.float64)
        baseline_sse += ((yte - y_mean) ** 2).sum(dim=1, dtype=torch.float64)

    per_sample_mse = (test_sse / float(target_dim)).to(torch.float32)
    per_sample_baseline_mse = (baseline_sse / float(target_dim)).to(torch.float32)
    mse = float(test_sse.sum().item() / (x_test.shape[0] * target_dim))
    baseline_mse = float(baseline_sse.sum().item() / (x_test.shape[0] * target_dim))
    r2 = 1.0 - (mse / baseline_mse) if baseline_mse > 0 else float("nan")

    return ProbeResult(
        r2=float(r2),
        mse=float(mse),
        baseline_mse=float(baseline_mse),
        per_sample_mse=per_sample_mse,
        per_sample_baseline_mse=per_sample_baseline_mse,
    )


def paired_bootstrap_mean_ci(
    values: torch.Tensor | Iterable[float],
    *,
    seed: int = 0,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for the mean of one per-sample effect vector."""
    x = torch.as_tensor(list(values) if not isinstance(values, torch.Tensor) else values)
    x = x.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    if x.numel() == 0:
        raise ValueError("values must be non-empty")
    if n_boot <= 0:
        raise ValueError("n_boot must be > 0")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")

    mean = float(x.mean().item())
    if x.numel() == 1:
        return mean, mean, mean

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    n = int(x.numel())
    # Chunk bootstrap indices so large n_boot does not allocate excessively.
    boot_means = []
    chunk = 256
    for start in range(0, n_boot, chunk):
        current = min(chunk, n_boot - start)
        idx = torch.randint(0, n, (current, n), generator=generator)
        boot_means.append(x[idx].mean(dim=1))
    boot = torch.cat(boot_means, dim=0)
    lo = float(torch.quantile(boot, alpha / 2.0).item())
    hi = float(torch.quantile(boot, 1.0 - alpha / 2.0).item())
    return mean, lo, hi


def relative_change(new: float, reference: float, eps: float = 1e-12) -> float:
    denom = max(abs(float(reference)), eps)
    return (float(new) - float(reference)) / denom
