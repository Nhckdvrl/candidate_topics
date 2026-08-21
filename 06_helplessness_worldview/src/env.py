from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Literal, Sequence

from .config import ExperimentConfig
from .renderers import get_family, schedule_for

LatentAction = Literal["a", "b", "wait"]


def stable_seed(*parts: object) -> int:
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


@dataclass(frozen=True)
class TrialFeedback:
    success: bool
    reward: float
    cost: float
    net: float


@dataclass(frozen=True)
class EpisodeSpec:
    family: str
    effective_action: Literal["a", "b"]
    uniforms: tuple[float, ...]


@dataclass(frozen=True)
class SessionSpec:
    diversity: str
    pair_id: int
    episodes: tuple[EpisodeSpec, ...]
    test: EpisodeSpec


def build_session_spec(diversity: str, pair_id: int, cfg: ExperimentConfig, seed: int) -> SessionSpec:
    cfg.validate()
    schedule = schedule_for(diversity, cfg.episodes, pair_id)
    eps: list[EpisodeSpec] = []
    for ep_idx, family in enumerate(schedule):
        rng = random.Random(stable_seed(seed, pair_id, "train", ep_idx))
        effective = "a" if rng.random() < 0.5 else "b"
        uniforms = tuple(rng.random() for _ in range(cfg.trials_per_episode))
        eps.append(EpisodeSpec(family, effective, uniforms))
    test_rng = random.Random(stable_seed(seed, pair_id, "test", cfg.test_family))
    test_effective = "a" if test_rng.random() < 0.5 else "b"
    test_uniforms = tuple(test_rng.random() for _ in range(cfg.test_trials))
    return SessionSpec(
        diversity=diversity,
        pair_id=pair_id,
        episodes=tuple(eps),
        test=EpisodeSpec(cfg.test_family, test_effective, test_uniforms),
    )


def controlled_feedback(action: LatentAction, spec: EpisodeSpec, trial_index: int, cfg: ExperimentConfig, *, test: bool = False) -> TrialFeedback:
    if test:
        p_eff, p_other, p_wait = cfg.test_p_effective, cfg.test_p_other, cfg.test_p_wait
    else:
        p_eff, p_other, p_wait = cfg.p_effective, cfg.p_other, cfg.p_wait
    if action == "wait":
        p = p_wait
    elif action == spec.effective_action:
        p = p_eff
    else:
        p = p_other
    success = spec.uniforms[trial_index] < p
    reward = cfg.success_reward if success else 0.0
    cost = 0.0 if action == "wait" else cfg.intervention_cost
    return TrialFeedback(success, reward, cost, reward - cost)


def yoked_feedback(action: LatentAction, master_success: bool, cfg: ExperimentConfig) -> TrialFeedback:
    reward = cfg.success_reward if master_success else 0.0
    cost = 0.0 if action == "wait" else cfg.intervention_cost
    return TrialFeedback(master_success, reward, cost, reward - cost)


def surface_to_latent(family: str, surface_action: str) -> LatentAction:
    mapping = get_family(family).action_map()
    try:
        return mapping[surface_action]  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError(f"invalid action {surface_action!r} for family {family!r}") from exc


def validate_yoke(master: Sequence[bool], yoked: Sequence[bool]) -> None:
    if list(master) != list(yoked):
        raise AssertionError("yoked outcome sequence differs from master")
