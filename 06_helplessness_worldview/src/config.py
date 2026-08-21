from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Diversity = Literal["concentrated", "distributed"]
Control = Literal["controllable", "uncontrollable"]


@dataclass(frozen=True)
class ExperimentConfig:
    episodes: int = 10
    trials_per_episode: int = 10
    test_trials: int = 8
    success_reward: float = 10.0
    intervention_cost: float = 1.0
    p_effective: float = 0.85
    p_other: float = 0.15
    p_wait: float = 0.15
    test_p_effective: float = 0.90
    test_p_other: float = 0.10
    test_p_wait: float = 0.10
    test_family: str = "orbital_station"

    @property
    def training_trials(self) -> int:
        return self.episodes * self.trials_per_episode

    def validate(self) -> None:
        if self.episodes < 2:
            raise ValueError("episodes must be >= 2")
        if self.trials_per_episode < 1 or self.test_trials < 1:
            raise ValueError("trial counts must be positive")
        for name in (
            "p_effective", "p_other", "p_wait",
            "test_p_effective", "test_p_other", "test_p_wait",
        ):
            p = getattr(self, name)
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if not self.p_effective > max(self.p_other, self.p_wait):
            raise ValueError("training effective action must be genuinely better")
        if not self.test_p_effective > max(self.test_p_other, self.test_p_wait):
            raise ValueError("test effective action must be genuinely better")
