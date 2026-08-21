from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ACTIONS: Tuple[str, ...] = ("A", "B", "C", "WAIT")
ACTIVE_ACTIONS: Tuple[str, ...] = ("A", "B", "C")
EFFECTS: Tuple[int, ...] = (-1, 0, 1)


@dataclass(frozen=True)
class EpisodePlan:
    """All exogenous randomness needed to replay one episode exactly."""

    start_state: int
    action_effects: Dict[str, int]
    random_effects: Tuple[int, ...]


@dataclass
class StepResult:
    step: int
    state_before: int
    action: str
    active: bool
    effect: int
    state_after: int
    interventions_left: int

    @property
    def improved(self) -> bool:
        return abs(self.state_after) < abs(self.state_before)


@dataclass
class ControlEnvironment:
    """A tiny causal-control task.

    In a controllable episode, A/B/C have a stable hidden mapping onto {-1,0,+1}.
    WAIT follows the exogenous random-effect schedule.

    In an uncontrollable episode, *all* actions are ignored and the next effect is
    taken from an exogenous schedule.  When that schedule is copied from a paired
    controllable episode, both learners receive the same raw outcome trajectory,
    while only one learner's actions caused it (classic yoking logic).
    """

    controllable: bool
    plan: EpisodePlan
    n_steps: int = 10
    intervention_budget: int = 6
    yoked_effects: Optional[Sequence[int]] = None
    state: int = field(init=False)
    step_index: int = field(default=0, init=False)
    interventions_left: int = field(init=False)
    results: List[StepResult] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if len(self.plan.random_effects) < self.n_steps:
            raise ValueError("random_effects shorter than n_steps")
        if self.yoked_effects is not None and len(self.yoked_effects) < self.n_steps:
            raise ValueError("yoked_effects shorter than n_steps")
        if set(self.plan.action_effects) != set(ACTIVE_ACTIONS):
            raise ValueError("action_effects must map exactly A/B/C")
        if sorted(self.plan.action_effects.values()) != list(EFFECTS):
            raise ValueError("controllable mapping must be a permutation of -1/0/+1")
        self.state = int(self.plan.start_state)
        self.interventions_left = int(self.intervention_budget)

    @property
    def done(self) -> bool:
        return self.step_index >= self.n_steps

    def valid_actions(self) -> Tuple[str, ...]:
        if self.interventions_left <= 0:
            return ("WAIT",)
        return ACTIONS

    def step(self, action: str) -> StepResult:
        if self.done:
            raise RuntimeError("episode already finished")
        action = action.upper().strip()
        if action not in ACTIONS:
            raise ValueError(f"invalid action: {action}")
        if action in ACTIVE_ACTIONS and self.interventions_left <= 0:
            action = "WAIT"

        before = self.state
        active = action in ACTIVE_ACTIONS
        if active:
            self.interventions_left -= 1

        if self.controllable and active:
            effect = int(self.plan.action_effects[action])
        else:
            schedule = self.yoked_effects if self.yoked_effects is not None else self.plan.random_effects
            effect = int(schedule[self.step_index])

        self.state = before + effect
        result = StepResult(
            step=self.step_index,
            state_before=before,
            action=action,
            active=active,
            effect=effect,
            state_after=self.state,
            interventions_left=self.interventions_left,
        )
        self.results.append(result)
        self.step_index += 1
        return result


def make_episode_plan(seed: int, n_steps: int = 10) -> EpisodePlan:
    rng = random.Random(seed)
    start_state = rng.choice((-4, -3, 3, 4))
    effects = list(EFFECTS)
    rng.shuffle(effects)
    action_effects = dict(zip(ACTIVE_ACTIONS, effects))
    random_effects = tuple(rng.choice(EFFECTS) for _ in range(n_steps))
    return EpisodePlan(start_state, action_effects, random_effects)


def make_plans(base_seed: int, n_episodes: int, n_steps: int) -> List[EpisodePlan]:
    # Large odd stride prevents accidental overlap with model/API seeds.
    return [make_episode_plan(base_seed * 1009 + i * 7919 + 17, n_steps) for i in range(n_episodes)]


def replay_effects(results: Iterable[StepResult]) -> Tuple[int, ...]:
    return tuple(int(r.effect) for r in results)
