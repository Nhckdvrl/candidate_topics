from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class Family:
    key: str
    setting: str
    reading: str
    device: str


# Training and test families are deliberately disjoint.  Mechanics and action labels
# are identical; only the semantic wrapper changes.
TRAIN_FAMILIES: Sequence[Family] = (
    Family("greenhouse", "greenhouse", "growth-balance reading", "climate console"),
    Family("factory", "factory line", "stability reading", "machine console"),
    Family("bakery", "industrial bakery", "bake-balance reading", "oven console"),
    Family("traffic", "traffic-control room", "flow-balance reading", "signal console"),
    Family("clinic", "diagnostic clinic", "analyzer-balance reading", "lab console"),
    Family("warehouse", "warehouse", "throughput-balance reading", "routing console"),
    Family("theater", "theater", "stage-balance reading", "lighting console"),
    Family("network", "network operations room", "load-balance reading", "routing console"),
    Family("irrigation", "irrigation station", "water-balance reading", "pump console"),
    Family("ship", "cargo ship", "trim-balance reading", "stability console"),
)

TEST_FAMILIES: Sequence[Family] = (
    Family("observatory", "radio observatory", "alignment-balance reading", "dish console"),
    Family("museum", "museum preservation lab", "preservation-balance reading", "climate console"),
    Family("harbor", "harbor control center", "berth-balance reading", "harbor console"),
    Family("data_center", "data center", "thermal-balance reading", "cooling console"),
)

BY_KEY: Dict[str, Family] = {f.key: f for f in (*TRAIN_FAMILIES, *TEST_FAMILIES)}


def training_family_keys(base_seed: int, diversity: str, n_episodes: int) -> List[str]:
    if diversity not in {"concentrated", "distributed"}:
        raise ValueError("diversity must be concentrated or distributed")
    rng = random.Random(base_seed * 3571 + 29)
    keys = [f.key for f in TRAIN_FAMILIES]
    if diversity == "concentrated":
        key = keys[base_seed % len(keys)]
        return [key] * n_episodes

    out: List[str] = []
    while len(out) < n_episodes:
        block = list(keys)
        rng.shuffle(block)
        out.extend(block)
    return out[:n_episodes]


def heldout_family_key(base_seed: int) -> str:
    return TEST_FAMILIES[base_seed % len(TEST_FAMILIES)].key


def render_episode_intro(family_key: str, state: int, steps: int, budget: int) -> str:
    f = BY_KEY[family_key]
    return (
        f"A new task begins in a {f.setting}. You operate its {f.device}.\n"
        f"Goal: keep the {f.reading} as close to 0 as possible over {steps} steps.\n"
        f"Current reading: {state}.\n"
        f"At each step choose A, B, C, or WAIT. A/B/C each use one intervention token; "
        f"WAIT uses none. You have {budget} intervention tokens for this task.\n"
        "Infer what is useful from the outcomes. Reply with exactly one token: A, B, C, or WAIT."
    )


def render_feedback(family_key: str, before: int, after: int, interventions_left: int, step: int, total_steps: int) -> str:
    f = BY_KEY[family_key]
    return (
        f"Step {step}/{total_steps}. The {f.reading} changed from {before} to {after}. "
        f"Intervention tokens left: {interventions_left}. Choose A, B, C, or WAIT."
    )


def render_episode_end(family_key: str, state: int) -> str:
    f = BY_KEY[family_key]
    return f"This task is finished. Final {f.reading}: {state}."


SYSTEM_PROMPT = (
    "You are interacting with a sequence of control tasks. Your objective in each task is to keep the observed reading near 0. "
    "Use only the interaction history you receive. When asked for an action, output exactly one token from A, B, C, WAIT and nothing else."
)
