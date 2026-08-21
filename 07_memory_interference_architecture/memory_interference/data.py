from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class Assignment:
    category: str
    value: str
    presentation_index: int
    is_initial: bool


@dataclass(frozen=True)
class Episode:
    episode_id: int
    num_updates: int
    categories: Tuple[str, ...]
    assignments: Tuple[Assignment, ...]
    histories: Mapping[str, Tuple[str, ...]]

    def first(self, category: str) -> str:
        return self.histories[category][0]

    def latest(self, category: str) -> str:
        return self.histories[category][-1]


def load_pool(path: str | Path) -> Dict[str, List[str]]:
    """Load the public Unable-to-Forget category->values JSON pool."""
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("dataset must be a non-empty JSON object")

    out: Dict[str, List[str]] = {}
    for key, values in raw.items():
        if not isinstance(key, str) or not isinstance(values, list):
            raise ValueError("expected mapping[str, list[str]]")
        cleaned = [str(v).strip() for v in values if str(v).strip()]
        seen = set()
        deduped = []
        for value in cleaned:
            marker = value.casefold()
            if marker not in seen:
                seen.add(marker)
                deduped.append(value)
        out[key.strip()] = deduped
    return out


def validate_pool(pool: Mapping[str, Sequence[str]], num_keys: int, max_updates: int) -> None:
    if num_keys < 2:
        raise ValueError("num_keys must be >= 2")
    needed = max_updates + 1
    eligible = [k for k, v in pool.items() if len(set(v)) >= needed]
    if len(eligible) < num_keys:
        raise ValueError(
            f"need {num_keys} categories with >= {needed} distinct values; found {len(eligible)}"
        )


def _interleave_updates(
    per_key_updates: Mapping[str, Sequence[str]], rng: random.Random
) -> List[Tuple[str, str]]:
    """Interleave updates round-by-round while preserving each key's update order.

    Every update round contains exactly one update for every key, in a newly
    shuffled key order. This mirrors the balanced interference stream more
    closely than globally shuffling all update events: each key receives the
    same number of intervening update rounds, while within-round position is
    randomized. Consecutive update events are also constrained to target
    different keys, matching the source paradigm.
    """
    lengths = {len(v) for v in per_key_updates.values()}
    if len(lengths) != 1:
        raise ValueError("all keys must have the same number of updates")
    n_rounds = next(iter(lengths), 0)
    events: List[Tuple[str, str]] = []
    keys = list(per_key_updates)
    previous_key = None
    for update_idx in range(n_rounds):
        round_keys = list(keys)
        rng.shuffle(round_keys)
        if previous_key is not None and round_keys[0] == previous_key and len(round_keys) > 1:
            swap_idx = next(
                i for i, key in enumerate(round_keys[1:], start=1) if key != previous_key
            )
            round_keys[0], round_keys[swap_idx] = round_keys[swap_idx], round_keys[0]
        events.extend((category, per_key_updates[category][update_idx]) for category in round_keys)
        previous_key = round_keys[-1]
    return events


def build_episode(
    pool: Mapping[str, Sequence[str]],
    *,
    episode_id: int,
    num_keys: int,
    num_updates: int,
    seed: int,
) -> Episode:
    """Create one shared stimulus stream used unchanged for RI and PI queries."""
    if num_updates < 1:
        raise ValueError("num_updates must be >= 1 for an interference episode")

    rng = random.Random((seed + 1) * 1_000_003 + episode_id * 1009 + num_updates * 9176)
    eligible = [k for k, values in pool.items() if len(set(values)) >= num_updates + 1]
    if len(eligible) < num_keys:
        raise ValueError("not enough eligible categories")
    categories = tuple(rng.sample(sorted(eligible), num_keys))

    sampled: Dict[str, List[str]] = {}
    for category in categories:
        sampled[category] = rng.sample(list(pool[category]), num_updates + 1)

    assignments: List[Assignment] = []
    histories: Dict[str, List[str]] = {k: [] for k in categories}
    presentation_index = 0

    initial_order = list(categories)
    rng.shuffle(initial_order)
    for category in initial_order:
        value = sampled[category][0]
        histories[category].append(value)
        assignments.append(Assignment(category, value, presentation_index, True))
        presentation_index += 1

    events = _interleave_updates({k: sampled[k][1:] for k in categories}, rng)
    for category, value in events:
        histories[category].append(value)
        assignments.append(Assignment(category, value, presentation_index, False))
        presentation_index += 1

    frozen_histories = {k: tuple(v) for k, v in histories.items()}
    assert all(len(v) == num_updates + 1 for v in frozen_histories.values())
    return Episode(
        episode_id=episode_id,
        num_updates=num_updates,
        categories=categories,
        assignments=tuple(assignments),
        histories=frozen_histories,
    )


def select_query_keys(episode: Episode, count: int, seed: int) -> Tuple[str, ...]:
    if count <= 0 or count > len(episode.categories):
        raise ValueError("query key count must be between 1 and num_keys")
    rng = random.Random(seed * 65537 + episode.episode_id * 257 + episode.num_updates)
    return tuple(rng.sample(list(episode.categories), count))
