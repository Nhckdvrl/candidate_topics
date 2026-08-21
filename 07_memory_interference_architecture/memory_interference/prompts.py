from __future__ import annotations

from typing import Literal

from .data import Episode

Condition = Literal["RI", "PI"]


def render_prompt(episode: Episode, query_key: str, condition: Condition) -> str:
    """Render the shared stream; RI/PI differ only in the final query wording."""
    if query_key not in episode.histories:
        raise KeyError(query_key)
    if condition not in ("RI", "PI"):
        raise ValueError(condition)

    initial = [a for a in episode.assignments if a.is_initial]
    updates = [a for a in episode.assignments if not a.is_initial]

    lines = [
        "You will learn initial category-value facts, then see later updates.",
        "Keep the full order of values for each category.",
        "",
        "INITIAL FACTS:",
    ]
    lines.extend(f"{a.category}: {a.value}" for a in initial)
    lines.extend(["", "UPDATES:"])
    lines.extend(f"{a.category}: {a.value}" for a in updates)
    lines.extend(["", "QUERY:"])
    if condition == "RI":
        lines.append(f'What was the INITIAL value of "{query_key}"?')
    else:
        lines.append(f'What was the LAST (most recent) value of "{query_key}"?')
    lines.append("ANSWER:")
    return "\n".join(lines) + "\n"


def target_value(episode: Episode, query_key: str, condition: Condition) -> str:
    return episode.first(query_key) if condition == "RI" else episode.latest(query_key)


def candidate_values(episode: Episode, query_key: str) -> tuple[str, ...]:
    """Primary candidate set: every value ever assigned to the queried key."""
    return tuple(episode.histories[query_key])
