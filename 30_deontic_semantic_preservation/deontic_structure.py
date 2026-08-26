from __future__ import annotations

import re
from dataclasses import asdict, dataclass


MODAL_PATTERNS = {
    "PROHIBITION": [
        r"\b(?:must not|shall not|may not|will not|is prohibited from|are prohibited from|cannot|can['’]t)\b",
    ],
    "OBLIGATION": [
        r"\b(?:(?:must|shall)(?!\s+not\b)|required to|is required to|are required to|has to|have to|agrees to|undertakes to)\b",
        r"\b(?:is|are|will be) responsible for\b",
    ],
    "PERMISSION": [
        r"\b(?:may(?!\s+not\b)|is permitted to|are permitted to|is allowed to|are allowed to|can(?!not\b|['’]t\b))\b",
    ],
    "ENTITLEMENT": [
        r"\b(?:is entitled to|are entitled to|will be entitled to|eligible to|has the right to|have the right to)\b",
    ],
}
COND = r"\b(?:if|unless|provided that|provided however|subject to|on condition that|in the event that|only if)\b"
EXC = r"\b(?:except(?: that)?|other than|save for|notwithstanding|provided however)\b"


@dataclass(frozen=True)
class Deontic:
    modality: str
    modalities: tuple[str, ...]
    actor: str | None
    conditional: bool
    exception: bool
    negation: bool


def extract(text: str) -> Deontic:
    t = " ".join(str(text).lower().split())
    actor_match = re.match(r"^\[([^\]]+)\]", t)
    modalities = tuple(
        modality
        for modality, patterns in MODAL_PATTERNS.items()
        if any(re.search(pattern, t) for pattern in patterns)
    )
    if not modalities:
        modality = "NONE"
    elif len(modalities) == 1:
        modality = modalities[0]
    else:
        modality = "MULTIPLE"
    return Deontic(
        modality=modality,
        modalities=modalities,
        actor=actor_match.group(1) if actor_match else None,
        conditional=bool(re.search(COND, t)),
        exception=bool(re.search(EXC, t)),
        negation=bool(re.search(r"\b(?:not|never|no)\b", t)),
    )


def compare(original: str, simplified: str) -> dict:
    a, b = extract(original), extract(simplified)
    return {
        "original": asdict(a),
        "simplified": asdict(b),
        "modality_changed": set(a.modalities) != set(b.modalities),
        "modalities_lost": sorted(set(a.modalities) - set(b.modalities)),
        "modalities_gained": sorted(set(b.modalities) - set(a.modalities)),
        "actor_changed": a.actor is not None and b.actor is not None and a.actor != b.actor,
        "condition_lost": a.conditional and not b.conditional,
        "exception_lost": a.exception and not b.exception,
        "negation_changed": a.negation != b.negation,
    }
