from __future__ import annotations

import re
from dataclasses import asdict, dataclass


# REJECTED is not the bottom of a commitment scale: rejection is a polarity
# branch.  Keep only the states for which an ordinal comparison is meaningful.
COMMITMENT_ORDER = {
    "OPEN": 0,
    "PROPOSED": 1,
    "TENTATIVE": 2,
    "CONDITIONAL": 2,
    "DECIDED": 3,
}

_PATTERNS = {
    "REJECTED": [
        r"\b(?:reject(?:ed|s)?|declin(?:e|ed|es)|decided not to|will not|won['’]t)\b",
    ],
    "CONDITIONAL": [
        r"\b(?:if|unless|provided that|subject to|pending|once|contingent on)\b",
        r"\bafter\b.{0,50}\bapproval\b",
    ],
    "TENTATIVE": [
        r"\b(?:tentative(?:ly)?|probably|likely|for now|provisionally|lean(?:ing)? toward)\b",
    ],
    "DECIDED": [
        r"\b(?:we|they|(?:the\s+)?(?:group|team|meeting|participants?)|management|the company)(?:\s+have|\s+has)?\s+(?:decided|agreed|confirmed|resolved)\b",
        r"\b(?:final decision|decision is|decision was made|has been (?:decided|agreed|confirmed|resolved))\b",
        r"\b(?:let['’]s|we['’]ll) go (?:with|for)\b",
    ],
    "PROPOSED": [
        r"\b(?:propos(?:e|ed|al)|suggest(?:ed|ion)?|recommend(?:ed|ation)?|could|should we|how about|what if|might)\b",
    ],
    "OPEN": [
        r"\b(?:open question|undecided|not decided|to be decided|tbd|still discussing|discuss further|no final decision)\b",
    ],
}

_NEGATED_DECISION = re.compile(
    r"(?:\b(?:not|never)\s+(?:yet\s+)?(?:decided|agreed|confirmed|resolved)\b"
    r"|\b(?:have|has|had|do|does|did|is|are|was|were|can|could|will|would)n['’]t\s+"
    r"(?:yet\s+)?(?:decide|decided|agree|agreed|confirm|confirmed|resolve|resolved)\b"
    r"|\bno\s+(?:final\s+)?(?:decision|agreement)\b"
    r"|\b(?:did|does|do|was|were|is|are)\s+not\s+(?:make|reach)\s+a?\s*(?:final\s+)?(?:decision|agreement)\b)"
)

_DECISION_TO_DELIBERATE = re.compile(
    r"\b(?:decided|agreed|resolved)\s+to\s+(?:consider|discuss|explore|review|investigate)\b"
)


@dataclass(frozen=True)
class StateParse:
    state: str
    matched: tuple[str, ...]
    conditional: bool
    negated_decision: bool
    explicit: bool
    ambiguous: bool


def classify_state(text: str, *, genre: str = "source") -> StateParse:
    """Conservative lexical first pass over one source window.

    UNKNOWN is deliberately distinct from an explicitly open decision.  The
    scorer should not manufacture an OPEN label merely because no cue matched.
    ``ambiguous`` marks windows containing cues from competing branches; these
    should be excluded or adjudicated in a headline scientific estimate.
    """
    t = " ".join(str(text).lower().split())
    negated_decision = bool(_NEGATED_DECISION.search(t))
    if genre not in {"source", "summary"}:
        raise ValueError("genre must be 'source' or 'summary'")
    hits = {state: [] for state in _PATTERNS}
    for state, patterns in _PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, t):
                hits[state].append(pattern)
    # "Decided to consider X" commits only to further deliberation; it does
    # not license a claim that X itself was adopted.
    decision_to_deliberate = bool(_DECISION_TO_DELIBERATE.search(t))
    if decision_to_deliberate:
        hits["DECIDED"].clear()
        hits["PROPOSED"].append("decision-to-deliberate [summary scope]")
    # In a minutes-style summary, a declarative future is conventionally a
    # commitment claim.  In meeting talk, generic "will" is far noisier (plans,
    # predictions, and descriptions), so do not use it as a source-side cue.
    if genre == "summary" and not decision_to_deliberate and re.search(r"\b(?:will|shall)\b", t):
        hits["DECIDED"].append(r"\b(?:will|shall)\b [summary genre]")

    if negated_decision:
        state = "OPEN"
    elif hits["REJECTED"]:
        state = "REJECTED"
    elif hits["CONDITIONAL"] and (hits["DECIDED"] or hits["PROPOSED"]):
        state = "CONDITIONAL"
    elif hits["TENTATIVE"]:
        state = "TENTATIVE"
    elif hits["DECIDED"]:
        state = "DECIDED"
    elif hits["PROPOSED"]:
        state = "PROPOSED"
    elif hits["OPEN"]:
        state = "OPEN"
    else:
        state = "UNKNOWN"

    matched = tuple(key for key, values in hits.items() if values)
    explicit = negated_decision or bool(matched)
    terminal_branches = sum(bool(hits[key]) for key in ("REJECTED", "DECIDED", "OPEN"))
    ambiguous = terminal_branches > 1 and not negated_decision
    return StateParse(
        state=state,
        matched=matched,
        conditional=bool(hits["CONDITIONAL"]),
        negated_decision=negated_decision,
        explicit=explicit,
        ambiguous=ambiguous,
    )


def transition(source: str, summary: str) -> dict:
    a = classify_state(source, genre="source")
    b = classify_state(summary, genre="summary")

    # The registered headline phenomenon is specifically an unsupported move
    # to an unconditional decision, not arbitrary movement on a fake total
    # order that treats rejection as "less committed" than proposal.
    unsupported_unconditional_decision = (
        a.explicit
        and not a.ambiguous
        and a.state != "DECIDED"
        and b.state == "DECIDED"
        and not b.conditional
    )
    downgrade = (
        a.state in COMMITMENT_ORDER
        and b.state in COMMITMENT_ORDER
        and COMMITMENT_ORDER[b.state] < COMMITMENT_ORDER[a.state]
    )
    return {
        "source": asdict(a),
        "summary": asdict(b),
        "upgrade": unsupported_unconditional_decision,
        "unsupported_unconditional_decision": unsupported_unconditional_decision,
        "downgrade": downgrade,
        "conditionality_lost": a.conditional and not b.conditional,
        "rejection_flipped_to_decision": a.state == "REJECTED" and b.state == "DECIDED",
        "source_scorable": a.explicit and not a.ambiguous,
    }
