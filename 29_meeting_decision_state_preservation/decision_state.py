from __future__ import annotations
import re
from dataclasses import dataclass, asdict

STATE_ORDER = {"REJECTED":0,"OPEN":1,"PROPOSED":2,"TENTATIVE":3,"CONDITIONAL":4,"DECIDED":5}
_PATTERNS = {
    "REJECTED": [r"\b(reject(?:ed|s)?|declin(?:e|ed)|decided not to|will not|won't)\b"],
    "CONDITIONAL": [r"\b(if|unless|provided that|subject to|pending|once|after .* approval|contingent on)\b"],
    "TENTATIVE": [r"\b(tentative(?:ly)?|probably|likely|for now|provisionally|lean(?:ing)? toward)\b"],
    "DECIDED": [r"\b(decid(?:e|ed)|agreed|agreement|will|shall|final(?:ly)?|confirmed|resolved)\b"],
    "PROPOSED": [r"\b(propos(?:e|ed|al)|suggest(?:ed|ion)?|recommend(?:ed|ation)?|could|should we|how about|what if|might)\b"],
    "OPEN": [r"\b(open question|undecided|not decided|to be decided|tbd|still discussing|discuss further)\b"],
}

@dataclass(frozen=True)
class StateParse:
    state: str
    matched: tuple[str, ...]
    conditional: bool
    negated_decision: bool

def classify_state(text: str) -> StateParse:
    t = " ".join(text.lower().split())
    negated_decision = bool(re.search(r"\b(?:not|n't)\s+(?:yet\s+)?(?:decided|agreed|confirmed|resolved)\b", t))
    hits = {k: [] for k in _PATTERNS}
    for state, pats in _PATTERNS.items():
        for pat in pats:
            if re.search(pat, t):
                hits[state].append(pat)
    if negated_decision: state = "OPEN"
    elif hits["REJECTED"]: state = "REJECTED"
    elif hits["CONDITIONAL"] and (hits["DECIDED"] or hits["PROPOSED"]): state = "CONDITIONAL"
    elif hits["TENTATIVE"]: state = "TENTATIVE"
    elif hits["DECIDED"]: state = "DECIDED"
    elif hits["PROPOSED"]: state = "PROPOSED"
    elif hits["OPEN"]: state = "OPEN"
    else: state = "OPEN"
    return StateParse(state, tuple(k for k,v in hits.items() if v), bool(hits["CONDITIONAL"]), negated_decision)

def transition(source: str, summary: str) -> dict:
    a, b = classify_state(source), classify_state(summary)
    return {"source": asdict(a), "summary": asdict(b), "upgrade": STATE_ORDER[b.state] > STATE_ORDER[a.state], "downgrade": STATE_ORDER[b.state] < STATE_ORDER[a.state], "conditionality_lost": a.conditional and not b.conditional}
