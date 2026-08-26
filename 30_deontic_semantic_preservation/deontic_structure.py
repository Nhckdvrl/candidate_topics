from __future__ import annotations
import re
from dataclasses import dataclass,asdict

MODAL_PATTERNS={
 'PROHIBITION':[r'\b(?:must not|shall not|may not|is prohibited from|are prohibited from|cannot)\b'],
 'OBLIGATION':[r'\b(?:must|shall|required to|is required to|are required to|has to|have to)\b'],
 'PERMISSION':[r'\b(?:may|is permitted to|are permitted to|is allowed to|are allowed to|can)\b'],
 'ENTITLEMENT':[r'\b(?:is entitled to|are entitled to|eligible to|has the right to|have the right to)\b'],
}
COND=r'\b(?:if|unless|provided that|provided however|subject to|on condition that|in the event that|when|only if)\b'
EXC=r'\b(?:except|except that|other than|save for|notwithstanding|provided however)\b'

@dataclass(frozen=True)
class Deontic:
    modality:str
    conditional:bool
    exception:bool
    negation:bool

def extract(text:str)->Deontic:
    t=' '.join(text.lower().split()); modality='NONE'
    for m,pats in MODAL_PATTERNS.items():
        if any(re.search(p,t) for p in pats): modality=m; break
    return Deontic(modality,bool(re.search(COND,t)),bool(re.search(EXC,t)),bool(re.search(r'\b(?:not|never|no)\b',t)))

def compare(original:str,simplified:str)->dict:
    a,b=extract(original),extract(simplified)
    return {'original':asdict(a),'simplified':asdict(b),'modality_changed':a.modality!=b.modality,'condition_lost':a.conditional and not b.conditional,'exception_lost':a.exception and not b.exception,'negation_changed':a.negation!=b.negation}
