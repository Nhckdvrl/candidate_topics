# Mother-Topic Branching Policy

This policy is added after repeated candidate failures where a clean G0 killed the entire project because the topic was only one narrow causal arrow.

The goal is **not** to make topics unfalsifiable. Individual hypotheses must still be killable. The change is to choose a stronger scientific object before writing those hypotheses.

## Core distinction

Bad project shape:

```text
seed phenomenon
→ one guessed explanation
→ one G0
→ G0 fails
→ entire topic disappears
```

Preferred project shape:

```text
reproduced mother phenomenon / mother contradiction
        ↓
several independent scientific questions
        ├── boundary / characterization
        ├── mechanism
        ├── intervention
        ├── training / development
        └── generalization / system implication
```

A branch may fail without logically erasing the mother phenomenon or the other branches.

## Promotion requirement

After the existing `REPRODUCTION_RECEIPT` passes, but before a numbered topic is registered, write a **Research Branch Map**.

A strong candidate should normally satisfy all of the following:

1. The mother object itself is already externally anchored and reproducible.
2. At least **three genuinely different branches** follow naturally from it.
3. The branches are not three parameterizations of the same hypothesis.
4. At least one branch is behavioral / descriptive and does not require a hidden-state interpretation.
5. At least one branch exposes a mechanism or causal intervention.
6. At least one branch exposes a practical training / inference / system lever.
7. Failure of one branch must not logically falsify the other branches.
8. The whole object is broad enough for ACL / EMNLP / NAACL main-paper scale, while each experiment remains individually clean.

## Anti-abuse rule

Branching is **not** permission to rescue a failed hypothesis.

After a frozen branch fails:

- do not tune that branch;
- do not rename its variables;
- do not search models/layers/prompts for a positive result.

A different branch is justified only if it was part of the branch map **before** seeing the failed result or if a new external observation changes the scientific premise.

## Good mother-object shapes

Prefer:

- the same intervention helps one natural regime and hurts another;
- a capability collapses and later recovers;
- aggregate improvement coexists with a large, distinct behavioral loss;
- two nominally related capabilities diverge under a clean natural axis;
- a stable failure remains after the obvious explanation has been removed;
- two published results genuinely conflict on the **same system/object**, with an accessible matched experiment capable of locating the boundary.

Avoid:

- `paper A shows X + paper B shows Y → maybe Z links them` without a shared experimental object;
- one benchmark error category inflated into a mother topic;
- generic `represented but not used` stories without a new direct behavioral dissociation;
- a branch map whose only entries are probe / SAE / patching / steering.

## Required pre-registration card

```text
Mother phenomenon:
Exact reproduction receipt:
Why surprising / important:

Branch A — characterization / boundary:
Branch B — mechanism:
Branch C — intervention / method:
Optional Branch D — training / development:
Optional Branch E — generalization / system implication:

What result kills only A but leaves B/C meaningful?
What result would kill the mother topic itself?
Exact 2025–2026 collision audit:
```

Only after both the reproduction receipt and this branching audit pass may the project enter a numbered topic directory.
