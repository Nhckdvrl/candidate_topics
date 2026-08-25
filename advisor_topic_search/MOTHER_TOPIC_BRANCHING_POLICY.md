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
9. The released artifact must expose the **instance-level fields required by the first frozen contrast**, not merely a benchmark-level or chain-level proxy.
10. Before registration, a metadata-contract preflight must show that the exact critical cell is constructible from released artifacts without changing field semantics, matcher semantics, sample size, or dataset after seeing support.

## Metadata-contract preflight

`artifact public` is **not** equivalent to `experiment executable`.

Before a mother topic is registered, audit the exact contract needed by the first branch:

```text
required unit of analysis:
required instance-level fields:
required labels / counterfactual values:
required pairing keys:
required model outputs, if any:
exact eligible support before model inference:
```

The preflight must run **before tokenizer/model download whenever possible**.

Hard warning signs:

- the paper describes a field that is absent from the released artifact;
- the needed value exists only at a coarser granularity than the frozen experiment;
- a chain-level/document-level field would need to be reused as a turn-level/example-level label;
- the official evaluator reconstructs a value from hidden or unreleased metadata;
- eligibility depends on post-hoc matching, aliases, semantic mapping, or outcome-dependent filtering;
- exact support cannot be counted before inference.

If the frozen experiment requires substituting a coarser field, changing dataset, relaxing matcher semantics, lowering N, or reconstructing missing labels after registration, **stop the registered route**. A new route requires a separately justified measurement object.

### Topic 26 lesson

Topic 26 (`temporal_scope_interference_reinstatement`) passed the high-level artifact audit but failed the exact metadata contract:

```text
raw structural candidates = 324,637
eligible exact-support     = 0
```

The pinned ChronoScope artifact had `present_day_answer` at chain level for many chains but on **zero of 3,335,698 turns**. The registered contrast required turn-level present-day truth. Reusing the chain-level field would have changed the measurement semantics, so the run correctly stopped before tokenization or model inference.

This is an **artifact/measurement stop, not a scientific negative**. The lesson is that metadata semantics must be verified at the exact experimental unit before registration.

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
- a branch map whose only entries are probe / SAE / patching / steering;
- a mother topic whose first scientific branch depends on unreleased or semantically ambiguous metadata.

## Required pre-registration card

```text
Mother phenomenon:
Exact reproduction receipt:
Why surprising / important:

Artifact unit-of-analysis:
Required instance-level metadata:
Exact eligible support before inference:
Metadata-contract verdict:

Branch A — characterization / boundary:
Branch B — mechanism:
Branch C — intervention / method:
Optional Branch D — training / development:
Optional Branch E — generalization / system implication:

What result kills only A but leaves B/C meaningful?
What result would kill the mother topic itself?
Exact 2025–2026 collision audit:
```

Only after the reproduction receipt, metadata-contract preflight, and branching audit pass may the project enter a numbered topic directory.
