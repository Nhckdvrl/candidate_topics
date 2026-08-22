# 09 — Does a VLA know its own limits?

## Status

**Candidate / identification-first. No mechanism claim yet.**

The topic asks one question:

> When a VLA carries a signal that predicts eventual success, is that signal specific to **this policy's own chance of succeeding**, or is it mostly a policy-agnostic estimate that **this state looks easy / hard**?

That distinction matters because recent work increasingly calls these internal signals *confidence*, *self-evaluation*, *value-like structure*, or *reliability*. A generic state-difficulty signal is still useful, but it is not the same thing as a policy knowing its own limits.

## Why this is not another “invent a phenomenon” topic

The prerequisite phenomenon already exists in the literature:

- frozen VLA representations linearly predict eventual task success, including after same-task / same-timestep matching;
- the same paper finds comparable success information in DINOv2 and CLIP, which makes the interpretation ambiguous;
- VLAConf and FabriMAE explicitly turn internal VLA features / attention statistics into task-success confidence or self-evaluation signals;
- Foresight shows that failure detectors can transfer across policies, consistent with a large policy-agnostic execution-difficulty component.

So we do **not** need to prove from scratch that success is decodable. We only ask what kind of information that success signal is.

## The one clean contrast

Use **the same physical simulator state** and **the same VLA family**, but different checkpoints with different learned competence.

For checkpoint pair `A, B` at state `s`:

```text
same task + same simulator state
        |
        +--> checkpoint A --> hidden state h_A(s) --> rollout outcome y_A(s)
        |
        +--> checkpoint B --> hidden state h_B(s) --> rollout outcome y_B(s)
```

We care about naturally occurring crossover states:

```text
A succeeds, B fails
```

and also

```text
A fails, B succeeds.
```

Bidirectional crossover is essential. If the later checkpoint always wins, a “confidence” readout can succeed by learning only that one checkpoint is globally better. We do not manufacture crossover states; if they are not naturally present at useful scale, the experiment is not identifiable and stops.

## Primary experimental axis

The preferred first family is **pi0.5 on LIBERO**, because COAST released checkpoints from the *same fine-tuning trajectory* at steps **2k, 3k, and 9k** and reports that early checkpoints already have mixed success/failure behavior. This is cleaner than comparing OpenVLA against pi0.5, where architecture identity and competence are inseparable.

The initial state / task / simulator seed is held fixed across checkpoints. Start with standard LIBERO evaluation states; do not add perturbations merely to manufacture disagreement.

## G0 — behavioral identifiability only

Before touching hidden states:

1. run the released 2k / 3k / 9k checkpoints on exactly the same LIBERO states;
2. record only `task, seed, checkpoint, success`;
3. measure pairwise disagreement and, crucially, two-way crossover counts;
4. select a checkpoint pair **on discovery states only** by maximum bidirectional support;
5. freeze that pair before representation confirmation.

The code in `src/analyze_disagreement.py` implements this screen.

If there is no checkpoint pair with enough natural two-way disagreement, stop. Do not rescue the topic with camera noise, adversarial states, hand-picked failures, or new checkpoints trained to create the contrast.

## G1 — does the hidden success signal follow *whose* success?

Only after G0 passes.

Use one frozen representation location chosen from prior work rather than layer-searching. The current primary choice is the pi0.5 action-expert hidden state around **layer 11**, motivated by COAST's released LIBERO analysis. Exact hook semantics must be checked against the released code before collection.

Fit **one shared linear success probe** across the two frozen checkpoints on discovery states. The decoder is shared; we do not fit one custom probe per checkpoint.

On independent same-state crossover states, compute:

```text
q_A = shared_probe(h_A(s))
q_B = shared_probe(h_B(s))
relative_score = q_A - q_B
```

Then ask only:

> When `A` succeeds and `B` fails, is `q_A > q_B`; and when the winner flips, does the sign flip too?

Primary statistic: AUROC of `q_A - q_B` for predicting which checkpoint wins on crossover states.

Why this contrast is useful:

- state difficulty is identical within each pair;
- task and image are identical within each pair;
- the same decoder is applied to both checkpoints;
- two-way crossover prevents a constant checkpoint-quality prior from solving the task.

If the paired relative score cannot predict the winner materially above chance, then the success signal may be useful but the strong “self-knowledge / own limits” interpretation is unsupported.

## What counts as interesting

A weak result such as “hidden states correlate with success” is already known and is not enough.

A positive result worth continuing is:

> On the same physical state, the internal success score changes with which checkpoint is actually capable of succeeding, and the direction reverses on natural crossover states.

That would establish policy-specific competence information rather than generic scene difficulty.

A negative result is also informative:

> Internal “confidence” remains mostly state-driven even when the identity of the successful policy changes.

That would directly qualify the current self-evaluation interpretation in VLA confidence work.

## What we deliberately do **not** do yet

No SAE. No activation steering. No causal mediation. No layer sweep. No robustness benchmark. No new confidence method.

If the paired phenomenon survives independent confirmation, then a mechanism question becomes natural: **where along the VLA does generic state difficulty become policy-specific competence?** Only then do layerwise probing or intervention.

## Files

```text
LITERATURE_AUDIT.md       collision audit and exact novelty boundary
VALIDATION.md             frozen logic of G0/G1 and stop conditions
SERVER_HANDOFF.md         concise execution handoff for the cluster
src/panel.py              same-state checkpoint outcome panel utilities
src/analyze_disagreement.py  G0 behavioral identifiability screen
src/relative_probe.py     shared-probe paired relative-success analysis
requirements.txt
tests/
```
