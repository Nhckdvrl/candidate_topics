# When Does a Fact Become Recallable?

**Status:** `PROVISIONAL SURVIVOR — DEEP-AUDIT PASSED ROUND 02`

This is **not** a registered numbered Topic. It survived the first phenomenon / collision / identification audit and is a candidate for the next G-0 design pass.

---

## One-sentence question

> **During pretraining, does a factual proposition become encoded before it becomes reliably recallable from a semantically different query, and if so, what does the per-fact encoding→recall transition look like?**

The object is not generic factuality and not another knowledge-probing benchmark. The object is the **developmental gap between storage and usable access**.

---

## Why this question exists: a very close seed paper

The direct seed is Calderon et al. (2026), **[Empty Shelves or Lost Keys? Recall Is the Bottleneck for Parametric Factuality](https://research.google/blog/empty-shelves-or-lost-keys-recall-is-the-bottleneck-for-parametric-factuality/)**.

That work introduces **knowledge profiling** and separates factual states that ordinary QA accuracy collapses together:

```text
encoding failure
recall failure
 direct recall
 recall with thinking
 inference without encoding
```

Its key empirical result is already strong: on WikiProfile, frontier systems encode almost all tested facts, yet a substantial fraction remain inaccessible under direct QA. The paper therefore establishes that:

```text
stored fact ≠ reliably accessible fact
```

This is exactly the kind of standing measurement / phenomenon that the labmate-topic search suggests building from.

### Labmate-style one-step rotation

The seed paper studies **final models / model scale**.

The proposed rotation is only:

```text
final-model knowledge profile
        ↓
knowledge profile over pretraining time
```

This is structurally similar to the healthy `human developmental age → LM training checkpoint` move observed in the lab: reuse a scientific object that already exists and change one meaningful axis rather than inventing a new latent bridge.

---

## Why the answer would matter

There are at least two qualitatively different pictures of factual learning.

### Picture A — acquisition and access are effectively synchronous

```text
not encoded
   ↓
encoded + recallable
```

Then the large final-model recall gap may mainly be a later consequence of scale, interference, post-training, or query mismatch. There is no separate developmental consolidation phase to explain.

### Picture B — storage precedes usable access

```text
not encoded
   ↓
encoded but inaccessible
   ↓
reliably recallable
```

Then factual learning has a real **encoding→access lag**. That changes what “learning a fact” means in training-dynamics studies: first exposure / memorization and usable knowledge are different events.

A robust result would connect two currently separated literatures:

- factual acquisition during pretraining;
- final-model knowledge-access / recall failures.

Importantly, the connection is not assumed. The first experiment directly asks whether the intermediate state exists often enough to be a phenomenon.

---

## Existing neighboring work / collision audit

### 1. The seed itself — close but does not use training time

- Calderon et al. (2026), *Empty Shelves or Lost Keys?* — establishes the encoding/recall distinction across final models and scale, and releases WikiProfile.

### 2. Factual acquisition over checkpoints already exists

- Liu et al. (Findings EMNLP 2025), **[Tracing Multilingual Factual Knowledge Acquisition in Pretraining](https://aclanthology.org/2025.findings-emnlp.113/)**, follows OLMo-7B checkpoints and shows that factual recall and cross-lingual consistency improve during training, strongly related to corpus frequency.
- Controlled factual-injection studies also follow memorization / generalization / forgetting during pretraining.

These papers are an important collision, but their primary observable is **whether a fact can be recalled**. They do not, as far as this audit found, apply the seed paper's fact-level separation of **encoding vs access** at each checkpoint.

### 3. Knowledge access in final models is active

Work on hidden factual knowledge, thinking-assisted recall, and reasoning-LM knowledge access already asks how inaccessible knowledge can be recovered at inference time.

That increases the importance of the proposed developmental question but also means the project must not drift into another inference-time retrieval method.

### Collision boundary

The candidate survives only as the following broad question:

> **Do facts pass through a stable encoded-but-not-recallable developmental state during pretraining?**

If a direct 2025–2026 paper is found that already profiles encoding and recall jointly over open pretraining checkpoints, kill this candidate rather than narrowing it to one relation, language, prompt, or checkpoint family.

---

## The main conceptual attack

### Attack: “encoding” may just mean prompt continuation, not parametric storage

The seed paper operationalizes encoding behaviorally by reconstructing the factual proposition in a pretraining-like context. At a checkpoint, success could reflect local lexical continuation rather than a meaningful stored fact.

This is the central identification risk.

### Required defense

Do **not** rely on one exact cloze.

For every candidate fact, require convergent evidence from at least two encoding probes that retain the source proposition while varying nuisance surface form, then separately evaluate access using direct/reverse questions whose wording is deliberately different.

The claim must stay behavioral:

> “encoded under the validated WikiProfile-style operationalization”

—not “we have proven the fact exists in a specific neural memory slot.”

If this operational distinction is unstable across mild paraphrases, kill the project.

---

## Cheapest decisive G-0

### Experimental object

Prefer **OLMo / another checkpoint-rich model whose pretraining corpus is inspectable enough to verify fact exposure**.

Do not begin with Pythia merely because checkpoints are convenient if exact training-data provenance makes the fact-exposure audit weak.

### Data

Start with roughly `200–500` natural factual propositions for which:

1. the relevant source text is present in the training corpus;
2. exposure order / approximate frequency can be estimated;
3. two independent encoding probes can be constructed;
4. direct and reverse QA probes are unambiguous.

WikiProfile is the natural starting instrument, but its facts must be intersected with the chosen model's real pretraining corpus and cutoff. Do not blindly run the full benchmark on a model trained on a different temporal snapshot.

### Per-checkpoint state

For fact `f` at checkpoint `t`, derive only a small state machine:

```text
U = not robustly encoded
E = encoded, not recallable
R = encoded and recallable
```

Optional thinking-assisted recovery is secondary and should not be required for G-0.

### Primary observable

For each fact that becomes `R`, estimate:

```text
t_encode(f)
t_recall(f)
lag(f) = t_recall(f) - t_encode(f)
```

The first figure should be the distribution of this lag, not a probe correlation matrix.

### Strong positive

A meaningful fraction of naturally exposed facts show a reproducible positive encoding→recall lag across probe variants, and the ordering is stable enough that `E` is a real transient state rather than threshold noise.

### Kill line

Kill / sharply downgrade if any of the following holds:

- encoding and recall transitions are essentially synchronous at available checkpoint resolution;
- the apparent lag disappears under mild encoding-probe paraphrase;
- most facts oscillate among states so strongly that transition time is not identifiable;
- the only clean effect appears in synthetic injected facts while natural-corpus facts do not support it.

No mechanism work should begin before this.

---

## Why this is not the failed “invent a bridge” pattern

We are **not** saying:

> paper A studies acquisition + paper B studies access → perhaps there is a mysterious consolidation mechanism.

Instead, a single 2026 seed paper already demonstrates the exact behavioral state `encoded but inaccessible`. We only ask whether that established state has a **developmental history** during pretraining.

The first experiment can answer “no” cleanly.

---

## Data / compute fit

- public checkpoint families exist;
- natural factual datasets exist;
- training-data audit can be automated for a moderate fact set;
- first pilot is inference over checkpoints, not new pretraining;
- if the phenomenon survives, controlled small-scale pretraining / factual injection can later isolate exposure schedule and frequency.

This is compatible with a solo project and a 1–2 week first pilot.

---

## Interestingness test

Assume the cleanest positive result:

> **For many natural facts, a model first becomes able to reproduce the fact in its learned source-like context, but only substantially later becomes able to answer a semantically reformulated question about it.**

That would be worth knowing. It means the moment a model **stores** information and the moment it can **use** that information are separable stages of pretraining, not merely two evaluation views of the same final capability.

A clean negative result is also useful: it would localize the final-model `lost keys` phenomenon to later interference / access conditions rather than a generic acquisition stage.

---

## Current verdict

`KEEP — HIGH PRIORITY`

Among Round-02 candidates this is one of the strongest because it has:

- a very recent seed with an already-established phenomenon;
- an exact one-axis rotation;
- open experimental objects;
- a first experiment with a genuine kill outcome;
- a question that survives model turnover.

Next step is **not method design**. Next step is to freeze the model/data pair and audit whether WikiProfile-style encoding can be made valid at checkpoint level.