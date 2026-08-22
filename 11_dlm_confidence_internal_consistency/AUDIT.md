# Topic 11 pre-run validation audit

Date: 2026-08-22

This audit was performed before any Topic-11 model score was inspected.

## Goal

Make the first experiment capable of killing or supporting the research question without post-hoc metric search.

## Risks found in the first implementation

### 1. Environment split with Topic 10

Topic 11 pinned `transformers==4.38.2`, while Topic 10 uses a modern `transformers>=4.49` environment. That would unnecessarily create a second LLaDA environment and cache workflow.

**Fix:** Topic 11 now uses a compatible modern range and is explicitly designed to reuse Topic 10's environment.

### 2. Padding / attention-mask dependence

The original scorer right-padded variable-length batches and relied on the remote LLaDA implementation's `attention_mask` path. LLaDA remote code has changed around attention-mask support, so this introduces an avoidable compatibility variable.

**Fix:** examples are bucketed by exact token length. Every model batch is rectangular without padding; no attention mask is needed.

### 3. Full-tail averaging can dilute the relevant signal

Most continuation tokens are boilerplate (`Step`, punctuation, operators). A genuine arithmetic-consistency effect could be diluted by averaging them together.

**Fix:** the design records exact character spans of every downstream arithmetic result; the tokenizer maps those frozen spans to token positions. `confidence_result` is therefore deterministic and judge-free.

### 4. First-step effects can be shallow locality

A mismatched announced state is adjacent to Step 1. A Step-1 confidence drop alone could be local numeric compatibility rather than a trajectory-level signal.

**Fix:** `confidence_result_late` excludes the first result and is the primary identification metric. First-result, final-result, all-result, all-tail and full-output scores are all produced from the same forward pass and reported transparently.

### 5. A scorer bug could masquerade as a null research result

A wrong chat template, incompatible model revision, or incorrect final-forward probability implementation could return a clean null and wrongly kill the topic.

**Fix:** the same loaded model runs a 100-pair arithmetic correct-vs-wrong positive control modeled on the seed paper's intervention. If this prerequisite does not reproduce a stable positive gap, the verdict is `INVALID_PROTOCOL_DO_NOT_INTERPRET` rather than a scientific negative.

### 6. Text-space matching does not imply token-space matching

Changing `23` to `29` can change multiple tokens under some tokenizers/contexts.

**Fix:** both mirrored orientations undergo a tokenizer-level audit. The default G-0 admits only one-token prompt and announcement interventions with identical lengths, identical downstream token IDs, and identical scored result positions.

### 7. Orientation-level pseudoreplication

The two mirrored orientations share the same anchor pair and operation chain.

**Fix:** effects are averaged across orientations first. Bootstrap and sign-flip inference operate on anchor-pair effects only.

## Locked G-0 stack

1. unit tests;
2. deterministic design construction;
3. tokenizer-only eligibility audit;
4. one model load per GPU;
5. arithmetic scoring positive control on shard 0;
6. factorial scoring on pair-sharded examples;
7. pair-level bootstrap + sign-flip diagnostics;
8. preregistered verdict.

No generation, training, hidden states, learned judge, threshold search or prompt sweep is required.

## Remaining limitations accepted for G-0

- The task is synthetic arithmetic rather than free-form GSM8K. This is intentional for identification; ecological generality is a later question only if G-0 is strong.
- `CW - IC` compares contradictions located in different textual regions (prompt vs trajectory). Therefore it is treated as the **strong** result, not the minimum evidence for an internal-consistency signal.
- LLaDA is the first model family because the motivating seed paper centers on it. Cross-DLLM replication is not required to decide whether the phenomenon exists at all.

Do not add more controls before seeing G-0. If the locked late-result effect is absent, archive the topic.
