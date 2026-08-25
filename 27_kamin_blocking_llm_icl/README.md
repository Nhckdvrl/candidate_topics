# 27 — Cue Competition / Kamin Blocking in LLM In-Context Associative Learning

**Status: REGISTERED / DIRECT G0 DESIGN NEXT / LOW-PREREQUISITE.**

## Natural scientific question

> **When two cues co-occur with the same outcome equally often, does an already-established predictor block an LLM from learning the new cue–outcome association?**

Sharper theoretical form:

> **Is in-context associative learning governed mainly by exposure/co-occurrence, or is it sensitive to cue competition and prediction error?**

This is a classical learning-theory question. The LLM is the experimental learner, not the source of the question.

## External anchor

NeurIPS 2025 Main, *Large Language Models as Model Organisms for Human Associative Learning*, establishes a controlled synthetic paradigm for studying associative learning in LLMs. The Round-08 collision audit did not find a direct LLM test of classical Kamin blocking / unblocking.

The external paper is a **model-organism / paradigm anchor**, not a claimed blocking seed. Therefore Topic 27 does not require reproducing an upstream blocking effect before its mother G0: the first experiment directly tests the new scientific question.

## Frozen mother contrast

```text
BLOCKING
Phase 1: A -> X repeated
Phase 2: A+B -> X repeated
Test:    B -> ?

MATCHED CONTROL
Phase 1: neutral matched exposure without an established A -> X predictor
Phase 2: A+B -> X repeated
Test:    B -> ?
```

Critical identification requirement:

```text
B-X co-occurrence count is exactly matched.
Only the prior predictive history of A differs.
```

Use nonce cues/outcomes and counterbalanced mappings so lexical semantics do not define the result.

Primary effect:

```text
P(X | B, blocking) < P(X | B, matched control)
```

## Why this is not the archived latent-inhibition route

```text
latent inhibition:
pre-expose target cue B without outcome, then learn B -> X

blocking:
learn A -> X first, then learn compound A+B -> X, test B
```

The manipulation, theory, and predicted learning principle differ. Topic 27 must not reuse the archived latent-inhibition schedule or its post-hoc rescue logic.

## First-shot requirements

Before large-scale model inference:

1. generate synthetic schedules deterministically;
2. verify exact exposure equality for B-X across conditions;
3. verify token-length / cue-position balance under the selected tokenizer;
4. freeze one small panel of open models and one prompt interface;
5. freeze a paper-worthiness effect gate before seeing outcomes;
6. do not search schedules, prompt wordings, cue salience, or models after a negative.

## Positive-paper runway

Only after G0 survives:

- unblocking by changing the outcome after compound introduction;
- overshadowing without Phase-1 prior learning;
- acquisition-vs-performance tests for B-X;
- scale/family generalization;
- representation trajectory and causal intervention only after the behavior is established.

## Kill lines

Stop if:

- a direct recent LLM blocking/unblocking paper is found;
- exact B-X exposure matching cannot be maintained;
- the effect depends on lexical cue meaning or one tokenization;
- the effect exists only after schedule/prompt/model fishing;
- a clean powered G0 is null or opposite.

## Files planned

```text
README.md
make_panel.py
run_g0.py
run_g0.sh
requirements.txt
tests/
```

No hidden-state work is authorized before the behavioral blocking object survives the frozen G0.
