# 29 — Positional Imprinting of Parametric Knowledge

**Status: REGISTERED / ARTIFACT VERIFIED / REPRODUCTION RECEIPT REQUIRED BEFORE MOTHER G0.**

## Natural scientific question

> **If two facts ultimately receive identical position exposure counts and identical later washout training, does the position at which each fact was first acquired leave a persistent imprint on final parametric accessibility?**

The distinction is:

```text
final / cumulative exposure statistics
vs
historical acquisition path
```

This is a learning-history / hysteresis question, not another measurement of ordinary positional bias.

## External anchor

NAACL 2025 Main / Oral, *Where is the answer? An empirical study of positional bias for parametric knowledge extraction in language model*.

Official artifact already verified in advisor search:

```text
repository: omron-sinicx/WhereIsTheAnswer
frozen upstream commit: 910fcddec93f7400b58257d70abf1dab31f1e179
```

The seed establishes that where a fact appears within training text affects later closed-book extraction. Topic 29 asks whether an **early acquisition-position history** remains after subsequent exposure statistics are deliberately equalized.

## Registration and receipt rule

Registration does **not** authorize the mother G0 yet.

Required sequence:

```text
exact upstream reproduction receipt
        ↓ PASS only
metadata / counterbalancing preflight
        ↓ PASS only
within-model positional-imprinting G0
```

If the selected executable seed relation does not reproduce after one justified engineering repair, stop the local route. Do not change model, data subset, seed, prompt, document construction, or thresholds to rescue it.

## Intended mother G0

Within one training run, counterbalance facts into two groups:

```text
Group E: first acquired in an early document position
Group L: first acquired in a late document position
```

Then equalize later exposure so the groups receive:

- identical total number of exposures;
- identical total early/late position counts;
- identical later washout phase;
- the same optimizer / learning-rate trajectory;
- matched document/fact properties.

Primary contrast after washout:

```text
closed-book QA(Group E) - closed-book QA(Group L)
```

The preferred design is **within-model fact-level counterbalancing** so treatment is not confounded with independent optimizer trajectories.

## Why the result would matter

A persistent difference after final exposure statistics are equalized would show that parametric knowledge access depends on the **path by which the fact was acquired**, not only on the final training-data marginal distribution.

A clean null after successful seed reproduction and sufficient power is also informative: the ordinary positional effect may be reversible once later evidence equalizes position statistics.

## Positive-paper runway

Only if G0 survives:

- washout / reversibility law;
- sensitive period: how many early exposures are required?;
- immediate interleaving vs delayed equalization;
- replay/interleaving mitigation;
- storage-vs-access characterization;
- representation work only after the behavioral effect stands;
- model/domain generalization.

## Kill lines

Stop if:

- the exact upstream positional seed relation fails locally;
- fact groups cannot be counterbalanced without document/optimizer confounds;
- total exposure or token-context statistics remain unequal;
- the imprint appears only across independent training runs but not within-model fact groups;
- a direct recent paper occupies first-acquisition position after equalized later exposure.

## Next files

```text
README.md
SEED_RECEIPT.md
run_seed_receipt.sh
prepare_counterbalanced_panel.py
run_g0.py
run_g0.sh
requirements.txt
tests/
```

No mother G0 or mechanism work is authorized before the frozen receipt passes.
