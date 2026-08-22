# Archive Summary — Topic 09: Does a VLA know its own limits?

**Final status: ARCHIVED / KILLED AT G0 — insufficient natural bidirectional competence crossover.**

Archived 2026-08-22 after the full frozen discovery panel completed. The project did **not** reach the representation hypothesis. It stopped at the prerequisite identification gate because the same-family checkpoint family did not supply enough natural state-level winner reversals to support the intended paired test.

This was not an operationally cheap experiment. Reproducing the OpenPI/LIBERO stack, debugging state identity and stochastic inference, and completing 3,600 rollouts consumed an afternoon of engineering and compute. The lesson is therefore not “the gate was cheap”; it is that **even the first scientific gate can be expensive enough that its prerequisite variation should be checked or evidenced before committing to the full harness.**

---

## 1. Scientific question

> When a VLA carries an internal signal that predicts eventual success, is that signal specific to **this policy's own** chance of succeeding, or is it mostly a policy-agnostic estimate that **this state looks hard/easy**?

The proposed identification was:

1. hold the physical simulator state fixed;
2. change only the checkpoint within one pi0.5 LIBERO fine-tuning trajectory;
3. require natural bidirectional competence crossover:
   - A clearly beats B on some states;
   - B clearly beats A on other states;
4. only then test whether a shared internal success readout flips with the actual winner.

If that crossover exists, generic state difficulty is fixed inside each state pair and cannot by itself explain a winner-following relative signal.

The identification argument is valid. The required natural support was not present.

---

## 2. Frozen G0 that was actually run

Protocol was frozen before the outcome was inspected (`LOCKED_CONFIG.json`, `topic09-v3`):

```text
suite            LIBERO-10, all 10 tasks
states           init_idx 0..14 = 150 physical states
checkpoints      released pi0.5 2k / 3k / 9k
repeats          8 common policy-noise seeds per (state, checkpoint)
total            150 × 3 × 8 = 3600 rollouts
technical fails  0
robust winner    success-rate gap >= 0.50
required support >= 15 robust wins in each direction
```

The stack also enforced realized simulator-state hashes and common policy RNG streams across checkpoint comparisons.

---

## 3. Main result

Overall LIBERO-10 success across 1,200 rollouts per checkpoint:

```text
2k   45.7%
3k   87.8%
9k   91.8%
```

Robust state-level winner counts over the same 150 states:

```text
pair        A-wins   B-wins   ambiguous   bidirectional support
2k vs 3k       0       74         76              0
2k vs 9k       0       77         73              0
3k vs 9k       3        3        144              3
```

Frozen gate:

```text
required bidirectional support = 15
best observed support           = 3
VERDICT                         = STOP_NO_NATURAL_CROSSOVER
```

The best pair was not near the gate: it reached one fifth of the required support.

---

## 4. What the result actually says

### 4.1 The early checkpoint is dominated at the item level

The striking part is not merely that 2k is weaker on average. It produced **zero robust wins** against both 3k and 9k across all 150 physical states.

So the missing crossover is not mainly a power issue. Along this particular fine-tuning trajectory, improvement is close to monotone at the state level: later competence is mostly accumulated rather than exchanged across different subsets of states.

That directly breaks the intended identification axis. A same-state paired design is useful only when policy competence changes *which system wins* across items. A globally weaker checkpoint gives mostly a constant quality ordering, exactly the situation the relative comparison was designed not to mistake for self-knowledge.

### 4.2 The two late checkpoints are too similar and partly saturated

3k and 9k are the only pair with genuine reversals, but they are too close overall. Sixty of 150 states have both checkpoints at 100% success, and only 3 states in each direction exceed the frozen 0.50 success-rate-gap threshold.

Thus the family presents a bad geometry for this question:

- early vs late: enough ability difference, almost no reversal;
- late vs late: some reversal, not enough ability separation/support.

There is no natural same-family pair that gives both properties simultaneously.

### 4.3 The observed 3+3 reversals are real, but that does not rescue the topic

A within-state relabeling null was added because the original winner-count gate was not automatically noise-proof. It showed the observed 3+3 crossover was unlikely under sampling-only reassignment (`p = 0.001`, null mean about 0.24).

So the conclusion is **not** “all crossover was sampling noise.” Genuine checkpoint-specific reversals exist. The problem is scale: six total robust reversal states cannot support the planned shared-readout test or a credible independent confirmation.

---

## 5. A validation flaw discovered during the run

The original frozen intuition treated a `>= 0.50` success-rate gap over 8 stochastic rollouts as obviously robust. That was false.

For two equally competent policies around `p=0.5`, finite-sample noise alone can create such an apparent robust winner in roughly 3.8% of states. On a 150-state panel this means several false wins per direction are expected; on a 500-state synthetic null panel, winner counts can even exceed the original `>=15` support threshold.

Therefore a count-of-winners gate needs an explicit null model, not just a large-looking per-state threshold. The relabeling null fixed this before the scientific decision was made.

This is an important reusable lesson because the final KILL becomes **more** trustworthy after accounting for this flaw: the few observed reversals survived the null, but remained far too sparse.

---

## 6. Important engineering finding: state identity cannot be assumed

The audit also found that long-lived LIBERO environment reuse can silently violate realized state identity. After enough episodes, `reset()` + `set_init_state()` can fail to reproduce the same settled MuJoCo state even though the nominal task/init identifier is unchanged.

A freshly constructed environment reproduced the state correctly.

For any future experiment claiming “same state, different policy/model/intervention”:

- hash the realized settled simulator state;
- verify the hash across compared conditions;
- rebuild the environment when necessary;
- do not trust an integer state/index identifier as physical equality.

Without this check, a beautiful paired result could simply compare different physical states.

---

## 7. Why we do not rescue it

The following would increase crossover counts but change the scientific object:

- camera/lighting perturbations;
- adversarial or hand-picked hard initial states;
- training a bespoke intermediate/weak checkpoint;
- lowering the 0.50 gap after observing the data;
- mining the reserve split for more favorable states;
- switching to unrelated VLA architectures where checkpoint identity, representation space and inference procedure all change together.

These are not forbidden because they are technically invalid. They are rejected because the original claim was about **naturally occurring policy-specific competence on identical states**. If the natural contrast is absent, manufacturing the contrast makes the project answer a different question.

G1 hidden-state extraction, the shared probe and independent confirmation were therefore not run.

---

## 8. What was wrong with the topic-selection reasoning

The key selection mistake happened **before code**.

We had evidence for two aggregate facts:

1. VLA representations can predict eventual success;
2. different checkpoints have different overall task competence.

We then implicitly assumed a third, much stronger instance-level fact:

> different same-family checkpoints will naturally trade wins across many identical states.

That implication does not follow.

Aggregate performance difference says nothing about the joint per-state ordering. A family can improve almost monotonically item-by-item, which is exactly what happened here.

For future candidates, whenever identification relies on natural disagreement / reversal / crossover / transition, require evidence about **the joint instance-level distribution**, not merely different aggregate scores.

A stronger screening rule is:

> **Before building a measurement around natural crossover, first establish that the crossover population itself exists at useful density.**

And a resource rule must accompany it:

> **A conceptually simple first gate can still be operationally expensive. Estimate wall-clock, simulator rollout count and engineering setup before calling a screen “cheap.”**

---

## 9. Reusable assets

Although the topic is archived, several pieces remain useful:

- `openpi_instrumented_server.py` — deterministic per-decision noise plus observational action-expert hooks;
- `state_contract.py` / `libero_common.py` — settled-state hashing and LIBERO reset/preprocessing contract;
- `noise_null.py` — within-state relabeling null for stochastic winner counts;
- `preflight.py` / `check_checkpoints_differ.py` — inference identity and checkpoint-distinctness gates;
- common-noise rollout infrastructure for fair stochastic-policy comparisons.

These tools are reusable only when a future question independently supplies a natural scientific contrast.

---

## 10. Final interpretation

**KILL / ARCHIVE.**

This result does **not** falsify the broader question of whether a VLA can represent its own competence. It falsifies the practicality of this particular clean identification strategy on the released pi0.5 LIBERO checkpoint trajectory.

The stopping reason is precise:

> **The natural bidirectional same-state competence crossover required by the paired design is too sparse.**

Do not revive Topic 09 by post-hoc threshold/model/state mining. Only a genuinely new external observation demonstrating abundant natural policy-specific reversals would justify a new candidate.