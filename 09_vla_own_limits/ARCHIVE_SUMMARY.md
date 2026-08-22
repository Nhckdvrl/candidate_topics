# Archive Summary — Topic 09: Does a VLA know its own limits?

**Final status: KILLED AT G0 — the identifying contrast does not exist in nature.**

Archived 2026-08-22. The technical stack was built, verified to an unusually strict
standard, and used to run the full frozen discovery panel. The experiment then failed at
the first gate, for the one reason that cannot be engineered around: **the natural
bidirectional competence crossover this design requires is not there.**

Nothing about the hypothesis was tested. That is the correct outcome — the gate exists
precisely so that the representation work is never attempted on a contrast that cannot
identify anything.

---

## 1. The question

> When a VLA carries a signal that predicts eventual success, is that signal specific to
> **this policy's own** chance of succeeding, or is it mostly a policy-agnostic estimate
> that **this state looks hard**?

The identifying idea was to hold the physical state fixed and vary only the checkpoint
within one pi0.5 LIBERO fine-tuning trajectory. If checkpoint A beats B on some states
while B beats A on others, then generic state difficulty is constant inside each pair and
anything that tracks the winner must be policy-specific.

That argument is sound. It just needs the crossover to exist.

## 2. What was run

Frozen protocol (`LOCKED_CONFIG.json`, `topic09-v3`), no deviations:

```text
suite            libero_10, all 10 tasks
states           init_idx 0-14 = 150 physical states
checkpoints      released pi0.5 2k / 3k / 9k
repeats          8 common policy-noise seeds per (state, checkpoint)
total            3600 rollouts, 0 technical failures
robust winner    p_A - p_B >= 0.50
```

## 3. Result

Overall LIBERO-10 success, 1200 rollouts each:

```text
2k  45.7%      3k  87.8%      9k  91.8%
```

Robust state-level winners, out of 150 states:

```text
pair        A-wins   B-wins   ambiguous   bidirectional support
2k vs 3k       0       74         76              0
2k vs 9k       0       77         73              0
3k vs 9k       3        3        144              3
```

The gate needs **15 in each direction**. The best pair reaches **3**.

```text
VERDICT: STOP_NO_NATURAL_CROSSOVER
```

## 4. Why the crossover is absent — two different reasons

**2k is uniformly dominated.** Not "usually worse" — there is not a single state out of
150 where 2k robustly beats either later checkpoint, and only 1-2 states where it is even
*marginally* ahead (by 0.12, i.e. one rollout in eight). Only 4 states sit at joint
ceiling, so this is not a saturation artifact. Along this fine-tuning trajectory,
competence improves **monotonically at the level of individual physical states**, not
merely in aggregate. That is a substantive finding, and it is fatal to the design: a
uniformly better policy gives a constant offset, which is exactly the thing the paired
contrast is built to cancel.

**3k and 9k are too close, and too high.** 60 of 150 states have *both* checkpoints at
100%. 26 states have 3k strictly ahead by some margin, so the trajectory is not perfectly
monotone at this end — but only 3 clear the 0.5 threshold in each direction.

Interestingly those 3+3 states are **real**, not noise: the within-state relabeling null
gives `p = 0.001` against a null mean of 0.24. Genuine crossover exists. There is just far
too little of it to support a representation test.

## 5. What was deliberately not done

The obvious rescues were all available and all refused:

- camera or lighting perturbation to manufacture hard states;
- adversarial or hand-picked initial states;
- training a bespoke weak checkpoint to pair against 9k;
- lowering the 0.5 rate gap after seeing that 0.5 was too strict;
- extending into the reserve states (30-49) to accumulate more crossover — at 3 per 150
  states, doubling the panel yields about 6, still far short of 15;
- switching to cross-architecture policies, where competence, representation space and
  inference algorithm all change together.

Each would have produced a crossover set. None would have produced a *natural* one, and
the whole point of the design was that the contrast be natural.

## 6. What survives

The instrumentation is sound and reusable, and the technical bar it clears is higher than
the topic needed:

- `openpi_instrumented_server.py` — controlled per-decision inference noise plus a purely
  observational action-expert layer hook, with eager execution so the hooked and unhooked
  paths are the same computation;
- `state_contract.py` / `libero_common.py` — settled-MuJoCo-state hashing and the frozen
  official LIBERO reset protocol;
- `noise_null.py` — the within-state relabeling null for crossover claims;
- `preflight.py` / `check_checkpoints_differ.py` — technical identity gates.

Measured, not assumed: same noise seed gives **bit-identical** actions and features on all
three checkpoints; the settled state hash reproduces across freshly constructed
environments; the three checkpoints are distinct, and their pairwise distances order
exactly as a single fine-tuning trajectory should.

## 7. The finding worth carrying forward

Two bugs were caught that would not have announced themselves:

**Environment reuse silently breaks state identity.** After enough episodes in one LIBERO
environment, `reset()` + `set_init_state()` stops reproducing the settled MuJoCo state,
while a freshly built environment reproduces it exactly. This does not raise an error — it
quietly presents two different physical states as one, and the checkpoints being compared
have different episode histories by construction. Caught only because the audit refused to
accept `task_id + init_idx` as proof of state identity.

**The frozen crossover rule was not noise-proof.** With 8 rollouts and a 0.5 gap, two
*equally competent* checkpoints produce a spurious robust win at ~3.8% of states; a
150-state panel would be expected to show ~6 false wins in each direction. On a 500-state
synthetic panel with no competence difference anywhere, noise alone produced 27 and 17
wins — clearing the `>=15` bar in both directions. The original protocol would have called
that a pass.

Ironically, the null control that was added to prevent a false positive is what certified
the 3+3 crossover as real. It simply confirmed there was too little of it.

## 8. Verdict

**KILL.** The question remains open and interesting. This particular way of identifying it
requires a phenomenon — natural bidirectional competence crossover between same-family
checkpoints on identical states — that the released pi0.5 LIBERO trajectory does not
exhibit.
