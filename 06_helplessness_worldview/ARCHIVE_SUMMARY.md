# Archive Summary — Topic 06: When Does Helplessness Become a Worldview?

**Final status: ARCHIVED / KILLED AT ACQUISITION PREMISE**

**The higher-order worldview/generalization hypothesis was not cleanly tested.** Across the original Qwen3-8B pilot and the one permitted preregistered Qwen3-32B v2 acquisition gate, the agent did not develop a sufficiently strong controllability-dependent behavioral separation during training. The project is therefore closed rather than rescued with additional models, probes, memories, reward sweeps, or environment redesigns.

## Original question

The topic began from a natural learned-helplessness question:

> When repeated experience says that actions do not affect outcomes, what determines whether that belief remains local to one situation or generalizes into a broader expectation that actions usually do not matter?

The specific hypothesis was that, holding experience amount fixed, uncontrollability distributed across many semantically different task families would generalize more strongly to a novel controllable task than the same uncontrollability concentrated in one family.

The intended causal chain was:

```text
experience with action–outcome contingency
        ↓
learn local controllability / uncontrollability
        ↓
transfer that belief to a novel task
        ↓
experience diversity changes breadth of transfer
```

The key point in the final diagnosis is that the first learned-behavior link never became strong enough. Once that prerequisite failed, the later worldview question was no longer cleanly instantiated in the chosen LLM-agent system.

## Experimental design

The registered experiment used a 2×2 master–yoked design:

| | one task family | many task families |
|---|---:|---:|
| controllable | C1 | C10 |
| uncontrollable | U1 | U10 |

Within each diversity level, the uncontrollable session replayed the exact success/failure sequence of its paired controllable master. Thus external reward exposure was matched; only action–outcome contingency differed.

All four cells had matched episode counts, trial counts, action-set cardinality, reward schedule, latent randomization, and an identical held-out `orbital_station` test. The locked primary endpoint was active intervention on novel-test step 1, before receiving any test feedback.

## v1 — Qwen3-8B

### Technical status

The initial Qwen3 thinking-format failure was repaired by disabling hidden thinking for the short JSON action response. The repaired preflight and full S2 pilot were technically clean:

- invalid action rate: `0%`;
- yoke mismatch count: `0`;
- distributed-vs-concentrated master success exposure gap: about `1.32pp` in S2.

### Training acquisition

Late-training active intervention differed only weakly between controllable and yoked-uncontrollable histories:

```text
concentrated: C1 67.6% vs U1 65.2%   (+2.4pp)
distributed:  C10 70.5% vs U10 70.2% (+0.3pp)
```

Late effective-action selection was similarly close:

```text
concentrated: C1 34.1% vs U1 32.5%
distributed:  C10 38.2% vs U10 37.9%
```

The agent therefore showed little evidence that it had robustly acquired the intended controllability distinction.

### Novel-test result

The locked test-step-1 action was also near ceiling:

```text
C1  = 94%
U1  = 92%
C10 = 100%
U10 = 100%
```

This gave:

```text
H1  =  +2pp
H10 =   0pp
D   =  -2pp
95% bootstrap CI for D = [-8pp, +4pp]
```

Pooled transfer was only `1pp`, triggering the registered stop/downgrade rule.

## Why one v2 was allowed

Two independently motivated defects prevented the v1 null from being treated as a final answer to the natural question.

First, adjacent in-context RL evidence used substantially larger Qwen3 models, while the initial pilot used Qwen3-8B. The weak training acquisition therefore could plausibly reflect insufficient learner capability rather than absence of the underlying phenomenon.

Second, the original intervention cost was only `1` while success paid `10`. In the novel task, active intervention had such favorable expected utility that the binary step-1 action was naturally ceilinged. This could hide a moderate shift in prior controllability belief without changing the optimal action.

A final v2 was therefore preregistered before inspection with exactly two changes:

1. `Qwen/Qwen3-32B`, non-thinking inference;
2. intervention cost `2` instead of `1`.

Everything else — four cells, yoking, families, trial counts, reward magnitude, test task, endpoint, temperature and seed block — remained frozen. The v2 protocol explicitly forbade hidden-state probes, self-report, memory summaries, same-family rescue tests, model sweeps, and additional reward/environment tuning.

## v2 — final acquisition gate

The v2 S1 used 24 pairs per diversity, for 96 sessions total.

### Technical gates

All passed:

- 4,416 rows / 96 sessions;
- invalid action rate: `0%` overall and test;
- yoke mismatch count: `0`;
- master success exposure gap: `0.3125pp`.

The cost change also removed the obvious v1 step-1 ceiling.

### Acquisition result

Despite the stronger model and better-calibrated decision cost, the controllability acquisition contrast remained small:

```text
late active:
C1  60.94% vs U1  58.85%  (+2.08pp)
C10 60.42% vs U10 55.73%  (+4.69pp)

late effective action:
C1  33.85% vs U1  31.77%  (+2.08pp)
C10 35.42% vs U10 30.73%  (+4.69pp)
```

This was not a sufficiently clear prerequisite separation to justify scaling into a larger worldview/generalization pilot.

### Frozen novel-test quantities

For completeness, the preregistered primary quantities were:

```text
H1  = +4.17pp
H10 =  0pp
D   = -4.17pp
bootstrap interval for D = [-12.5pp, 0pp]
```

The distributed condition therefore did not show the predicted stronger transfer. However, because the acquisition premise was itself weak, this should not be reported as a clean falsification of the psychological claim that diverse uncontrollability can generalize more broadly.

## Why the topic is closed

At this point the two most defensible rescue explanations from v1 had already been addressed:

- learner size/capability was increased from Qwen3-8B to Qwen3-32B;
- the action utility was recalibrated to remove the primary endpoint ceiling.

Continuing would require an open-ended sequence such as:

```text
try another larger model
→ tune reward/cost again
→ add same-family transfer
→ add memory summaries
→ add explicit belief elicitation
→ add hidden-state probes
→ redesign the environment
```

That would convert a falsification-first topic into post-hoc search for a system in which the desired story appears. It directly violates the repository's no-rescue and complexity-smell rules.

Therefore:

> **Do not run S2, 250-pair confirmation, further model sweeps, probes, memory additions, or additional reward/environment redesign for Topic 06.**

## Failure type

**Layer D — prerequisite phenomenon / acquisition failure in the chosen AI system.**

This is not the same as showing that the higher-order generalization hypothesis is false. The experiment required the agent to first learn local controllability strongly enough for transfer breadth to be meaningful. That prerequisite remained weak across both the original pilot and the one permitted stronger-model v2.

## Main lessons

### 1. Validate the prerequisite phenomenon before studying its abstraction level

A question of the form

```text
when does learned X generalize?
```

should begin with a hard gate that the chosen learner robustly acquires `X` at all.

For this topic, the cleanest first question should have been simply:

> Does the agent reliably behave differently after controllable versus yoked-uncontrollable experience?

Only after a strong yes should diversity/generalization have entered the design.

### 2. A strong literature basis for a human phenomenon does not imply the AI system instantiates it

The psychological question was natural and well motivated. That did not guarantee that a vanilla interaction-history LLM agent would acquire the relevant latent belief through scalar outcomes alone.

The missing bridge was empirical:

```text
human learned helplessness exists
≠
ordinary LLM-agent interaction history produces an analogous controllability state
```

Future topic selection should demand evidence for that bridge before building higher-order AI claims on top of it.

### 3. Calibrate the behavioral readout before locking it

The v1 step-1 action was near ceiling because active intervention was cheap relative to its expected upside. A binary action can be a poor readout even when the underlying prior moves.

Before registration, analytically inspect whether plausible latent-belief changes can actually move the chosen observable away from floor/ceiling.

### 4. Do not confuse an unresolved natural question with a viable AI research topic

The question "when does helplessness become global?" remains scientifically meaningful. But the selected AI system did not provide a strong enough acquired phenomenon to study it cleanly.

A natural question can therefore be good while the candidate topic built around a particular model/system is bad.

### 5. One principled repair can be acceptable; a repair sequence is not

v2 was allowed because both changes had independent pre-result justification and were frozen in advance. Once v2 still failed the prerequisite gate, another rescue would become model/reward/environment search.

The reusable rule is:

> **If a candidate survives only by repeatedly changing the system until its prerequisite phenomenon appears, the candidate should be archived rather than optimized into existence.**

## Repository disposition

The directory is retained with:

- original `README.md` and `VALIDATION.md`;
- literature audit;
- complete v1 preflight/pilot results;
- `V2_PROTOCOL.md` and `V2_DECISION.md`;
- complete Qwen3-32B v2 S1 logs/results;
- environment, runner, analyzer and tests;
- this final archive summary.

The code may be reused for unrelated controllability experiments, but the Topic 06 claim is closed.

**Final decision: archive and move on.**