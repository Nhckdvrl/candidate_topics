# Archive Summary — Topic 18: Is Negative Behavioral Adaptation Intrinsically Harder?

**Final status: ARCHIVED / VALID G0, INCONCLUSIVE MODEL HETEROGENEITY / NO RESCUE SWEEP**

Archived 2026-08-23. The matched behavioral experiment was technically valid,
but it did not establish the large, model-general inhibition deficit required to
justify an inhibition-specific research program. Phi and Gemma showed substantial
positive-minus-negative gaps; Qwen did not. The topic therefore stops at G0.

## 1. Original claim and required bar

The motivating benchmark reported a very large aggregate difference between
inhibition and preference adaptation, but those conditions came from different
task families. Topic 18 asked the identifying question:

> When action space, experience count, feedback magnitude, wording and output
> structure are matched, do language models still learn positive selection much
> more readily than negative suppression?

This was registered as a cheap falsification-only candidate. Promotion required
a large paired gap with the same direction across a frozen multi-family panel,
robustness to action identity and order, and a clean equal-outcome baseline.
Prompt/task/model rescue after a weak or heterogeneous result was forbidden.

## 2. Measurement repair before the run

The initial prototype said “choose the command with the better observed outcome.”
That would measure explicit comparison and instruction following, not behavioral
adaptation. The final design instead gave a standing score-maximization goal,
showed two prior uses, inserted interference, and requested the first action on a
new use without restating the comparison rule.

One complete block contained 64 base cells:

```text
8 symbol pairs
× 2 marked identities
× 2 observation orders
× 2 answer-option orders
```

Every cell had positive (`+1/0`), negative (`-1/0`) and equal-outcome baseline
(`0/0`) variants. Thus each model answered 192 prompts. The scorer verified that
condition prompts were byte-equivalent after masking the numeric outcomes.

## 3. Technical preflight and one permitted repair

The first Qwen3 attempt spent the short output budget inside a `<think>` preamble
and was correctly labeled `INVALID`; it was never scored as incorrect behavior.
The local runner was repaired to disable optional thinking mode and to flush each
batch for resumability. This was a format-level repair with an independently
observable failure, not an outcome-driven scientific change.

The final panel used three explicitly recorded, distinct model families and exact
cached revisions:

- Qwen2.5-7B-Instruct;
- Microsoft Phi-4-mini-instruct;
- Google Gemma-3-12B-it.

All 576 final outputs were parseable. Every model selected the marked action at
exactly 0.5 on the counterbalanced equal-outcome baseline.

## 4. Frozen G0 result

| Family / model | Positive accuracy | Negative accuracy | Paired delta | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Google Gemma / Gemma3-12B | 1.000 | 0.797 | 0.203 | [0.109, 0.297] |
| Microsoft Phi / Phi-4-mini | 0.969 | 0.578 | 0.391 | [0.266, 0.531] |
| Qwen / Qwen2.5-7B | 0.953 | 0.906 | 0.047 | [-0.047, 0.141] |
| **Stimulus-clustered panel** | — | — | **0.214** | **[0.135, 0.292]** |

The pooled effect looked large, but pooling was not allowed to override the
predeclared model-consistency requirement. Qwen failed the per-model `0.10` bar,
and counterbalance-stratum consistency also failed. Therefore `SURVIVE` was not
permitted.

The clean-null `KILL` region was also not reached: the pooled upper interval was
well above `0.10`, and two independent families showed large gaps. The exact
frozen verdict was `INCONCLUSIVE`.

## 5. Why an inconclusive G0 is still an archive decision

The project was not registered to catalog which individual models show a sign
effect. Its value depended on a simple, general limitation that could motivate
inhibition-specific memory or post-training methods. The actual finding is model
family heterogeneity.

Explaining that heterogeneity would require the prohibited rescue/search path:

```text
more model families
→ prompt/template variants
→ label-set variants
→ delay/interference sweeps
→ hidden-state or training analysis
```

That is a different, broader project without the one-factor identifying advantage
that justified this cheap G0. Topic 18 therefore does not advance.

## 6. Failure type and reusable lesson

**Layer D — valid substantive G0, frozen gray zone / model-general claim not
established.** The phenomenon exists strongly in some models, but it is not the
stable cross-model regularity required by the proposed framing.

The reusable lesson is:

> A large pooled effect cannot rescue a failed generality gate. If the method
> opening requires a model-general deficit, model-family heterogeneity is a stop
> result even when selected models show striking effects.

Preserved artifacts:

- `g0_design_v2.jsonl` — exact 192-item design;
- `g0_predictions_final.jsonl` — 576 outputs with family/model/revision;
- `G0_LOCAL_RESULTS.json` — complete scores, controls, strata and decision;
- `G0_RESULTS.md` — compact result report;
- `VALIDATION.md` — frozen measurement and gate contract.

Do not run model, prompt, label, delay, interference, or training rescue sweeps
under Topic 18.
