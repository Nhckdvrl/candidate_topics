# Topic 11 archive summary

## Final status

**ARCHIVED — frozen G-0 falsified the meaningful retroactive/global-consistency interpretation.**

The validation harness worked: both preregistered protocol prerequisites passed strongly. But the locked primary effect on unchanged middle reasoning-result tokens was essentially zero, with a 95% confidence interval that excluded the predeclared `0.010` minimum-worthwhile effect by roughly two orders of magnitude in absolute terms and about 400× at the upper bound.

No G-1 is justified.

## Scientific question

Does native diffusion-LM confidence encode **global internal consistency of a reasoning trajectory**, independently of whether the trajectory is externally correct?

The motivating literature showed that DLM confidence can discriminate correctness and responds strongly to arithmetic contradiction. Topic 11 asked a stricter question: is that consistency sensitivity meaningfully distributed across the reasoning trajectory, or is aggregate confidence mainly reacting to local/sequence-level compatibility?

## Identification design

After a pre-run audit, the experiment used a retroactive `internal consistency × external correctness` factorial.

A fixed trajectory appears first. External correctness is manipulated in the prompt. Internal consistency is manipulated only in a semantic consistency-check suffix that comes **after** the trajectory. Prompt and suffix encode anchors through different arithmetic aliases so the factor is not literal digit copying.

The primary metric, `confidence_result_middle`, scores the unchanged Step-2/Step-3 result tokens that occur **before** the consistency suffix.

Thus a positive primary effect would have required a future contradiction to retroactively change confidence assigned to earlier, unchanged reasoning tokens.

## Protocol validity

Both prerequisites passed decisively:

```text
Arithmetic result substitution:
mean gap = 0.426361
95% CI   = [0.390624, 0.462064]
locked floor = 0.100

Semantic-alias comprehension:
mean gap = 0.214693
95% CI   = [0.186151, 0.244417]
locked floor = 0.020
```

Eligible mirrored anchor pairs: `140`.

The digit-level tokenizer required a narrow engineering compatibility repair to preserve one-token eligibility. It did not alter the scientific factors or locked verdict contract.

## Decisive result

```text
Primary: confidence_result_middle
Delta_consistency = -0.000003
95% CI            = [-0.000055, 0.000025]
locked meaningful floor = 0.010
```

The upper confidence bound is approximately `0.000025`, around **400× below** the preregistered `0.010` effect floor.

This is not an underpowered gray zone. The frozen run excludes the project-level effect size that would have made the hypothesis interesting.

The strongest cross-cell headline also failed:

```text
coherent-wrong - incoherent-correct = -0.000587
95% CI                            = [-0.001737, -0.000009]
```

So coherent-but-wrong did not receive greater retroactive confidence than incoherent-but-correct.

## The tempting post-hoc signal

Full-sequence confidence showed a strong positive consistency effect:

```text
confidence_full Delta_consistency = 0.013765
95% CI                           = [0.013069, 0.014462]
```

But `confidence_full` includes the manipulated consistency-check suffix itself.

The experiment was explicitly designed to distinguish:

```text
aggregate/full-sequence sensitivity
from
meaningful retroactive signal on earlier unchanged reasoning tokens
```

The former was strong; the latter was absent.

Switching the headline from the locked primary to `confidence_full` after seeing the result would therefore reverse the identification logic and constitute metric shopping, not a legitimate rescue.

## Why the project stops here

The topic's scientific contribution depended on a meaningful global/retroactive interpretation of DLM confidence. G-0 directly targeted that interpretation and produced a valid negative while the positive controls remained very strong.

The appropriate action is therefore:

- no larger frozen run;
- no second DLM to search for a positive version;
- no token-region or pooling sweep;
- no distance-to-suffix sweep to rescue the same claim;
- no G-1.

A new external observation could motivate a separately registered question, but it would not retroactively rescue Topic 11.

## Transferable lessons

1. **Aggregate signal != distributed mechanism.** A sequence-level score can strongly track a property even when the same property is essentially absent from the specific internal/positional readout required by the mechanistic story.
2. **Positive controls make a negative interpretable.** The very large arithmetic and semantic-alias gaps rule out the easiest engineering explanations for the primary null.
3. **Predeclare a minimum-worthy effect, not only significance.** The experiment could distinguish a scientifically meaningful `1pp` effect from microscopic tendencies; this prevented a tiny token-level tendency from being promoted into a project.
4. **Do not rescue a failed primary with an attractive secondary metric.** Here the secondary `confidence_full` result is especially tempting because it is large and significant, but it measures a different object that includes the intervention itself.
5. **A good G-0 can kill the interpretation rather than the observable.** DLM confidence clearly responds to consistency somewhere in the complete sequence; what failed was the stronger claim that this sensitivity behaves like a meaningful global reasoning-state signal.

## Reopen condition

Only reopen if an independent result creates a genuinely new natural question with a distinct identification strategy. Do not reopen merely to search for a token region, pooling rule, model, prompt, or dataset where the same claim becomes positive.

See [`G0_RESULT.md`](./G0_RESULT.md) for the full frozen table.
