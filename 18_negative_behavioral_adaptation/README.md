# Topic 18 — Is Negative Behavioral Adaptation Intrinsically Harder?

**Status:** `ARCHIVED / VALID G0, INCONCLUSIVE MODEL HETEROGENEITY / STOP_NO_RESCUE_SWEEP`

The frozen local three-family run is complete. Phi and Gemma showed large gaps,
while Qwen did not; the preregistered cross-model survival condition therefore
failed, but the pooled effect was too large for the null kill region. The only
permitted decision is `INCONCLUSIVE`, followed by stop. See
[ARCHIVE_SUMMARY.md](./ARCHIVE_SUMMARY.md) and [G0_RESULTS.md](./G0_RESULTS.md).

## Natural question

> **When behavior, exposure structure and outcome magnitude are matched, are language models worse at learning to suppress an action after negative experience than at learning to select an action after positive experience?**

Short form:

> **Can a model learn “do this again” much more easily than “do not do this again”?**

## Why this gets only a cheap G0 registration

The seed phenomenon is strong but the causal interpretation is not yet established.

Qin et al., ACL 2026, *ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models*, reports a large aggregate gap between inhibitory and preference adaptation across modern models.

Seed:

- https://aclanthology.org/2026.acl-long.1301/

However, the benchmark's inhibition and preference comparisons use different task families. Therefore the observed gap can reflect task object, prior behavior, output structure, negation, or other differences rather than inhibition itself.

This is exactly the failure mode our archive warns about:

```text
strong motivating phenomenon
!=
our proposed explanatory axis matters
```

So Topic 18 receives **one matched falsification-first experiment**, not an open-ended project.

## Excitement test

A strong positive would be genuinely surprising:

> **Even for arbitrary actions with symmetric feedback and matched presentation, models reliably convert positive experience into future selection but fail to convert equally strong negative experience into future suppression.**

That would expose a basic asymmetry in experience-based adaptation.

If the original gap collapses under matched pairs, the topic is killed. We do not search for another task family where it comes back.

## Method opening

Only if the matched gap survives:

- inhibition-aware implicit memory;
- training objectives that specifically turn negative experience into automatic behavioral suppression;
- negative-trajectory post-training that is evaluated on *future first action*, not explicit verbal recollection of the failure;
- mechanisms that distinguish remembering a failure from actually changing behavior.

This gives a real lever: a system can explicitly know “that failed” yet still repeat the behavior.

## Why this is safer than our failed topics

- **One factor only:** the same generator makes positive and negative counterfactuals.
- **No mechanism claim in G0:** first establish the behavioral asymmetry.
- **No model/task fishing:** freeze a small model panel and generator; a weak result is a stop.
- **No hidden-state escalation after a null:** representation analysis is forbidden unless the paired behavioral gap is strong.

## G0 design

Use arbitrary action symbols with near-neutral prior preference.

For every base pair, construct matched episodes with the same:

```text
action labels
action order
number of observations
interference text
test wording
absolute feedback magnitude
```

Positive and negative conditions change only feedback sign; an equal-outcome
baseline uses the same actions, event count, interference, and test. Marked
identity, observation order, and answer-option order are fully crossed in each
64-cell block.

### Positive condition

```text
marked action -> +1
other action  ->  0
```

Correct later behavior: select the marked action.

### Negative condition

```text
marked action -> -1
other action  ->  0
```

Correct later behavior: avoid the marked action and select the neutral alternative.

Counterbalance which symbol is marked and presentation order.

### Primary statistic

For matched pair `i`:

```text
positive_correct_i - negative_correct_i
```

Aggregate:

```text
Δ_inhibition = accuracy_positive - accuracy_negative
```

### Important prerequisite

Before interpreting the gap, run the frozen equal-outcome baseline. It is
generated and scored automatically. Because marked identity is fully crossed,
a preference for one literal symbol is orthogonal to feedback sign and is
reported as a warning; an aggregate baseline preference for the marked action
fails the run as `INVALID`. Labels may only be replaced before inspecting
positive/negative outcomes.

### Survival bar

Do not freeze an exact publication threshold yet, but the first pilot must show all of the following to justify a proper preregistered G0:

- a large paired positive-minus-negative gap, not a few percentage points;
- same sign across the small frozen model panel;
- robustness to identity swap / order counterbalancing;
- no comparable gap in no-feedback label preference.

### Kill line

Kill the inhibition framing if:

- the large seed-paper gap largely disappears in matched pairs;
- the sign changes with arbitrary labels or order;
- baseline label preference explains the difference;
- the effect requires explicit natural-language negation;
- only one model gives a large result.

No prompt/task/model rescue sweep after this.

## Initial code

Generate one complete frozen factorial block:

```bash
python 18_negative_behavioral_adaptation/generate_g0.py \
  --output /tmp/g0_design.jsonl \
  --n-base 64
```

Run at least three predeclared local model families (or use an equivalent frozen
inference harness):

```bash
.venv_clean2/bin/python 18_negative_behavioral_adaptation/run_local_models.py \
  --design /tmp/g0_design.jsonl --output predictions.jsonl \
  --model family_a:model_a=/local/path/a \
  --model family_b:model_b=/local/path/b \
  --model family_c:model_c=/local/path/c
```

Predictions are JSONL:

```json
{"model_family": "family_a", "model_id": "model_a", "item_id": "...", "output": "KEL"}
```

Then score:

```bash
python 18_negative_behavioral_adaptation/score_g0.py \
  --design /tmp/g0_design.jsonl \
  --predictions predictions.jsonl
```

The scorer validates the factorial design and panel coverage, reports paired
bootstrap intervals and an exact McNemar/sign test, checks every counterbalance
stratum and baseline bias, then emits exactly one frozen verdict. Full thresholds
and their interpretation are in [VALIDATION.md](./VALIDATION.md).
