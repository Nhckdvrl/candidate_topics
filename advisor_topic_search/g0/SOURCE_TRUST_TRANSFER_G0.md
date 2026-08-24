# Source-Trust Transfer G0 — frozen before the new hypothesis run

Date frozen: 2026-08-24

Status: **REPRODUCTION RECEIPT PASSED / NEW-HYPOTHESIS G0 READY / DO NOT REGISTER TOPIC 23 YET**

## 0. Why this G0 is now allowed

The seed phenomenon has been reproduced on the exact public object before introducing our new question:

- seed: ACL 2026 main/long, *Whose Facts Win? LLM Source Preferences under Knowledge Conflicts*;
- official artifact: `JaSchuste/llm-source-preference`;
- locked upstream commit: `87dd466f10a76ea1cadc21a552d423d2d60c0cce`;
- model: `google/gemma-3-4b-it`;
- official data, prompt machinery and Source Preference scorer;
- seed 42 reproduction completed locally;
- reported local receipt anchors from the completed run:
  - no-source repetition shift `44.73` vs paper about `45.9`;
  - social-media repetition shift `30.55` vs paper about `33.2`;
  - 1-table majority near zero;
  - 2-table majority and exact repetition large.

The detailed local receipt is `SOURCE_TRUST_REPRODUCTION_RECEIPT.md`; it should be committed separately without overwriting unrelated local changes.

This clears only the **seed reproduction**. The new source-level-transfer claim remains untested.

---

# 1. Scientific question

The seed shows that repeating low-credibility information can strongly change which conflicting claim an LLM prefers.

That leaves a larger classical distinction:

> **Does repetition make the model trust only the repeated claim, or does repetition update trust in the source itself and therefore affect a novel claim from the same source?**

Competing hypotheses:

- **H1 — claim-local repetition:** duplicate exposure increases the influence/familiarity of that claim only. Nothing transfers to an unrelated new claim from the source.
- **H2 — source-level trust transfer:** association with a repeated claim changes the effective weight assigned to the source, so the source gains influence on a novel claim.
- **H3 — mixed:** source-level transfer exists but is much smaller than the immediate same-claim repetition effect.

The G0 is designed to distinguish H1 from a paper-scale H2/H3 without hidden states, generation, parser, judge, or prompt search.

---

# 2. The matched causal contrast

For every independent target item, construct two anonymous social-media sources `S1` and `S2` of the same official source type. Both have hidden follower count, matching the seed's generic social-media source construction.

Choose two historical claims `H1`, `H2` and one unrelated target claim `Y`. Historical and target entities are distinct but drawn from the same NeoQA entity class, so the transfer is cross-claim/cross-entity without introducing a domain change.

## Condition R1 — S1 is associated with repetition

```text
S1: H1
S2: H1
S1: H1
S2: H2

CURRENT TARGET:
S1: Y = value 1
S2: Y = value 2
```

## Condition R2 — S2 is associated with repetition

```text
S1: H1
S2: H1
S1: H2
S2: H1

CURRENT TARGET:
S1: Y = value 1
S2: Y = value 2
```

The target is byte-identical across R1/R2.

Critically, the historical marginals are also exactly matched:

```text
S1 exposure count = 2 in both
S2 exposure count = 2 in both
H1 global count    = 3 in both
H2 global count    = 1 in both
total prompt length = identical
```

The only manipulated relation is:

```text
which source is the one associated with the duplicate historical claim
```

This blocks the easy alternatives:

- source-name familiarity;
- more mentions of one source;
- more global repetition in one condition;
- more total evidence;
- a different target question/value/entity;
- direct repetition of the target claim.

A positive R1-vs-R2 target preference shift is therefore evidence for **source-associated cross-claim transfer**, not merely immediate claim repetition.

---

# 3. Counterbalancing

Each of the 128 independent target entities gets the full frozen factorial:

```text
repeated source:       S1 / S2
history order:         forward / reversed
target table order:    canonical / swapped
answer option order:   canonical / swapped
```

That is:

- 8 prompt variants per repeated-source condition;
- 16 prompts per independent target item;
- 128 independent target items;
- 2048 one-token probability evaluations total.

The model does not produce free text. As in the official seed artifact, the measurement is the next-token probability over `A/B` under greedy one-token inference.

---

# 4. Frozen object

```text
upstream commit = 87dd466f10a76ea1cadc21a552d423d2d60c0cce
model           = google/gemma-3-4b-it
new-G0 seed     = 20260824
N independent target items = 128
history claims per prompt   = 4
target source type          = social media vs social media
follower count              = hidden for both
```

The script refuses a different upstream commit, model, seed, or N.

No model shopping, source-type shopping, prompt rephrasing, or threshold change after results.

---

# 5. Primary statistic

For each item, average over the eight positional/order variants.

Let

```text
p1(R1) = P(model chooses the target value supported by S1 | S1 was the repeated-history source)
p1(R2) = P(model chooses the target value supported by S1 | S2 was the repeated-history source)
```

Define

```text
Delta_transfer = p1(R1) - p1(R2)
```

A positive value means that changing only which source is associated with repetition shifts the novel target answer toward that source.

Inference unit: **target item**, not prompt variation.

95% CI: 10,000-item bootstrap, frozen seed derived from `20260824`.

---

# 6. Frozen promotion gate

The reproduction receipt found an immediate same-claim social-media repetition shift of `30.55` percentage points.

We predeclare **5 percentage points** as the minimum paper-scale cross-claim transfer: roughly one-sixth of the already reproduced immediate effect. A smaller statistically detectable residual is not enough to register a major source-trust paper.

`GO_REGISTER_TOPIC23_SOURCE_LEVEL_TRANSFER` requires all:

1. exactly `128` independent target items;
2. mean `Delta_transfer >= 0.05`;
3. item-bootstrap 95% CI lower bound `> 0`;
4. at least `60%` of items have positive transfer delta;
5. every predeclared counterbalance level (history order, target table order, answer order) has positive mean delta;
6. at least `12` independent target items show the strong discrete crossover:

```text
S1 repeated -> averaged choice prefers S1's target value
S2 repeated -> averaged choice prefers S2's target value
```

The final requirement creates a nontrivial discrete critical cell for later causal mechanism work rather than relying only on a tiny distributed logit change.

Other frozen outcomes:

- bootstrap CI upper `< 0.03` -> `KILL_SOURCE_LEVEL_TRANSFER_PAPER_SCALE`;
- CI lower `> 0` but mean `< 0.05` -> `WEAK_POSITIVE_BELOW_PAPER_SCALE_DO_NOT_REGISTER`;
- otherwise -> `INCONCLUSIVE_DO_NOT_TUNE`.

Do not rescue an inconclusive/negative result by changing model, source type, historical distance, prompt wording, N, or thresholds.

---

# 7. Run

From `candidate_topics` after syncing main, while preserving any local uncommitted receipt work:

```bash
python advisor_topic_search/g0/source_trust_transfer_g0.py \
  --upstream-repo /ABS/PATH/TO/llm-source-preference \
  --data-dir /ABS/PATH/TO/llm-source-preference/data
```

The upstream checkout must be exactly at:

```text
87dd466f10a76ea1cadc21a552d423d2d60c0cce
```

Outputs:

```text
artifacts/source_trust_transfer_g0/summary.json
artifacts/source_trust_transfer_g0/records.jsonl
artifacts/source_trust_transfer_g0/prompt_audit.jsonl
```

`prompt_audit.jsonl` intentionally contains only a small sample of full prompts. `records.jsonl` stores item-level statistics without duplicating all 2048 prompts.

---

# 8. Audit before accepting the result

Before trusting the verdict, inspect at least 8 paired prompt examples and verify:

- S1 and S2 each occur exactly twice in history;
- H1 occurs exactly three times and H2 once in both conditions;
- the target section is identical between R1 and R2;
- neither historical claim is the target entity/field;
- target sources are the same two identities across the pair;
- answer/table order remapping correctly returns probability of S1's semantic target value.

The script performs the strongest structural checks automatically before model inference, including byte-identical target sections and identical paired prompt lengths.

---

# 9. What a positive G0 would and would not prove

A positive G0 supports:

> **Associating a source with duplicate historical information changes that source's influence on an unrelated novel claim, beyond mere claim frequency or source exposure.**

It does **not** yet prove that the model has an explicit scalar `source credibility` representation.

Only after G0 passes may mechanism work ask whether:

1. repetition changes a source-identity representation before the target claim appears;
2. that state transfers to the target source mention;
3. causal patching/ablation of the source-history state specifically changes the novel target choice;
4. the effect generalizes beyond one generated username/source pair without broad search.

A generic linear probe saying `repeated source is decodable` is not enough.

---

# 10. Registration rule

Do **not** create `23_*` before this G0.

If and only if verdict is:

```text
GO_REGISTER_TOPIC23_SOURCE_LEVEL_TRANSFER
```

then register Topic 23 with the behavioral phenomenon already established and make the next experiment a bounded causal mechanism test.
