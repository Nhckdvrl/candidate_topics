# Topic 25 Validation Audit

## Verdict before running

**RUN RECEIPT. If and only if the receipt passes, run the frozen matched G0. Do not run mechanism work.**

---

## 1. Why this is not just Round-09 prose promoted without review

The later active-search table downgraded the Round-09 object for advisor fit and neighboring work. That is a ranking judgment, not a local scientific negative.

Registration is justified only because the current implementation turns the broad story into two highly falsifiable stages:

1. reproduce the exact open Weakest-Link Qwen3 seed relation;
2. localize or kill the first proposed boundary on matched MuSiQue evidence.

A failure at either stage creates an immediate stop. Numbering the topic does not weaken the kill rule.

---

## 2. Nearest internal failure structures

### Topic 20: decodable does not imply causal use

Topic 25 does not begin from a representation or probe. G0 is purely behavioral and paired. No hidden feature is used as evidence for mechanism.

### Topic 21: official artifact existence does not imply seed reproduction

This is the most important operational lesson. Topic 25 therefore makes an upstream receipt mandatory and blocks novel G0 downstream of it.

### Topic 22: measurement support can invalidate a seemingly strong phenomenon

Topic 25 uses the upstream programmatic Exact Match scorer and released answers. There is no open-text-to-closed-label semantic canonicalizer and no LLM judge on the primary endpoint.

### Round-09 / GuarantRAG collision

The occupied statement is roughly `thinking helps harder / more complex questions more`.

Topic 25 survives only if it establishes a stronger matched result:

```text
same item + same 18-doc evidence + same placement
atomic evidence query vs composed query
× think vs no-think
```

If that interaction is absent, do not retreat to cross-dataset complexity bins.

---

## 3. Receipt audit

### Upstream pin

```text
cambridgeltl/weakest-link-effect
9b01abaad354208a6a8fb26c58eb5c330036fb94
```

`run_seed_receipt.sh` refuses any other HEAD.

### Shell drift found upstream

The frozen upstream shell wrapper `musique_gold-ablation.sh` is not used because its current orchestration has drifted from the paper-era 8B thinking cell.

The direct Python entrypoint is still clean:

```text
python -m src.infer.entity.run_ablation --mode gold_only ...
```

and internally maps `Qwen3-8B + --enable-thinking` to the `Qwen3-8B-Think` result directory while using the same Qwen3-8B checkpoint.

This is an engineering repair with zero change to the scientific manipulation.

### Receipt support

The checker requires every expected file to contain the exact same complete bank ID set. Missing calls, duplicates, wrong thinking flags, wrong prompt IDs, or malformed `correct` fields fail support.

### Receipt threshold policy

No numerical closeness threshold is invented. Published gold-only values are recorded as diagnostics. The gate uses complete support and the qualitative seed relations claimed by the paper.

---

## 4. Matched G0 identification audit

### What is changed

Only the query computation:

- two released single-hop questions, with prior placeholders resolved from released gold intermediate answers;
- original composed 2-hop question.

And independently:

- thinking off vs on.

### What is fixed

- exact item;
- exact 18 documents;
- distractor order;
- gold documents;
- bucket placement;
- within-bucket distance;
- Qwen3-8B checkpoint;
- prompt template ID;
- temperature/top-p/seed;
- scorer.

### Why placeholder substitution is necessary

If `#1` were left in the question, the atomic task would be malformed.

If the model were required to infer `#1`, the second atomic condition would still require the first hop and would no longer isolate local evidence use.

Substituting the **released intermediate answer** therefore defines the intended single-step task. It does not use the final composed answer.

### Support mapping

For each atomic step, the released `paragraph_support_idx` paragraph must exact-normalize to exactly one of the two Weakest-Link bank gold documents. The two steps must cover both bank gold documents. No fuzzy matching is allowed.

### Sampling

Eligible cases are ranked by SHA-256 of `20260825:item_id`; the first 256 ranks are frozen. This removes source-file-order dependence without looking at any model outcome.

---

## 5. Endpoint audit

The atomic endpoint is not the mean of two unrelated row accuracies. At item level:

```text
atomic_both_correct = atomic_0_correct AND atomic_1_correct
```

This provides one binary atomic endpoint per item, comparable to the one binary composed endpoint.

The primary interaction is therefore paired at the item level.

Bootstrap resampling also occurs at the **item ID** level, keeping all bucket placements and query conditions for an item together.

---

## 6. G0 gates and their meaning

The competence floors prevent a dramatic interaction from being produced by an essentially broken baseline task.

The effect gates demand a paper-sized selective benefit:

```text
composed gain >= 8 pp
atomic gain   <= 3 pp
interaction  >= 8 pp
90% paired bootstrap lower bound > 0
positive interaction in >=2/3 buckets
```

These are deliberately strict. A weak positive trend is not enough to justify mechanism work.

A literal atomic-negative/composed-positive sign reversal is reported as a diagnostic, not used as a second chance if the frozen gates fail.

---

## 7. Main remaining risks

### Risk A — atomic questions are easier in more ways than compositionality alone

Yes. They also have shorter dependency depth and one resolved entity. That is the intended first computation contrast, not a claim that hop count is the only latent variable.

The project only earns broader interpretation after the matched interaction exists and subsequent predeclared characterization separates depth/topology/noise.

### Risk B — thinking protocol uses greedy decoding although Qwen recommends sampling in some settings

The receipt and G0 stay faithful to the frozen Weakest-Link scientific contract: temperature 0.0, top-p 1.0. Do not change sampling after seeing results.

A later sampling robustness study would require a new preregistered experiment and cannot rescue G0.

### Risk C — source dataset revision / bank alignment

The G0 pins the source MoreDocsSameLen revision and independently verifies exact support-paragraph equality against the bank. If fewer than 256 clean cases remain, that is an identification/data failure to resolve before any model calls—not a reason to relax matching.

### Risk D — large output-token cost

The frozen panel is 4,608 calls (`256 × 3 buckets × 3 query types × 2 modes`). This is inference-only on an 8B model and fits a single local multi-GPU node through vLLM. Resume logic operates at exact task-key granularity.

---

## 8. Authorized interpretation matrix

### Receipt fails

`STOP TOPIC 25`.

No novel G0.

### Receipt passes, G0 fails

`STOP_MATCHED_BOUNDARY`.

The first proposed atomic→composed boundary is unsupported. Do not mine subsets or switch Qwen models.

The broad external contradiction can return to the search log only if a new externally motivated axis appears; it does not stay active automatically.

### Receipt passes, G0 passes

`GO_MATCHED_BOUNDARY`.

Authorized next work is characterization of the behavioral boundary. Hidden-state/attention causal claims still require a separate frozen experiment.

---

## 9. No-go list after a failed G0

Do not rescue by:

- selecting only questions whose atomic steps are already correct;
- dropping one bucket after seeing its interaction;
- choosing a different within-bucket distance;
- changing prompt ID;
- changing think sampling settings;
- trying Qwen3-4B/14B to find a favorable sign;
- filtering decomposition wording/types;
- using F1 instead of the frozen Exact Match endpoint;
- lowering the 8 pp effect gates.

A new experiment requires a new scientific premise, not a better-looking result.