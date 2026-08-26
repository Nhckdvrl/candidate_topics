# Topic 29 temporal-prefix G0 result

Date: 2026-08-27

## Verdict

**PASS: the phenomenon is large, directly observable, and cheaply reducible. Keep the original topic.**

This result tests the registered causal contrast rather than treating an AMI decision abstract as source-state ground truth. For each summlink decision chain, `run_g0_temporal_prefix.py` withholds at least the final linked turn and retains only a latest prefix with an explicit `PROPOSED`, `TENTATIVE`, or `CONDITIONAL` cue. Prefixes containing broad finality, rejection, or explicit-open cues are excluded.

## Fixed run

- Data: AMI manual annotations v1.6.2, public summlink conversion.
- Model: `Qwen/Qwen2.5-7B-Instruct`, snapshot `a09a35458c702b33eeacc393d103063234e8bc28`.
- Decoding: greedy, one minutes-style sentence, maximum 64 new tokens.
- Candidate order: deterministic SHA-256 order.
- Candidates: 54; source-scorable: 54; content-grounded: 52.

| Condition | Unsupported unconditional decisions |
|---|---:|
| neutral minutes prompt | 39/52 (75.0%) |
| state-preservation prompt, same inputs/model | 0/52 (0.0%) |
| full linked chain summarized as decided | 45/54 (83.3%) |

Neutral-prompt upgrade rates by licensed prefix state were 25/30 (83.3%) for `PROPOSED`, 7/11 (63.6%) for `TENTATIVE`, and 7/11 (63.6%) for `CONDITIONAL`.

Representative contrast:

> Source: “I think we could go for ... maybe ... a very sort of curvy type shape.”  
> Neutral summary: “The group decided to pursue a curvy shape ...”  
> Preservation summary: “A curvy shape was proposed ... but no decision was made.”

## Review repairs made before freezing the result

- Removed total-order treatment of `REJECTED` and separated uncued `UNKNOWN` from explicit `OPEN`.
- Separated transcript cues from minutes-genre commitment cues.
- Blocked prefixes containing broad finality/rejection language, not just the narrow state regex.
- Stopped counting “decided to consider/discuss X” as adoption of X.
- Made explicit “did not make/reach a final decision” override positive decision words.
- Added summary subjects such as `meeting`, `participants`, and passive `decision was made` to reduce false negatives.
- Added the preservation-prompt matched control; it shows a direct remedy runway rather than only a failure count.

## What this establishes—and what it does not

It establishes a strong feasibility signal for **decision-state transmutation under meeting compression** and shows that an explicit preservation intervention can almost eliminate it in this controlled setting. It is already enough to reject the “dead topic” hypothesis.

It is not a publication-ready prevalence estimate. AMI prefixes are lexically licensed rather than human state gold, linked support may omit pragmatic context, and the current run uses one model. A paper should human-audit state and proposition identity and then test multiple summarizers/corpora. Those steps strengthen external validity; they do not change the G0 decision to continue.
