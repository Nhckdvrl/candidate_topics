# Topic 29 temporal-prefix G0 result

Date: 2026-08-27

## Verdict

**PASS: the phenomenon is large across three model families, directly observable, and cheaply reducible. Keep the original topic.**

This result tests the registered causal contrast rather than treating an AMI decision abstract as source-state ground truth. For each summlink decision chain, `run_g0_temporal_prefix.py` withholds at least the final linked turn and retains only a latest prefix with an explicit `PROPOSED`, `TENTATIVE`, or `CONDITIONAL` cue. Prefixes containing broad finality, rejection, or explicit-open cues are excluded.

## Fixed runs

- Data: AMI manual annotations v1.6.2, public summlink conversion.
- Models:
  - `Qwen/Qwen2.5-7B-Instruct`, snapshot `a09a35458c702b33eeacc393d103063234e8bc28`;
  - `Qwen/Qwen3-8B`, snapshot `b968826d9c46dd6066d109eabc6255188de91218`, thinking disabled;
  - `google/gemma-3-12b-it`, snapshot `96b6f1eccf38110c56df3a15bffe176da04bfd80`.
- Decoding: greedy, one minutes-style sentence, maximum 64 new tokens.
- Candidate order: deterministic SHA-256 order.
- Candidates: 54; source-scorable: 54; content-grounded: 52.

| Model | Neutral, model-grounded denominator | Preservation, same denominator | Neutral, common grounded 49 | Full-chain decided |
|---|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | 39/52 (75.0%) | 0/52 (0.0%) | 39/49 (79.6%) | 45/54 (83.3%) |
| Qwen3-8B | 34/53 (64.2%) | 1/53 (1.9%) | 32/49 (65.3%) | 36/54 (66.7%) |
| Gemma-3-12B-IT | 22/50 (44.0%) | 0/50 (0.0%) | 21/49 (42.9%) | 27/54 (50.0%) |

All three models therefore show the same directional failure on the same source pool. The common-grounded intersection prevents different content-recall denominators from explaining the cross-model result. The single Qwen3 preservation failure was manually checked and is genuine: for a source saying “I think ... We could just start with the assumption ...”, it generated “The group agreed to start ...”.

For the initial Qwen2.5 run, neutral-prompt upgrade rates by licensed prefix state were 25/30 (83.3%) for `PROPOSED`, 7/11 (63.6%) for `TENTATIVE`, and 7/11 (63.6%) for `CONDITIONAL`.

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
- Disabled Qwen3 thinking through its chat template and stripped any defensive residual thinking block before scoring.
- Batched all three matched conditions through one model load so model weights and decoding configuration are identical across conditions.

## What this establishes—and what it does not

It establishes a strong feasibility signal for **decision-state transmutation under meeting compression** and shows that an explicit preservation intervention can almost eliminate it in this controlled setting. It is already enough to reject the “dead topic” hypothesis.

It is not a publication-ready prevalence estimate. AMI prefixes are lexically licensed rather than human state gold, and linked support may omit pragmatic context. A paper should human-audit state and proposition identity, broaden model scale/API coverage, and add a transfer corpus. Those steps strengthen external validity; they do not change the G0 decision to continue.
