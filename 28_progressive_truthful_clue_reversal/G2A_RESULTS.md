# Topic 28 — G2a Destructive-Conjunction Results

## Frozen verdict

`STOP_DESTRUCTIVE_CONJUNCTION`

The outcome-blind G2a screen found only one destructive-conjunction event among
229 boundaries where P and C were each independently correct. All artifact and
measurement gates passed; every scientific density/support gate failed.

Per the frozen contract, this stops the destructive-composition/specificity-
capture rescue route. It does not authorize natural clue substitution,
aggregation-law fitting, another model/prompt/panel/scorer, or hidden-state
work.

## Environment and receipt

| Field | Frozen value |
|---|---|
| reused panel | complete outcome-blind G1 panel |
| panel support | 498 boundaries / 415 questions |
| question revision | `3dae05a66d3e0fd8c6b23ef8656ff6f4437bb1d4` |
| model | `Qwen/Qwen2.5-7B-Instruct` |
| model revision | `a09a35458c702b33eeacc393d103063234e8bc28` |
| inference | bfloat16, greedy, `max_new_tokens=24` |
| scorer | frozen exact normalized `clean_answers` match |
| uncertainty | 2,000 whole-question bootstrap resamples, seed `20260825` |
| host/GPU | `fvcrc20`, NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 3 |
| runtime | Python 3.12.0, PyTorch 2.8.0+cu128, Transformers 4.57.6 |

Commands:

```bash
cd 28_progressive_truthful_clue_reversal
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  /home/xiang/venvs/ragen/bin/python -m unittest discover -s tests -v

# Hash/panel preflight; no model output
/home/xiang/venvs/ragen/bin/python g2a_destructive_conjunction.py \
  --preflight-only --out-dir artifacts/g2a_preflight

# DEBUG schema smoke; forced DEBUG_NO_VERDICT
CUDA_VISIBLE_DEVICES=3 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  /home/xiang/venvs/ragen/bin/python g2a_destructive_conjunction.py \
  --debug-limit 8 --batch-size 16 --device cuda:0 \
  --out-dir artifacts/g2a_debug

# Complete frozen run
CUDA_VISIBLE_DEVICES=3 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  /home/xiang/venvs/ragen/bin/python g2a_destructive_conjunction.py \
  --batch-size 32 --device cuda:0 --out-dir artifacts/g2a
```

The final code passed 23/23 unit tests. G2a generated only the 498 missing
C-alone states; P and P+C were reused from hash-locked G1 artifacts.

## Reuse and output audit

The following preregistered G1 hashes matched exactly:

| Artifact | SHA-256 |
|---|---|
| `panel.csv` | `427deda33f45bb2a6c2d2caa1e64cead2b920b579412d0f7dbae1fe92f7b6f92` |
| `state_outputs.csv` | `ddf8eeb10e6dd4b69d1e1c09b2998e8bd6cb1a67cb1bf83f6ee2c348e4e4d11a` |
| `paired_results.csv` | `f3a05d65b07dc68977161fe77eb4593d9bdaf612bee4e575787a2e249750bc97` |

| Audit item | Result |
|---|---:|
| expected / observed C-alone outputs | 498 / 498 |
| duplicate C-alone keys | 0 |
| valid C-alone outputs | 497/498 (99.80%) |
| valid reused P outputs | 498/498 |
| valid reused P+C outputs | 498/498 |
| jointly sufficient P-and-C cases | 229 |

One C-alone continuation (`q4010_2_3`) reached the 24-token cap and contained
multiple lines. It was marked invalid exactly as frozen and did not enter the
destructive cell. The valid-output gate passed comfortably.

## Primary result

Marginal exact correctness was:

```text
P:     316/498 = 63.45%
C:     285/498 = 57.23%
P+C:   357/498 = 71.69%
```

There were 229 jointly sufficient boundaries where both P and C independently
elicited the exact gold alias. Of these, 228 remained correct when combined.

```text
P correct AND C correct AND P+C wrong: 1/229
R_destructive = 0.004367 (0.44%)
95% qid-cluster bootstrap CI = [0.0000, 0.01364]
```

The same single event passed the preregistered conservative `clear_wrong`
diagnostic:

```text
clear destructive events: 1/229 (0.44%)
unique clear questions: 1
95% qid-cluster bootstrap CI = [0.0000, 0.01402]
```

## All correctness cells

Bit order is `(P, C, P+C)`:

| Cell | Count |
|---|---:|
| `000` | 106 |
| `001` | 20 |
| `010` | 20 |
| `011` | 36 |
| `100` | 14 |
| `101` | 73 |
| `110` — destructive conjunction | 1 |
| `111` | 228 |

The composition pattern is therefore mostly constructive or stable, not
anti-synergistic. Twenty cases were wrong under both constituents but correct
when combined (`001`), while only one case showed the target reverse pattern.

## The single destructive event

```text
qid: q1522, boundary 2 -> 3
gold: 2

P clue:
This integer times pi gives the number of radians in the unit circle.
P prediction: 2 (correct)

C clue:
Truth tables can evaluate to this many outputs.
C prediction: 2 (correct)

P+C prediction: 8 (clear wrong)
```

This is a genuine representative of the proposed object, but one event cannot
support the intended population claim.

The event was in specificity Q4, at `t=2`, and in Science. Q3 had zero events
among 117 jointly sufficient cases; Q4 had one among 112. This support is far
too sparse for a specificity-by-composition claim.

## Frozen gates

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| G1 artifact hashes | exact | exact | PASS |
| panel boundaries | `==498` | 498 | PASS |
| unique questions | `==415` | 415 | PASS |
| valid C-alone fraction | `>=0.98` | 0.99799 | PASS |
| jointly sufficient support | `>=100` | 229 | PASS |
| exact destructive events | `>=10` | 1 | FAIL |
| exact destructive rate | `>=0.03` | 0.00437 | FAIL |
| lower exact-rate CI | `>0.01` | 0.0000 | FAIL |
| clear destructive events | `>=5` | 1 | FAIL |
| unique clear questions | `>=5` | 1 | FAIL |
| clear destructive rate | `>=0.01` | 0.00437 | FAIL |

## Failure classification and interpretation

This is a **scientific support failure**, not an artifact or measurement
failure. The panel, hashes, model receipt, output coverage, joint-correct
support, and conservative wrong-answer audit all passed. The critical cell is
simply too sparse.

G0 continues to establish a real released-trajectory reversal phenomenon, and
the G0 structure analysis continues to show a descriptive specificity
gradient. But two controlled explanatory upgrades now fail:

1. G1 found no adjacent-order effect;
2. G2a found destructive conjunction in only `1/229` jointly sufficient cases.

The planned ACL story that interprets the specificity gradient as destructive
evidence composition is therefore not supported. Natural clue substitution,
aggregation-law fitting, mitigation, and hidden-state work are not run.
