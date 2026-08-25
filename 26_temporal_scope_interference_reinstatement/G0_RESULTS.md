# Topic 26 — Frozen G0 Preflight Result

Date run: 2026-08-25
Frozen verdict: **`STOP_INSUFFICIENT_EXACT_SUPPORT`**

## Outcome

The exact-eligibility gate failed before tokenization or model inference:

```text
target N = 512
eligible = 0
selected = 0
```

The official downloaded benchmark contains no turn-level
`present_day_answer` field on any of its 3,335,698 turns. Topic 26 froze the
turn-level Stage-3 metadata contract specifically to avoid the upstream
evaluator's chain-level lookup discrepancy. Substituting the chain-level field
would therefore change the registered measurement and is not permitted.

No selection manifest, condition prompts, tokenizer measurements, model
weights, or model generations were produced.

## Exact command

From `26_temporal_scope_interference_reinstatement/`:

```bash
/usr/bin/time -v /home/xiang/venvs/ragen/bin/python \
  g0_temporal_scope.py prepare \
  --data data/merged_scope_benchmark.jsonl \
  --panel results/g0_panel.jsonl \
  --report results/g0_preflight.json
```

The command exited with status 1 and:

```text
HARD STOP: only 0 exact eligible items; need 512.
```

Peak resident memory was 4,862,240 KiB and elapsed time was 63.65 seconds.

## Environment

An existing local environment was used; no new experiment environment was
kept.

```text
host OS             Linux
Python              3.12.0
environment         /home/xiang/venvs/ragen
PyTorch             2.8.0+cu128
Transformers        4.57.6
huggingface_hub     0.36.2
CUDA available      true
GPU                 4 x NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
repository base     a5c02c04e8cceb29836765b507bcc03f5cad6e19
```

The frozen model was `Qwen/Qwen2.5-7B-Instruct`, but model revision is **not
applicable** because the support gate stopped the run before model/tokenizer
download.

## Dataset provenance

```text
source               yashkumaratri/ChronoScope on Hugging Face
filename             merged_scope_benchmark.jsonl
dataset revision     f2498a7e1820c403690393d49f8db81fb7390d21
downloaded bytes     1,272,579,530
LFS/local SHA-256    f070a174d6cc67105670919580b2af4028c8bfd06735429c78518748435d6156
upstream code URL    https://github.com/yashkumaratri/ChronoScope
upstream code HEAD   0e800a6b06eac83b34e1c54f54dc3f9ba5354566
```

The locally recomputed SHA-256 exactly matches the Hugging Face LFS metadata.

## Raw artifact audit

```text
raw chains                         1,469,628
raw turns                          3,335,698

family counts
  carryover                          552,408
  carryover_then                     324,637
  scope_switch                       233,098
  cross_entity_then                  144,618
  multi_turn_chain                    72,009
  interval_reasoning                  30,335
  temporal_narrative                  30,335
  distinct_count                      27,655
  interval_change                     27,655
  change_point                        24,357
  bridged_multi_pid                    2,521

turn-length distribution
  2                                1,254,761
  3                                   84,056
  4                                   87,758
  5                                   37,486
  6                                    4,502
  7                                      515
  8                                      229
  9                                      126
  10                                     182
  11                                      13

truth_type
  temporal                          1,098,999
  missing                             370,629
```

All turns had `year`, `pid`, `subject_label`, `answer`, and `question`.
Zero turns had `present_day_answer`. Chain-level `present_day_answer` existed
on 1,017,020 chains, confirming that the two metadata locations are not
interchangeable in this artifact.

Ten deterministic reservoir samples of real two-turn temporal `carryover`
chains were manually inspected. Their turn fields matched the counts above:
`year`, `pid`, `subject_label`, `answer`, and `value_label` were turn-level;
`present_day_answer` and `is_drift_candidate` were chain-level. The inspected
chain IDs included:

```text
265be213-ec6c-49e5-8d9f-62839c51d809
431c35d6-dc7f-4fbd-adc7-f977f9309256
26396074-4310-4645-8529-2cbf9a820d05
d49ebd78-2d16-4c31-a579-36134080601d
40a55d18-1a5f-45cb-b4fa-a76982de2cd5
9c54f977-f803-42a0-8603-90e6638a5606
dba8011d-349c-464e-9577-7a9f5c135322
d2700dde-a125-4678-a4fa-68d6d33ee622
ba04d076-19e4-46ef-a859-35ca46b18bbc
daa1814c-3993-45d3-958f-2d24044b844a
```

## Exact eligibility audit

The structural raw-candidate pool is the set with family `carryover`, truth
type `temporal`, and exactly two turns.

```text
raw candidate count                         324,637
eligible count                                    0
selected final N                                  0

rejection reason counts
  not_target_family                        1,144,991
  probe_not_drift_eligible                    324,637

forensic split of probe rejection
  final turn present answer missing           324,637
  evaluable historical == present                   0
  evaluable historical != present                   0

eligible family composition                     none
eligible PID/property composition                none
eligible historical-year distribution           none
eligible target entity uniqueness                   0
stable turn-level donor facts                       0
stable donor availability                           0
duplicate selected items                            0
selected item-ID collisions                         0
```

For completeness, before the turn-level present-answer gate the 324,637 raw
candidates had 14,264 unique normalized target entities, 324,637 unique chain
IDs, and 324,637 unique prospective item IDs with zero collisions.

Raw target PID composition:

```text
P102  76,866    P108  76,545    P39   76,859    P463  76,545
P169   7,008    P35    4,400    P6     4,400    P127   1,600
P286     207    P54      207
```

Raw historical-year composition:

```text
2022  64,460
2023  79,532
2024  79,200
2025 101,445
```

The chain-level value cannot repair this. Among raw candidates, 307,357 had a
chain-level value; it equaled turn 0's answer for 273,062 chains, turn 1's
answer for only 3,018, and neither answer for 34,112. It has no target PID, so
it is not a reliable released present value for the final probe.

The emitted preflight record is `results/g0_preflight.json` with SHA-256:

```text
be5647b3d5816feac15a3a327b7dc042390028ffb333c2141da199e48024e185
```

There is no selection manifest/hash because no item was selected. There are no
ten selected-item prompt examples or seven-condition prompt audit because no
exact item exists under the frozen contract.

## Static implementation audit

Before preflight, the condition constructor was checked without model output:

- the final probe is copied identically across all seven conditions;
- prior factual assistant answers use released Gold answers;
- semantic and bounded-present conditions use the same `stable_fact` object,
  hence the same target entity, PID/property, and value;
- the aside explicitly says it does not change the main time frame;
- the reinstatement text contains neither a concrete year nor an answer;
- filler generation and acknowledgement text are deterministic;
- selection uses only metadata, one seeded shuffle (`20260825`), and no model
  output;
- bootstrap resampling is paired by item;
- seed, target N, model name, decoding, token cap, and 5 pp criterion match the
  frozen documents.

Three implementation issues were identified before outcomes: the token-gap
check currently occurs after generation, stable donors are not indexed by the
target historical year, and the local relaxed scorer implements only a subset
of upstream prediction post-processing. None was reached and none can affect
the zero-support verdict. They were deliberately not repaired after the hard
stop, because this exact experiment cannot proceed and a repair must not be
used to manufacture eligibility.

## Measurement gates and G0 metrics

```text
exact-support gate       FAIL (0 < 512)
prompt token-gap gate    NOT RUN
model inference          NOT RUN
raw generations          NOT PRODUCED
condition metrics        NOT APPLICABLE
paired contrasts / CI    NOT APPLICABLE
error-flow matrices      NOT APPLICABLE
```

## Frozen verdict and interpretation

**`STOP_INSUFFICIENT_EXACT_SUPPORT`**

This result proves only that the frozen Topic 26 experiment is not executable
on the pinned official artifact under its registered turn-level metadata
contract. It provides no evidence for or against temporal decay, semantic
interference, present-default attraction, or reinstatement.

## Limitations and next-step decision

The public Stage-3 source currently writes turn-level present answers, but the
pinned merged public artifact does not expose them. That source/artifact
interface mismatch is the reason for the stop.

Per the frozen no-rescue rule, Topic 26 is archived here. Continuing would
require a separately registered experiment using a new officially released
artifact (or a separately justified reconstruction), with eligibility audited
before any model output. The present run will not switch fields, datasets,
matcher, family, seed, or sample size.
