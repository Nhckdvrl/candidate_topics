# Topic 28 — G0 Results

## Final decision

`GO_REVERSAL_OBJECT`

The frozen released-trajectory object passes every frozen gate. This establishes
a released AI trajectory phenomenon, not a mechanism, a causal order effect, or
a general cross-task paper claim.

## Environment and commands

- Repository worktree: `candidate_topics_t24`, branch `main`
- Base repository commit at execution: `f95288e` (`origin/main` at the time of the final run)
- Host: `fvcrc20`
- Python: `3.13.13`
- Environment: `/home/xiang/venvs/topic28`
- Packages: `datasets 3.6.0`, `huggingface_hub 1.28.0`, `numpy 2.5.2`, `pandas 3.0.5`, `pyarrow 25.0.1`

Commands:

```bash
cd 28_progressive_truthful_clue_reversal
python -m unittest discover -s tests -v
PATH=/home/xiang/venvs/topic28/bin:$PATH bash run_g0.sh
```

The final full run used no `--response-configs` subset, no `--include-human`,
and no alternate scorer or threshold. The final run completed 9/9 unit tests
and loaded all 128 response configs.

## Dataset receipt and schema audit

Frozen data objects:

| Object | Revision | Split/config |
|---|---|---|
| `mgor/protobowl-11-13-agent-responses` | `a6d18c63e08e6cf9ad56b529ce5b10e217240e36` | `train`, all configs |
| `mgor/protobowl-11-13` | `3dae05a66d3e0fd8c6b23ef8656ff6f4437bb1d4` | `progressive-clues` / `eval` |

Response schema audit:

- available configs: `128`; loaded configs: `128`; debug subset: `false`;
- raw response rows: `362,120`;
- every config had exactly `agent_type`, `qc_id`, `answer`, `prediction`, `score`;
- `score` was a float in exactly `{0.0, 1.0}`: `247,417` ones and `114,703` zeros;
- all `362,120` `qc_id` values matched `q<digits>_<digits>`;
- `prediction` and `answer` were strings for all rows;
- response agent types were `ai` (`282,906`) and `human_team` (`79,214`).

Question schema audit:

- `progressive-clues/eval` had `3,042` rows and `3,042` unique `qc_id` values;
- columns were `qc_id`, `clue_text`, `n_clues`, `clean_answers`, `orig_qid`,
  `full_quiz_question`, `clue_spans`, `orig_answer_string`, and `metadata`;
- `clean_answers` was a list, `clue_spans` a list at dataset level, and
  `metadata` a dict for all rows;
- the implementation's `qc_id`/`orig_qid`/`n_clues` contract had zero failures.

The stable trajectory key was `(config, agent_type, qid)` with state key
`clue_idx`. After cleaning there were `72,612` trajectories, `61,912` ever-correct
trajectories, and only `ai` trajectories in the scientific analysis.

## Cleaning and duplicate audit

| Audit item | Result |
|---|---:|
| bad qc rows dropped | 0 |
| non-binary score rows dropped | 0 |
| human rows seen | 79,214 |
| human rows included | false |
| question join coverage before contract filter | 1.000000 |
| metadata-contract rows dropped | 0 |
| clean joined rows | 281,620 |
| frozen ambiguous duplicate cells dropped | 372 |

Independent duplicate classification at the frozen cell key
`(config, agent_type, qid, clue_idx)` found `914` duplicate cells: `542` were
identical and safe to collapse, `129` disagreed on correctness, and `243` had
consistent correctness but conflicting normalized predictions. The frozen
implementation intentionally treats both disagreement types as ambiguous and
drops `372` cells; no row was selected post hoc.

## Primary results

Primary correctness is the released `score`. Only adjacent cumulative clue
states count; gap pairs are diagnostics only.

| Quantity | Observed |
|---|---:|
| `0 -> 0` transitions | 55,694 |
| `0 -> 1` transitions | 32,485 |
| `1 -> 1` transitions | 112,251 |
| eligible adjacent transitions from correct | 120,353 |
| primary official `1 -> 0` events | 8,102 |
| primary reversal rate | 0.0673186 (6.7319%) |
| clustered bootstrap 95% CI | [0.0658642, 0.0688079] |
| unique primary reversal questions | 760 |
| configs with primary reversals | 93 |
| gap pairs excluded from primary | 368 |
| gap `1 -> 0` reversals | 1 |

All final `reversal_events.csv` rows had `to_clue - from_clue == 1`; no gap
reversal entered the primary event table.

Recovery diagnostics use the current frozen code's trajectory-level flags
(they are not an event-level estimator): among `7,851` reversal-containing
trajectories, `3,616` had immediate recovery (`46.06%`) and `4,567` had
eventual recovery (`58.17%`).

## Strict alias audit

The strict SQuAD-style alias check was diagnostic only and did not replace
released score. Among the `8,102` official score reversals:

| strict alias before | strict alias after | Events |
|---|---|---:|
| false | false | 2,618 |
| true | false | 3,871 |
| true | true | 1,613 |
| false | true | 0 |

Thus `3,871` events are strict-alias-supported reversals and pass the frozen
support gate. The primary score event count remains `8,102`; the alias check
was not used to rescore or filter it.

## Added-clue audit

A deterministic sample of `50` primary events (`random_state=20260825`) was
checked against the released question artifact. All `50/50` passed every check:

- both before/after `qc_id` rows existed;
- the transition was the exact next official clue;
- `new_clue_text` matched the frozen `clue_spans` extraction and was non-empty;
- before/after cumulative text matched the corresponding question rows;
- before/after cumulative text matched the official full-question span prefixes.

Representative events:

```text
T0-11b_1shot, q1011, clue 2 -> 3:
  Pharaoh Ramesses II -> Theban king
  added: This son of Seti I built several complexes, including a tomb in the
  Valley of the Queens for his wife Nefertari.

T0-11b_1shot, q1020, clue 2 -> 3:
  The Curiosity rover -> The Opportunity rover
  added: (*) For 10 points, name this semi-autonomous, car-sized rover
  currently conducting experiments on a certain red planet.

T0-11b_1shot, q1052, clue 1 -> 2:
  melting point -> fusion
  added: Salt is placed on roads to favor this process over its opposite.
```

## Concentration audit

Reversals are uneven across configs but not concentrated in one config or one
question:

- all `93/93` non-human configs had eligible correct-state transitions and at
  least one reversal;
- config-level reversal-density quantiles were min `0.00093`, Q1 `0.03402`,
  median `0.06405`, Q3 `0.13128`, max `0.47676`;
- the top config contributed `311/8,102` events (`3.84%`), top 5 `14.93%`,
  and top 10 `25.67%`;
- the top category was Science with `2,524/8,102` (`31.15%`), followed by
  History `1,444` and Literature `1,086`;
- the top question contributed `40/8,102` (`0.49%`), top 10 questions
  `4.52%`, and top 50 `17.67%`.

The event table contains only `agent_type=ai`; no human team drives the result.
The spread across 93 configs and the low per-question concentration argue
against a single model/config or a handful of malformed questions being the
sole source, while the density heterogeneity should be reported rather than
averaged away.

## Frozen gate table

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| question join coverage | >= 0.98 | 1.000000 | PASS |
| eligible correct-state transitions | >= 500 | 120,353 | PASS |
| primary reversal events | >= 100 | 8,102 | PASS |
| primary reversal rate | >= 0.02 | 0.0673186 | PASS |
| unique reversal questions | >= 50 | 760 | PASS |
| unique reversal configs | >= 5 | 93 | PASS |
| strict-alias-supported events | >= 30 | 3,871 | PASS |

## Engineering and measurement notes

The first full run exposed one narrow output-contract bug: after Hugging Face
`to_pandas()`, `clue_spans` and inner spans are NumPy arrays, while the helper
accepted only list/tuple. That first run therefore had blank `new_clue_text`,
although its primary score counts and gates were already complete. The repair
extended the helper's container types and added a regression test. The full
128-config G0 was then rerun; all primary counts and the verdict were identical,
and the final event artifact passed the 50-case added-clue audit. This was an
engineering/artifact-output repair, not an outcome-driven scientific repair.

There was no scientific failure, measurement failure, or artifact-data failure:
the released scores, joins, qc_id contract, duplicate policy, and human
exclusion all passed. The only repaired issue was the event receipt's clue-text
serialization contract.

## Verdict and next step

`GO_REVERSAL_OBJECT` means that systematic released AI trajectories exist where
adding the next truthful QuizBowl clue changes an already-correct answer to an
incorrect one at nontrivial density. It does not establish why the reversal
happens, that the correct representation was erased, causal order dependence,
modern open-model reproduction, or a paper-level cross-task construct.

Per the frozen protocol, the next step is analysis design only: competitor
introduction, clue specificity/ambiguity, fixed-multiset order permutation,
recovery dynamics, and a matched local open-model reproduction. Do not jump to
mechanism or hidden-state work from this G0 alone.
