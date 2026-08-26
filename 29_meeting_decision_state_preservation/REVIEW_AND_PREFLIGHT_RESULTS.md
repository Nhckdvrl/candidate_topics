# Topic 29 code review and public-data preflight

Date: 2026-08-27

## Outcome

**Keep the topic.** The artifact is large enough, and the repaired temporal-prefix identification strategy has now passed a fixed-model G0. See `G0_RESULTS.md`.

## Reproducible artifact receipt

Source: official AMI manual annotations v1.6.2, converted with `guokan-shang/ami-and-icsi-corpora` commit `81716f66f88930dd54f28bf7f9e886605411835b`.

| Frozen support check | Observed | Result |
|---|---:|---|
| decision abstracts | 624 | pass (>=200) |
| multi-turn linked chains | 366 | pass (>=100) |
| chains spanning >=15 seconds | 281 | pass (>=75) |
| chains with conservative explicit state cue | 180 | pass (>=100) |

There are 137 converted summlink files, 129 of which contain a linked decision abstract. No malformed items, duplicate abstract IDs, or JSON parse errors were observed.

## Logic defects fixed

- `REJECTED` was incorrectly placed at the bottom of a total state order. This made rejection-to-proposal movement look like an upgrade. Rejection is now handled as a polarity branch; the headline upgrade is specifically a move to an unconditional `DECIDED` claim.
- No-cue text was silently labeled `OPEN`. It is now `UNKNOWN` and is excluded from the scorable source denominator.
- Transcript and minutes language used the same rule. Generic future `will` is now a summary-side commitment cue but not a high-precision source-side cue.
- The audit now deduplicates abstract IDs, sorts linked turns by time, uses first-start to last-end span, and reports malformed data and ambiguous windows.
- The score summary now reports its scorable denominator rather than dividing by all rows.
- Preflight explicitly uses Python 3 and is location-independent.

## Identification limitation

AMI `decisions` and summlink annotations identify an eventual decision proposition and supporting material. They do not label each linked utterance as proposed, conditional, tentative, or final. Therefore:

`proposal-looking linked turns -> decision abstract`

is not by itself a gold example of unsupported state upgrade. It can also arise when agreement is implicit (for example, uptake such as “yeah” or “okay”), when the extractive link omits the final commitment act, or when a condition belongs to the chosen product behavior rather than to the decision license.

This invalidates the claim that summlink alone supplies outcome ground truth. It does not invalidate the research question.

## Completed next shot

Summlink now anchors propositions and the runner constructs temporally truncated prefixes with explicit non-final cues. The fixed Qwen2.5-7B-Instruct run produced 39/52 unsupported unconditional upgrades under a neutral minutes prompt and 0/52 under a matched state-preservation prompt. Human adjudication remains necessary before presenting the number as corpus prevalence, but no further feasibility gate is required.

This remains a compact ACL/EMNLP/NAACL-sized question when framed as: (1) a state-preservation failure taxonomy and validated evaluation set, (2) prevalence and boundary-condition analysis across summarizers, and (3) a state-ledger or constrained-generation remedy. AMI scale is adequate; a second corpus is desirable for transfer, not for the first feasibility test.
