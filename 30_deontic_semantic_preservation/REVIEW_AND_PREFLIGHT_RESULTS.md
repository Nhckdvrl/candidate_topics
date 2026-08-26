# Topic 30 code review and public-data preflight

Date: 2026-08-27

## Outcome

**Keep the topic, and continue to avoid treating regex natural-pair flags as prevalence.** Both public-data support questions pass, and the repaired gold-span controlled G0 now directly establishes generic-metric blindness. See `G0_RESULTS.md`.

## P0a: LexDeMod receipt

Source: `adobe-research/LexDeMod` commit `7a5e7d02a672bd9d586dd014378c5cd39bf5f7ad`.

| Split artifact | Rows | non-none rows | unique clauses | multi-label rows | parser validity |
|---|---:|---:|---:|---:|---:|
| train/eval annotations | 4,612 | 3,463 | 1,470 | 387 | 100% |
| test annotations | 1,777 | 1,238 | 477 | 162 | 100% |

All frozen P0a conditions pass. One released train/eval row has a label/span-key mismatch; it is reported separately and does not make the CSV structurally malformed.

## P0b: aligned simplification support

SIMPLE-LAW describes over 6,000 pairs, but a stable downloadable pair artifact was not located during this run. The preregistration permits a public substitute before outcome inspection. We used Lex-Simple commit `be3b5edb3fd5e2d62faba6770cfc9967ef717b64`, taking the two human reference files whose line counts exactly match the 1,000-line source file. The third file has 1,007 lines and was excluded rather than silently misaligned.

| Support statistic | Observed |
|---|---:|
| aligned pairs (two references) | 2,000 |
| unique source texts | 793 |
| deontic-eligible pairs | 446 (22.3%) |
| unique eligible source texts | 175 |
| surface modality occurrences | permission 254; obligation 180; prohibition 52; entitlement 18 |

All frozen P0b conditions pass. Repeated sources with different human references are legitimate pairs, but the unique-source count is reported to prevent the pair total from being mistaken for proposition diversity.

## Logic defects fixed

- The LexDeMod audit used `argmax` on a multi-label vector, silently dropping secondary gold labels. It now counts every active class and reports multi-label rows.
- Parser validity previously checked only vector length. It now validates required columns, binary labels, NONE exclusivity, span container shape, and offsets; blank clause IDs no longer count as a unique clause.
- A descriptive condition with no normative cue previously entered the deontic-eligible denominator. Eligibility now requires a modality cue.
- Multi-modality clauses were collapsed according to dictionary order. The triage representation now preserves the full modality set.
- Audit examples overwrote their source and simplified strings with parse dictionaries. Text and parses are now retained under distinct keys.
- The public third Lex-Simple reference is not line-aligned with the source under the repository's plain-text contract; it was excluded.
- Preflight explicitly uses Python 3 and is location-independent.

## Measurement limitation

On held-out LexDeMod test annotations, the current lexical baseline obtains 0.367 micro-precision, 0.596 recall, 0.454 F1, and 0.324 exact match for obligation, entitlement, prohibition, and permission. This is expected because LexDeMod is agent-specific: the same clause can be an obligation for one party and an entitlement for another, while a surface-modal matcher sees only `shall` or `may`.

On Lex-Simple, the triage scorer retrieves 82 modality-change, 6 condition-loss, and 41 negation-change pair flags. Inspection shows a mixture of genuine-looking candidates and false alarms caused by explicit-to-implicit paraphrase, epistemic/capability `can`, and non-deontic legal uses of `shall`. These counts are not a prevalence result.

## Completed controlled shot and natural-study next step

The controlled receipt now uses exact held-out gold spans and yields 222 clean contrasts; 97.75% retain MiniLM cosine >=0.95 despite changing normative force. Entitlement perturbations were removed because the agent-specific relation made the naive transformation invalid.

For the natural study:

1. Calibrate an actor-specific multi-label parser on LexDeMod train/eval and freeze it on test. Preserve actor, modality set, action, condition scope, and exception scope.
2. Use the current rules only to retrieve a balanced pool: modality-change candidates, condition/negation candidates, and matched no-change controls.
3. Blindly adjudicate a small stratified sample. If precision is adequate and real changes occur, estimate prevalence with correction for sampling; if not, do not kill the topic—move first to the already-registered controlled perturbation study.
4. For the paper narrative, compare generic meaning metrics (including a legal-meaning metric where licensing permits) against operator-targeted perturbations, then test a structure-preserving simplification constraint.

The ACL/EMNLP/NAACL-width contribution is not “we built a deontic extractor.” It is: normative force is a distinct, systematically fragile dimension of simplification; generic and even legal-domain meaning scores can miss controlled operator drift; and preserving an explicit normative structure reduces that failure without preventing simplification. A paper limited to regex counts would be too narrow and weak, while a paper claiming all legal meaning preservation would be too broad and collide with existing work.
