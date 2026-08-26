# Topic 30 gold-span controlled G0 result

Date: 2026-08-27

## Verdict

**PASS: normative-force changes are nearly invisible to ordinary similarity while reversing legally material meaning. Keep the topic, with natural simplification drift as the main empirical question.**

## Identification

`run_g0_controlled.py` uses held-out LexDeMod gold labels and gold trigger spans. It retains clauses with exactly one active headline class and edits only the annotated trigger—not the first surface modal found in the sentence.

The valid controlled classes are:

- obligation→permission;
- permission→obligation;
- prohibition loss.

Entitlement is intentionally excluded from this controlled receipt. LexDeMod labels are actor-specific: one party's obligation can be the other party's entitlement, so swapping that surface modal does not necessarily change the annotated actor's entitlement into an obligation.

## Fixed run

- Gold source: LexDeMod held-out test CSV.
- Controlled contrasts: 222 (100 obligation→permission, 87 permission→obligation, 35 prohibition-loss).
- Embedding metric: `sentence-transformers/all-MiniLM-L6-v2`, snapshot `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Natural comparison distribution: 2,000 aligned Lex-Simple human simplification pairs.

| Metric | Mean | Rate >=0.90 | Rate >=0.95 |
|---|---:|---:|---:|
| token Jaccard | 0.942 | 88.74% | 48.20% |
| sequence similarity | 0.930 | 91.44% | 89.19% |
| TF-IDF cosine | 0.966 | 96.85% | 81.08% |
| MiniLM cosine | 0.986 | 98.65% | 97.75% |
| MiniLM on natural Lex-Simple pairs | 0.917 | 69.75% | 47.90% |

By change type, every obligation↔permission contrast scored at least 0.95 MiniLM cosine. Even prohibition loss scored at least 0.95 in 30/35 cases (85.7%). Thus a materially reversed rule is usually *more similar* than an ordinary human simplification pair.

## Review repairs made before freezing the result

- Replaced sentence-level regex selection with exact LexDeMod gold-span targeting.
- Removed case-insensitive replacements that could corrupt capitalized defined terms.
- Removed malformed partial replacements such as `shall must`.
- Excluded multi-label clauses so a second gold operator cannot confound the contrast.
- Excluded the conceptually invalid actor-specific entitlement perturbation.
- Manually inspected a deterministic sample across all three retained change types; edits were grammatical minimal operator changes, with no unchanged or `shall must` outputs.

## What this establishes—and what it does not

It establishes the registered G0 claim that high generic similarity does not protect normative force. Combined with 446 deontic-eligible Lex-Simple pairs, it also establishes practical data and method runway. This is enough to reject the “dead topic” hypothesis.

It does not yet establish natural-drift prevalence in human or model simplifications, nor does MiniLM stand in for all modern factuality/legal metrics. The paper-scale study must add actor-aware outcome annotation, multiple simplifiers, stronger baselines (including a legal-meaning metric), and a preservation-constrained method.
