# Data and measurement pipeline

`build_gold.py` creates a frozen 68-edge adversarial measurement set with
same-core matches, scope/population/direction/mechanism near-misses, evidence
provenance variants, certainty shifts, incomplete audits, and four stable-ID
Greenberg-style historical-chain edges. Gold labels are stored separately in
`gold_*` fields and are not included in the annotation prompt.

`llm_annotate.py` consumes only the required raw edge fields and writes the
locked schema. It performs blind pairwise certainty judgments in both orders;
disagreement is `UNKNOWN`. Raw proposition and certainty responses are retained
under `measurement_meta`. `temperature=0` and the annotation seed are frozen.

`retrieve_open_edges.py` retrieves open-access Europe PMC contexts with stable
IDs and source-reference IDs. It defaults evidence audits to incomplete;
`prepare_g0_edges.py` requires an explicit human audit decision before a row can
enter a formal primary dataset. Since the gold measurement gate failed on the
prespecified known-case recovery check, that candidate dataset was intentionally
not promoted and no formal G0 effect was computed.
