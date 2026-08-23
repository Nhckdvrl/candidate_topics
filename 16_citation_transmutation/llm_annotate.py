#!/usr/bin/env python3
"""LLM measurement runner for Topic 16.

Consumes retrieval-complete raw citation edges and produces the locked schema
accepted by g0_core.py. It targets an OpenAI-compatible /chat/completions API,
so local vLLM/SGLang servers can be used directly.

Important: this script is a measurement instrument, not ground truth. Its output
must be validated on a human gold set before a decisive G0 run.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import urllib.error
import urllib.request
from pathlib import Path

EVIDENCE_STATUSES = {
    "NONE",
    "OWN_PRIMARY",
    "EXTERNAL_PRIMARY",
    "SYNTHESIS",
    "UNKNOWN",
}


def extract_json_object(text: str) -> dict:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"model response is not a JSON object: {text[:300]!r}")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("model response JSON must be an object")
    return obj


def chat_completion(base_url: str, model: str, messages: list[dict], max_tokens: int) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP {exc.code}: {body[:1000]}") from exc
    return data["choices"][0]["message"]["content"]


def validate_raw_edge(obj: dict, lineno: int) -> None:
    required = [
        "edge_id",
        "claim_id",
        "source_paper_id",
        "citing_paper_id",
        "source_claim",
        "citing_claim",
        "source_context",
        "citing_context",
        "evidence_audit_complete",
        "evidence_bundle",
    ]
    missing = [key for key in required if key not in obj]
    if missing:
        raise ValueError(f"line {lineno}: missing fields {missing}")
    if not isinstance(obj["evidence_audit_complete"], bool):
        raise ValueError(f"line {lineno}: evidence_audit_complete must be Boolean")
    if not isinstance(obj["evidence_bundle"], list):
        raise ValueError(f"line {lineno}: evidence_bundle must be a list")


def proposition_and_evidence_prompt(obj: dict) -> str:
    bundle = json.dumps(obj["evidence_bundle"], ensure_ascii=False, indent=2)
    return f"""You are annotating one scientific citation edge.

Judge only the supplied text and evidence bundle. Return JSON only.

Definitions:
- same_core_proposition=true only when subject/entity, relation, direction,
  population/conditions, and scope match. Ignore only epistemic modality such as
  may/likely/established; modality is measured separately.
- directly_supported_by_source=true only if the source context genuinely supports
  the core proposition restated by the citing paper.
- evidence_status describes NEW SUPPORT available to the citing restatement beyond
  the cited source:
    NONE: audit is complete and no new supporting evidence is present.
    OWN_PRIMARY: the citing paper itself adds new primary evidence for this claim.
    EXTERNAL_PRIMARY: other primary studies add new support.
    SYNTHESIS: a meta-analysis/review/synthesis adds new support.
    UNKNOWN: evidence audit is incomplete or the status is ambiguous.
- If evidence_audit_complete is false, evidence_status MUST be UNKNOWN.

SOURCE CLAIM:
{obj['source_claim']}

SOURCE CONTEXT:
{obj['source_context']}

CITING CLAIM:
{obj['citing_claim']}

CITING CONTEXT:
{obj['citing_context']}

EVIDENCE_AUDIT_COMPLETE:
{obj['evidence_audit_complete']}

RETRIEVED EVIDENCE BUNDLE:
{bundle}

Return exactly:
{{"same_core_proposition": true|false,
  "directly_supported_by_source": true|false,
  "evidence_status": "NONE|OWN_PRIMARY|EXTERNAL_PRIMARY|SYNTHESIS|UNKNOWN",
  "reason": "brief reason"}}
"""


def certainty_prompt(a: str, b: str) -> str:
    return f"""Compare epistemic commitment to the shared scientific proposition.
Do NOT judge scientific truth. Do NOT infer chronology. You are not told which
statement is the source or citing paper.

A: {a}
B: {b}

Return JSON only:
{{"stronger": "A|B|SAME", "reason": "brief reason"}}
"""


def parse_certainty_answer(answer: dict) -> str:
    stronger = str(answer.get("stronger", "")).upper()
    if stronger not in {"A", "B", "SAME"}:
        raise ValueError(f"invalid certainty answer {stronger!r}")
    return stronger


def map_certainty_to_underlying(stronger: str, a_is_source: bool) -> str:
    if stronger == "SAME":
        return "SAME"
    source_stronger = (stronger == "A" and a_is_source) or (
        stronger == "B" and not a_is_source
    )
    return "DOWN" if source_stronger else "UP"


def judge_certainty_twice(
    obj: dict, base_url: str, model: str, max_tokens: int, rng: random.Random
) -> tuple[str, list[dict]]:
    first_a_is_source = bool(rng.randrange(2))
    first_a = obj["source_claim"] if first_a_is_source else obj["citing_claim"]
    first_b = obj["citing_claim"] if first_a_is_source else obj["source_claim"]
    second_a_is_source = not first_a_is_source
    second_a = obj["source_claim"] if second_a_is_source else obj["citing_claim"]
    second_b = obj["citing_claim"] if second_a_is_source else obj["source_claim"]

    raw1 = chat_completion(
        base_url,
        model,
        [{"role": "user", "content": certainty_prompt(first_a, first_b)}],
        max_tokens,
    )
    raw2 = chat_completion(
        base_url,
        model,
        [{"role": "user", "content": certainty_prompt(second_a, second_b)}],
        max_tokens,
    )
    ans1 = extract_json_object(raw1)
    ans2 = extract_json_object(raw2)
    shift1 = map_certainty_to_underlying(
        parse_certainty_answer(ans1), first_a_is_source
    )
    shift2 = map_certainty_to_underlying(
        parse_certainty_answer(ans2), second_a_is_source
    )
    shift = shift1 if shift1 == shift2 else "UNKNOWN"
    return shift, [ans1, ans2]


def annotate_edge(
    obj: dict, base_url: str, model: str, max_tokens: int, rng: random.Random
) -> dict:
    raw = chat_completion(
        base_url,
        model,
        [{"role": "user", "content": proposition_and_evidence_prompt(obj)}],
        max_tokens,
    )
    pe = extract_json_object(raw)
    same_core = pe.get("same_core_proposition")
    direct_support = pe.get("directly_supported_by_source")
    if not isinstance(same_core, bool) or not isinstance(direct_support, bool):
        raise ValueError("LLM must return Boolean proposition/support labels")
    evidence_status = str(pe.get("evidence_status", "")).upper()
    if evidence_status not in EVIDENCE_STATUSES:
        raise ValueError(f"invalid evidence_status {evidence_status!r}")
    if not obj["evidence_audit_complete"]:
        evidence_status = "UNKNOWN"

    certainty_shift, certainty_raw = judge_certainty_twice(
        obj, base_url, model, max_tokens, rng
    )

    return {
        "edge_id": str(obj["edge_id"]),
        "claim_id": str(obj["claim_id"]),
        "source_paper_id": str(obj["source_paper_id"]),
        "citing_paper_id": str(obj["citing_paper_id"]),
        "source_claim": str(obj["source_claim"]),
        "citing_claim": str(obj["citing_claim"]),
        "same_core_proposition": same_core,
        "directly_supported_by_source": direct_support,
        "evidence_audit_complete": obj["evidence_audit_complete"],
        "evidence_status": evidence_status,
        "certainty_shift": certainty_shift,
        "measurement_meta": {
            "model": model,
            "proposition_evidence_reason": pe.get("reason", ""),
            "certainty_judgments": certainty_raw,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default=os.environ.get("MODEL"))
    p.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    )
    p.add_argument("--max-tokens", type=int, default=700)
    p.add_argument("--seed", type=int, default=20260823)
    args = p.parse_args()
    if not args.model:
        p.error("set --model or MODEL")
    if args.max_tokens <= 0:
        p.error("--max-tokens must be > 0")

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as src, args.output.open(
        "w", encoding="utf-8"
    ) as dst:
        for lineno, line in enumerate(src, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            validate_raw_edge(obj, lineno)
            measured = annotate_edge(
                obj, args.base_url, args.model, args.max_tokens, rng
            )
            dst.write(json.dumps(measured, ensure_ascii=False) + "\n")
            dst.flush()


if __name__ == "__main__":
    main()
