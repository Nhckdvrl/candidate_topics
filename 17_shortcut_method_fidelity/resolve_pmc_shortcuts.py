#!/usr/bin/env python3
"""Resolve candidate shortcut sentences to cited resources using PMC XML."""

from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"


def get(url: str, attempts: int = 3) -> bytes:
    last = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "candidate-topics-topic17/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except Exception as exc:  # network failures are recorded per lineage
            last = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"GET failed: {url}: {last}")


def fetch_pmc(pmcid: str) -> bytes:
    """Fetch JATS XML, falling back to NCBI when Europe PMC lacks the body."""
    try:
        return get(f"{EUROPE_PMC}/{pmcid}/fullTextXML", attempts=1)
    except RuntimeError:
        numeric = pmcid.upper().removeprefix("PMC")
        return get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
            f"db=pmc&id={urllib.parse.quote(numeric)}"
        )


def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def element_text(node: ET.Element | None) -> str:
    return "" if node is None else " ".join("".join(node.itertext()).split())


def find_best_paragraph(root: ET.Element, sentence: str) -> tuple[ET.Element | None, float]:
    target = normalized(sentence)
    target_tokens = set(target.split())
    best = None
    best_score = 0.0
    for paragraph in root.findall(".//p"):
        text = normalized(element_text(paragraph))
        if not text:
            continue
        tokens = set(text.split())
        overlap = len(target_tokens & tokens) / max(1, len(target_tokens))
        ratio = difflib.SequenceMatcher(None, target[:800], text[:1600]).ratio()
        score = max(overlap, ratio)
        if score > best_score:
            best, best_score = paragraph, score
    return best, best_score


def ref_record(root: ET.Element, rid: str) -> dict | None:
    for ref in root.findall(".//ref"):
        if ref.get("id") != rid:
            continue
        pubs = {node.get("pub-id-type", ""): element_text(node) for node in ref.findall(".//pub-id")}
        return {
            "rid": rid,
            "title": element_text(ref.find(".//article-title")),
            "year": element_text(ref.find(".//year")),
            "doi": pubs.get("doi", ""),
            "pmid": pubs.get("pmid", ""),
            "pmcid": pubs.get("pmcid", ""),
            "citation": element_text(ref),
        }
    return None


def europe_pmc_lookup(ref: dict) -> str:
    if ref.get("pmcid"):
        value = ref["pmcid"].upper()
        return value if value.startswith("PMC") else f"PMC{value}"
    if ref.get("doi"):
        query = f'DOI:"{ref["doi"]}"'
    elif ref.get("pmid"):
        query = f'EXT_ID:{ref["pmid"]} AND SRC:MED'
    else:
        return ""
    url = f"{EUROPE_PMC}/search?query={urllib.parse.quote(query)}&format=json&pageSize=1"
    data = json.loads(get(url))
    results = data.get("resultList", {}).get("result", [])
    return results[0].get("pmcid", "") if results else ""


def relevant_methods(root: ET.Element, family: str) -> str:
    pattern = re.compile(
        r"immun|stain|histolog|antibod" if family == "immunostaining"
        else r"western|immunoblot|electrophores|membrane|antibod",
        re.I,
    )
    # JATS section titles are inconsistent across publishers and older PMC
    # deposits. Search body paragraphs directly; restricting to a title that
    # literally says "Methods" produced false missing-method calls.
    body = root.find(".//body")
    paragraphs = []
    for paragraph in (body.findall(".//p") if body is not None else []):
        text = element_text(paragraph)
        if pattern.search(text):
            paragraphs.append(text)
    return "\n".join(dict.fromkeys(paragraphs))[:30000]


def resolve(candidate: dict) -> dict:
    result = {**candidate, "resolution_status": "unresolved"}
    try:
        parent_xml = fetch_pmc(candidate["parent_pmcid"])
        parent_root = ET.fromstring(parent_xml)
        paragraph, similarity = find_best_paragraph(parent_root, candidate["shortcut_sentence"])
        result["paragraph_match_score"] = similarity
        result["current_evidence"] = element_text(paragraph)
        if paragraph is None or similarity < 0.45:
            result["resolution_status"] = "parent_sentence_not_found"
            return result
        rids = list(dict.fromkeys(x.get("rid", "") for x in paragraph.findall(".//xref[@ref-type='bibr']")))
        refs = [ref_record(parent_root, rid) for rid in rids]
        refs = [ref for ref in refs if ref]
        result["references"] = refs
        if not refs:
            result["resolution_status"] = "citation_not_resolved"
            return result
        # Freeze the first cited resource in textual order for G0; multi-resource
        # shortcuts remain visible in `references` and are not silently combined.
        target = refs[0]
        target_pmcid = europe_pmc_lookup(target)
        result["target_reference"] = target
        result["target_pmcid"] = target_pmcid
        if not target_pmcid:
            result["resolution_status"] = "target_not_open_pmc"
            return result
        target_root = ET.fromstring(fetch_pmc(target_pmcid))
        cited_methods = relevant_methods(target_root, candidate["method_family"])
        result["cited_method_evidence"] = cited_methods
        result["resolution_status"] = "resolved" if cited_methods else "target_method_not_found"
        return result
    except Exception as exc:
        result["resolution_status"] = "network_or_parse_error"
        result["error"] = str(exc)
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-family", type=int, default=12)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    candidates = [json.loads(line) for line in args.candidates.open(encoding="utf-8") if line.strip()]
    selected = []
    for family in sorted({row["method_family"] for row in candidates}):
        seen = set()
        for row in candidates:
            if row["method_family"] != family or row["parent_pmcid"] in seen:
                continue
            seen.add(row["parent_pmcid"])
            selected.append(row)
            if len(seen) == args.per_family:
                break
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(resolve, selected))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {}
    for row in rows:
        counts[row["resolution_status"]] = counts.get(row["resolution_status"], 0) + 1
    print(json.dumps({"selected": len(selected), "statuses": counts, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
