#!/usr/bin/env python3
"""Retrieve auditable citation-edge candidates from the Europe PMC OA corpus.

This is a reconstruction utility, not a labeler.  It preserves stable IDs,
verbatim local contexts, all paragraphs that cite the resolved source, and
keyword-retrieved passages for later evidence audit.  Every produced row starts
with ``evidence_audit_complete=false``; only a subsequent explicit audit may
promote it to a measurement-eligible edge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9α-ωΑ-Ωβ-]+")
STOP = {"the", "and", "that", "with", "from", "this", "were", "was", "are", "for",
        "has", "have", "had", "into", "than", "may", "might", "could", "would", "should",
        "study", "studies", "paper", "results", "using", "used", "also", "which", "between"}


def get(url: str, *, binary: bool = False, retries: int = 4):
    req = urllib.request.Request(url, headers={"User-Agent": "Topic16-G0/1.0 (research audit)"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                payload = r.read()
            time.sleep(0.36)  # stay below Europe PMC's public 3 rps guidance
            return payload if binary else json.loads(payload)
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)


def norm(text: str) -> str:
    return " ".join(text.split())


def words(text: str) -> set[str]:
    return {x.lower() for x in TOKEN.findall(text) if len(x) > 2 and x.lower() not in STOP}


def sentences(text: str) -> list[str]:
    return [norm(x) for x in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", norm(text)) if len(x) > 25]


def render_with_citations(elem: ET.Element) -> str:
    chunks = [elem.text or ""]
    for child in elem:
        if child.tag.rsplit("}", 1)[-1] == "xref" and child.attrib.get("ref-type") == "bibr":
            chunks.append(f" [CIT:{child.attrib.get('rid', '')}] ")
        else:
            chunks.append(render_with_citations(child))
        chunks.append(child.tail or "")
    return norm("".join(chunks))


def ref_map(root: ET.Element) -> dict[str, dict]:
    out = {}
    for ref in root.findall(".//ref"):
        rid = ref.attrib.get("id")
        if not rid:
            continue
        ids = {}
        for node in ref.findall(".//pub-id"):
            ids[node.attrib.get("pub-id-type", "unknown")] = norm("".join(node.itertext()))
        out[rid] = {"ids": ids, "citation": norm("".join(ref.itertext()))}
    return out


def source_record(pmid: str, cache: dict[str, dict | None]) -> dict | None:
    if pmid in cache:
        return cache[pmid]
    query = urllib.parse.quote(f"EXT_ID:{pmid} AND SRC:MED")
    data = get(f"{API}/search?query={query}&resultType=core&format=json&pageSize=1")
    hits = data.get("resultList", {}).get("result", [])
    cache[pmid] = hits[0] if hits else None
    return cache[pmid]


def source_claim_for(citing: str, abstract: str) -> tuple[str, float]:
    cw = words(re.sub(r"\[CIT:[^]]+\]", "", citing))
    ranked = []
    for sent in sentences(abstract):
        sw = words(sent)
        overlap = len(cw & sw)
        score = overlap / max(1, min(len(cw), len(sw)))
        ranked.append((score, overlap, sent))
    if not ranked:
        return "", 0.0
    score, overlap, sent = max(ranked)
    return sent, score if overlap >= 3 else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("raw_candidates.jsonl"))
    p.add_argument("--n-articles", type=int, default=40)
    p.add_argument("--max-edges", type=int, default=350)
    p.add_argument("--min-overlap", type=float, default=0.45)
    p.add_argument("--query", default="OPEN_ACCESS:Y AND FIRST_PDATE:[2020 TO 2024]")
    args = p.parse_args()

    q = urllib.parse.quote(args.query)
    data = get(f"{API}/search?query={q}&resultType=core&format=json&pageSize={args.n_articles}")
    works = [x for x in data.get("resultList", {}).get("result", []) if x.get("pmcid")]
    cache: dict[str, dict | None] = {}
    rows = []
    for work in works:
        pmcid = work["pmcid"]
        try:
            raw_xml = get(f"{API}/{pmcid}/fullTextXML", binary=True)
            root = ET.fromstring(raw_xml)
        except Exception:
            continue
        refs = ref_map(root)
        parent = {child: node for node in root.iter() for child in node}
        paragraph_records = []
        for node in root.findall(".//body//p"):
            ancestor = node
            section = ""
            while ancestor in parent:
                ancestor = parent[ancestor]
                if ancestor.tag.rsplit("}", 1)[-1] == "sec":
                    title = ancestor.find("./title")
                    section = norm("".join(title.itertext())) if title is not None else ""
                    break
            paragraph_records.append((render_with_citations(node), section))
        rendered_paragraphs = [text for text, _ in paragraph_records]
        article_abstract = norm(" ".join("".join(x.itertext()) for x in root.findall(".//abstract")))
        for paragraph, section in paragraph_records:
            for sent in sentences(paragraph):
                markers = re.findall(r"\[CIT:([^]]+)\]", sent)
                rids = [rid for marker in markers for rid in marker.split()]
                if len(rids) != 1 or rids[0] not in refs:
                    continue
                ref = refs[rids[0]]
                pmid = ref["ids"].get("pmid")
                if not pmid:
                    continue
                record = source_record(pmid, cache)
                if not record or not record.get("abstractText"):
                    continue
                citing_claim = norm(re.sub(r"\s*\[CIT:[^]]+\]\s*", " ", sent))
                source_claim, overlap = source_claim_for(citing_claim, record["abstractText"])
                if overlap < args.min_overlap:
                    continue
                key = words(citing_claim) & words(source_claim)
                related = [p for p in rendered_paragraphs
                           if len(words(p) & key) >= max(3, min(6, len(key) // 2))]
                citing_source_passages = [p for p in rendered_paragraphs if f"[CIT:{rids[0]}]" in p]
                digest = hashlib.sha256(raw_xml).hexdigest()
                edge_key = hashlib.sha1(f"{pmcid}|{pmid}|{citing_claim}".encode()).hexdigest()[:12]
                rows.append({
                    "edge_id": f"epmc-{edge_key}",
                    "claim_id": f"pmid-{pmid}-{hashlib.sha1(source_claim.encode()).hexdigest()[:8]}",
                    "source_paper_id": f"PMID:{pmid}",
                    "citing_paper_id": pmcid,
                    "source_claim": source_claim,
                    "citing_claim": citing_claim,
                    "source_context": norm(record["abstractText"]),
                    "citing_context": paragraph,
                    "evidence_audit_complete": False,
                    "evidence_bundle": [
                        {"kind": "citing_article_abstract", "text": article_abstract},
                        {"kind": "all_passages_citing_resolved_source", "passages": citing_source_passages},
                        {"kind": "keyword_retrieved_full_text_passages", "passages": related},
                        {"kind": "source_reference", "text": ref["citation"]},
                    ],
                    "retrieval_meta": {
                        "provider": "Europe PMC REST", "citing_pmcid": pmcid,
                        "source_pmid": pmid, "source_doi": ref["ids"].get("doi"),
                        "source_reference_rid": rids[0],
                        "citing_section": section,
                        "citing_xml_sha256": digest, "lexical_overlap": overlap,
                        "query": args.query,
                    },
                })
                if len(rows) >= args.max_edges:
                    break
            if len(rows) >= args.max_edges:
                break
        if len(rows) >= args.max_edges:
            break

    # Stable ranking, with at most two edges per source claim to avoid early domination.
    rows.sort(key=lambda x: (-x["retrieval_meta"]["lexical_overlap"], x["edge_id"]))
    per_claim: dict[str, int] = {}
    selected = []
    for row in rows:
        n = per_claim.get(row["claim_id"], 0)
        if n >= 2:
            continue
        per_claim[row["claim_id"]] = n + 1
        selected.append(row)
    with args.output.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"articles_considered": len(works), "candidates": len(selected),
                      "independent_claims": len(per_claim), "output": str(args.output)}))


if __name__ == "__main__":
    main()
