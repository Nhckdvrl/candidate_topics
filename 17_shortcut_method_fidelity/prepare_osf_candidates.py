#!/usr/bin/env python3
"""Extract frozen Topic 17 candidates from the Standvoss et al. OSF workbook."""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
FAMILIES = {
    "immunostaining": re.compile(r"immunostain|immunohisto|immunofluorescen", re.I),
    "western_blot": re.compile(r"western blot|immunoblot", re.I),
}


def column_index(cell_ref: str) -> int:
    value = 0
    for char in re.match(r"[A-Z]+", cell_ref).group():
        value = value * 26 + ord(char) - 64
    return value - 1


def read_xlsx(path: Path) -> list[dict]:
    ns = {"m": NS}
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        strings = [
            "".join(node.text or "" for node in item.iter(f"{{{NS}}}t"))
            for item in root.findall("m:si", ns)
        ]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    raw_rows = []
    for row in sheet.findall(".//m:sheetData/m:row", ns):
        values = {}
        for cell in row.findall("m:c", ns):
            node = cell.find("m:v", ns)
            value = "" if node is None else node.text or ""
            if cell.get("t") == "s" and value:
                value = strings[int(value)]
            values[column_index(cell.get("r", ""))] = value
        raw_rows.append(values)
    headers = raw_rows[0]
    return [
        {headers[index]: value for index, value in row.items() if index in headers}
        for row in raw_rows[1:]
    ]


def candidates(rows: list[dict]) -> list[dict]:
    field = ""
    found = []
    for row in rows:
        field = row.get("field") or field
        pmcid = row.get("pmc_id", "").replace("PMC", "").strip()
        if not pmcid:
            continue
        for number in range(1, 149):
            shortcut_class = row.get(f"Cit{number}_SC")
            if shortcut_class not in {"probable", "possible"}:
                continue
            sentence = row.get(f"MethCit{number}", "").strip()
            for family, pattern in FAMILIES.items():
                if pattern.search(sentence):
                    found.append({
                        "candidate_id": f"PMC{pmcid}-cit{number}",
                        "field": field,
                        "method_family": family,
                        "parent_pmcid": f"PMC{pmcid}",
                        "parent_pmid": row.get("pubmed_id", "").strip(),
                        "parent_title": row.get("title", "").strip(),
                        "shortcut_number": number,
                        "shortcut_class": shortcut_class,
                        "shortcut_sentence": sentence,
                        "source": "Standvoss_2024_OSF_methodological_citations",
                    })
    return sorted(found, key=lambda x: (x["method_family"], x["parent_pmcid"], x["shortcut_number"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = candidates(read_xlsx(args.workbook))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {family: sum(r["method_family"] == family for r in rows) for family in FAMILIES}
    print(json.dumps({"output": str(args.output), "n": len(rows), "families": counts}, indent=2))


if __name__ == "__main__":
    main()
