from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path

from deontic_structure import extract


LABELS = ("obl", "ent", "pro", "per", "pow", "dis", "none")
SPAN_LABEL = {"obl": "obl", "ent": "ent", "pro": "pro", "per": "per", "pow": "oth", "dis": "nen", "none": "none"}


def _parse_row(row: dict) -> tuple[tuple[str, ...], dict]:
    vector = ast.literal_eval(row["label"])
    spans = ast.literal_eval(row["span"])
    if not isinstance(vector, (list, tuple)) or len(vector) != len(LABELS):
        raise ValueError("label must be a seven-element vector")
    if any(type(value) not in (int, bool) or value not in (0, 1) for value in vector):
        raise ValueError("label vector must be binary")
    active = tuple(LABELS[index] for index, value in enumerate(vector) if value)
    if not active or ("none" in active and len(active) > 1):
        raise ValueError("label vector must have an active class and NONE must be exclusive")
    if not isinstance(spans, dict):
        raise ValueError("span must be a dictionary")
    for key, values in spans.items():
        if not isinstance(key, str) or not isinstance(values, list):
            raise ValueError("invalid span dictionary")
        for value in values:
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or any(type(index) is not int or index < 0 for index in value)
                or value[1] < value[0]
            ):
                raise ValueError("span offsets must be non-negative [start, end] pairs")
    return active, spans


def audit_csv(path: Path) -> dict:
    counts = Counter()
    span_counts = Counter()
    n_rows = 0
    unique_cids = set()
    malformed = 0
    missing_cid = 0
    multi_label_rows = 0
    span_label_mismatches = 0
    malformed_examples = []
    extractor_tp = extractor_fp = extractor_fn = extractor_exact = 0
    extractor_rows = 0
    extractor_label_map = {
        "obl": "OBLIGATION",
        "ent": "ENTITLEMENT",
        "pro": "PROHIBITION",
        "per": "PERMISSION",
    }

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"cid", "text", "label", "span"}
        missing_columns = required - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"missing required columns: {sorted(missing_columns)}")
        for row_number, row in enumerate(reader, start=2):
            n_rows += 1
            cid = str(row.get("cid", "")).strip()
            if cid:
                unique_cids.add(cid)
            else:
                missing_cid += 1
            try:
                active, spans = _parse_row(row)
                counts.update(active)
                multi_label_rows += int(len(active) > 1)
                for key, values in spans.items():
                    span_counts[key] += len(values)
                expected_span_labels = {SPAN_LABEL[label] for label in active}
                span_label_mismatches += int(set(spans) != expected_span_labels)
                gold = {extractor_label_map[label] for label in active if label in extractor_label_map}
                predicted = set(extract(row["text"]).modalities)
                extractor_tp += len(gold & predicted)
                extractor_fp += len(predicted - gold)
                extractor_fn += len(gold - predicted)
                extractor_exact += int(gold == predicted)
                extractor_rows += 1
            except (KeyError, SyntaxError, TypeError, ValueError) as exc:
                malformed += 1
                if len(malformed_examples) < 10:
                    malformed_examples.append({"row": row_number, "error": str(exc)})

    valid_rows = n_rows - malformed
    non_none = valid_rows - counts["none"]
    parse_validity = valid_rows / max(n_rows, 1)
    gates = {
        "G_gold_non_none_500": non_none >= 500,
        "G_unique_clauses_250": len(unique_cids) >= 250,
        "G_parse_clean_99pct": parse_validity >= 0.99,
    }
    precision = extractor_tp / max(extractor_tp + extractor_fp, 1)
    recall = extractor_tp / max(extractor_tp + extractor_fn, 1)
    extractor_receipt = {
        "scope": "diagnostic lexical baseline on OBLIGATION/ENTITLEMENT/PROHIBITION/PERMISSION; not a gate",
        "n_rows": extractor_rows,
        "exact_match": extractor_exact / max(extractor_rows, 1),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }
    return {
        "file": str(path),
        "n_rows": n_rows,
        "n_unique_clauses": len(unique_cids),
        "missing_cid": missing_cid,
        "label_counts_multilabel": dict(counts),
        "multi_label_rows": multi_label_rows,
        "span_counts": dict(span_counts),
        "span_label_mismatches": span_label_mismatches,
        "malformed": malformed,
        "malformed_examples": malformed_examples,
        "parser_validity": parse_validity,
        "non_none": non_none,
        "extractor_receipt": extractor_receipt,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("lexdemod_audit.json"))
    args = parser.parse_args()
    result = audit_csv(args.csv)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
