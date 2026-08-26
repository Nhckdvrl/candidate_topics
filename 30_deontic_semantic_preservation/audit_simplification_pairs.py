from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from deontic_structure import compare, extract


def rows(path: Path):
    if path.suffix.casefold() == ".jsonl":
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
    else:
        with path.open(encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)


def audit_pairs(path: Path, original_col: str = "original", simplified_col: str = "simplified") -> dict:
    n_pairs = 0
    eligible = 0
    flags = Counter()
    modalities = Counter()
    examples = []
    unique_originals = set()
    unique_eligible_originals = set()

    for row_number, row in enumerate(rows(path), start=1):
        missing = [column for column in (original_col, simplified_col) if column not in row]
        if missing:
            raise ValueError(f"row {row_number} missing columns: {missing}")
        original = str(row[original_col]).strip()
        simplified = str(row[simplified_col]).strip()
        if not original or not simplified:
            raise ValueError(f"row {row_number} contains an empty aligned text")

        n_pairs += 1
        unique_originals.add(original)
        parsed_original = extract(original)

        # A condition marker by itself is not deontic.  Eligibility requires a
        # normative modality cue; otherwise descriptive if-clauses inflate the
        # scientific denominator.
        if not parsed_original.modalities:
            continue

        eligible += 1
        unique_eligible_originals.add(original)
        modalities.update(parsed_original.modalities)
        comparison = compare(original, simplified)
        flag_names = ("modality_changed", "condition_lost", "exception_lost", "negation_changed")
        for key in flag_names:
            flags[key] += int(comparison[key])
        if len(examples) < 20 and any(comparison[key] for key in flag_names):
            examples.append(
                {
                    **comparison,
                    "original_text": original,
                    "simplified_text": simplified,
                }
            )

    eligible_rate = eligible / max(n_pairs, 1)
    gates = {
        "G_eligible_300": eligible >= 300,
        "G_eligible_rate_5pct": eligible_rate >= 0.05,
        "G_multiple_modalities": sum(value > 0 for value in modalities.values()) >= 2,
    }
    return {
        "file": str(path),
        "n_pairs": n_pairs,
        "n_unique_originals": len(unique_originals),
        "n_deontic_eligible": eligible,
        "n_unique_deontic_eligible_originals": len(unique_eligible_originals),
        "eligible_rate": eligible_rate,
        "original_modality_hist": dict(modalities),
        "structural_flag_counts": dict(flags),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "examples": examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--original-col", default="original")
    parser.add_argument("--simplified-col", default="simplified")
    parser.add_argument("--out", type=Path, default=Path("simplification_audit.json"))
    args = parser.parse_args()
    result = audit_pairs(args.input, args.original_col, args.simplified_col)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["gates"], indent=2))
    print("eligible", result["n_deontic_eligible"], "/", result["n_pairs"])


if __name__ == "__main__":
    main()
