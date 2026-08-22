#!/usr/bin/env python
"""Re-grade frozen baseline responses after a grader-only correction.

This does not run inference or change the run contract; it preserves the
frozen responses and recomputes only the Math-Verify fields.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topic12.benchmarks import grade_math


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    destination = Path(args.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.source).open() as src, destination.open("w") as dst:
        for line in src:
            row = json.loads(line)
            correct, parse_ok, error = grade_math(row["response"], row["gold"])
            row.update(correct=bool(correct), parse_ok=bool(parse_ok), grade_error=error)
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
