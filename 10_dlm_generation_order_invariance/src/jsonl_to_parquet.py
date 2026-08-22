"""Convert the locked v4 JSONL corpus to Dream's official SFT parquet schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    frame = pd.DataFrame([{"prompt": row["prompt"], "response": row["response"]} for row in rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(json.dumps({"rows": len(frame), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
