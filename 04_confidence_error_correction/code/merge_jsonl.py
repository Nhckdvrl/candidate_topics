#!/usr/bin/env python3
"""Merge JSONL files with optional sorting and duplicate-ID protection."""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sort-key", default=None)
    ap.add_argument("--allow-duplicates", action="store_true")
    args = ap.parse_args()

    paths = []
    for pattern in args.inputs:
        matched = glob.glob(pattern)
        paths.extend(matched if matched else [pattern])
    paths = sorted(dict.fromkeys(paths))
    if not paths:
        raise SystemExit("no input files")

    rows = []
    seen = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if args.sort_key and args.sort_key in r:
                    key = r[args.sort_key]
                    if not args.allow_duplicates and key in seen:
                        raise ValueError(f"duplicate {args.sort_key}={key!r}")
                    seen.add(key)
                rows.append(r)
    if args.sort_key:
        rows.sort(key=lambda r: str(r.get(args.sort_key, "")))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"merged {len(paths)} files, {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
