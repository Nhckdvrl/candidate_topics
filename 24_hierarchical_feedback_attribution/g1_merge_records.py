"""Assemble the G1 four-condition panel, reusing G0 rows for RR and LL.

G0's `vla_replay` replays the whole recorded `vla_cmd` tape, i.e. both the
navigation/base channel and the upper-body channel come from the tape: that is
exactly G1's `RR`. G0's `fresh` runs both channels live: that is `LL`. So only
`LR` and `RL` need new rollouts.

The two are physically equivalent, not merely similar. In G0's `vla_replay` the
policy server is never queried (the queue is pre-filled); in G1 the server is
queried and its output is then overwritten field-by-field before the command
reaches the whole-body controller. Either way the command the WBC consumes is
byte-identical to the tape, `_last_base_height_cmd` is taken from that same
command, the observation is untouched, and the virtual clock advances once per
control tick in both. The discarded VLA forward pass has no path to the
simulator.

`server_queries` is preserved on every row and is meaningful in all four
conditions: `RR` (reused from G0's `vla_replay`) must be 0 because the VLA is
never queried there, while `LL`, `LR` and `RL` all run the VLA live and must be
greater than 0. It is evidence about which channels were live, so it is never
stripped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FORCE_N = 100.0
G0_TO_G1 = {"vla_replay": "RR", "fresh": "LL"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--g0-records", type=Path, required=True)
    p.add_argument("--g1-records", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    out: list[dict] = []

    for r in load_jsonl(args.g0_records):
        if float(r.get("force_n", -1)) != FORCE_N:
            continue
        cond = G0_TO_G1.get(str(r["condition"]))
        if cond is None:
            continue
        row = dict(r)
        row["condition"] = cond
        row["nav_replayed"] = cond in ("RR", "RL")
        row["upper_replayed"] = cond in ("RR", "LR")
        row["source"] = "g0_reused"
        out.append(row)

    for r in load_jsonl(args.g1_records):
        row = dict(r)
        row["source"] = "g1_new"
        out.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")

    counts: dict[tuple[str, str], int] = {}
    for row in out:
        k = (str(row["direction"]), str(row["condition"]))
        counts[k] = counts.get(k, 0) + 1
    print(f"wrote {len(out)} rows to {args.out}")
    for k in sorted(counts):
        print(f"  {k[0]:6s} {k[1]:3s}  {counts[k]}")


if __name__ == "__main__":
    main()
