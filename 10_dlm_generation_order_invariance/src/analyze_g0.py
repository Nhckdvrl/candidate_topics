from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from metrics import bootstrap_ci, kendall_tau_b, spearman, summarize
from schema import read_jsonl
from sudoku import SudokuTransform, boundary_distance


def _steps(rec: dict) -> dict[int, int]:
    return {int(k): int(v) for k, v in rec["finalization_step"].items()}


def analyze(records: list[dict], manifest: list[dict], split: str) -> dict:
    manifest_by_id = {r["puzzle_id"]: r for r in manifest}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        if rec["split"] == split and rec["remasking"] == "low_confidence":
            grouped[rec["puzzle_id"]].append(rec)

    base_correct = []
    iso_correct = []
    pair_taus = []
    positional_delta_x = []
    positional_delta_y = []
    easy_x, easy_y = [], []
    n_pairs = 0

    for pid, recs in grouped.items():
        base = next((r for r in recs if r["variant_id"] == "identity"), None)
        if base is None:
            continue
        base_correct.append(float(base["exact_solution"]))
        if not base["exact_solution"]:
            continue
        bsteps = _steps(base)
        m = manifest_by_id[pid]
        for k, count in m["candidate_counts"].items():
            i = int(k)
            if i in bsteps:
                easy_x.append(float(count))
                easy_y.append(float(bsteps[i]))

        for iso in [r for r in recs if r["variant_id"].startswith("iso-")]:
            iso_correct.append(float(iso["exact_solution"]))
            if not iso["exact_solution"]:
                continue
            t = SudokuTransform.from_dict(iso["transform"])
            isteps = _steps(iso)
            xs, ys = [], []
            for old_i, s in bsteps.items():
                new_i = t.map_index(old_i)
                if new_i not in isteps:
                    continue
                xs.append(float(s))
                ys.append(float(isteps[new_i]))
                positional_delta_x.append(float(boundary_distance(new_i) - boundary_distance(old_i)))
                positional_delta_y.append(float(isteps[new_i] - s))
            if len(xs) >= 2:
                pair_taus.append(kendall_tau_b(xs, ys))
                n_pairs += 1

    random_taus = []
    controls = {(r["puzzle_id"], r["variant_id"]): r for r in records if r["split"] == split and r["remasking"] == "random"}
    for pid, recs in grouped.items():
        base = next((r for r in recs if r["variant_id"] == "identity" and r["exact_solution"]), None)
        ctrl = controls.get((pid, "random-control"))
        if base is None or ctrl is None:
            continue
        bs, cs = _steps(base), _steps(ctrl)
        common = sorted(set(bs) & set(cs))
        if len(common) >= 2:
            random_taus.append(kendall_tau_b([bs[i] for i in common], [cs[i] for i in common]))

    easy_rho = spearman(easy_x, easy_y) if len(easy_x) >= 2 else float("nan")
    position_rho = spearman(positional_delta_x, positional_delta_y) if len(positional_delta_x) >= 2 else float("nan")
    tau_ci = bootstrap_ci(pair_taus, seed=17) if pair_taus else (float("nan"), float("nan"))

    return {
        "split": split,
        "n_puzzles_seen": len(grouped),
        "identity_exact_accuracy": sum(base_correct) / len(base_correct) if base_correct else None,
        "isomorph_exact_accuracy_conditional_on_identity_success": sum(iso_correct) / len(iso_correct) if iso_correct else None,
        "n_valid_exact_isomorph_pairs": n_pairs,
        "tau_iso": summarize(pair_taus),
        "tau_iso_mean_bootstrap_95ci": list(tau_ci),
        "seed_replication_candidate_count_vs_finalization_spearman": easy_rho,
        "position_shift_vs_rank_shift_spearman": position_rho,
        "random_remasking_tau_vs_identity": summarize(random_taus),
        "decision_flags": {
            "enough_exact_pairs": n_pairs >= 50,
            "seed_easy_first_present": bool(easy_rho == easy_rho and easy_rho >= 0.15),
            "random_control_sane": (not random_taus) or abs(sum(random_taus) / len(random_taus)) <= 0.20,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="results/g0_traces.jsonl")
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--split", choices=["discovery", "confirmation"], default="discovery")
    ap.add_argument("--out", default="results/g0_summary.json")
    args = ap.parse_args()
    summary = analyze(read_jsonl(args.traces), read_jsonl(args.manifest), args.split)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
