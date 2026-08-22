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


def _key(rec: dict) -> tuple[str, str, str, str]:
    return (rec["split"], rec["puzzle_id"], rec["variant_id"], rec["remasking"])


def _dedupe_or_fail(records: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for rec in records:
        key = _key(rec)
        if key in seen:
            raise ValueError(f"duplicate trace key: {key}")
        seen.add(key)
        out.append(rec)
    return out


def _native_fraction(rec: dict) -> float | None:
    value = rec.get("metadata", {}).get("native_digit_argmax_fraction")
    return None if value is None else float(value)


def analyze(records: list[dict], manifest: list[dict], split: str, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    records = _dedupe_or_fail([r for r in records if r["split"] == split])
    manifest_by_id = {r["puzzle_id"]: r for r in manifest}

    low_conf: dict[str, dict[str, dict]] = defaultdict(dict)
    random_controls: dict[str, dict] = {}
    for rec in records:
        if rec["remasking"] == "low_confidence":
            low_conf[rec["puzzle_id"]][rec["variant_id"]] = rec
        elif rec["remasking"] == "random" and rec["variant_id"] == "random-control":
            random_controls[rec["puzzle_id"]] = rec

    identity_exact: list[float] = []
    iso_exact_all: list[float] = []
    iso_retention_when_identity_exact: list[float] = []
    solve_flips = 0
    solve_pair_total = 0
    base_yes_iso_no = 0
    base_no_iso_yes = 0

    pair_taus_by_pid: dict[str, list[float]] = defaultdict(list)
    surface_null_by_pid: dict[str, list[float]] = defaultdict(list)
    boundary_null_by_pid: dict[str, list[float]] = defaultdict(list)
    excess_surface_by_pid: dict[str, list[float]] = defaultdict(list)
    excess_boundary_by_pid: dict[str, list[float]] = defaultdict(list)
    positional_delta_x: list[float] = []
    positional_delta_y: list[float] = []
    easy_rhos: list[float] = []
    repeat_taus: list[float] = []
    repeat_exact_agreement: list[float] = []
    random_taus: list[float] = []
    native_fractions: list[float] = []

    n_exact_pairs = 0
    exact_pair_puzzles: set[str] = set()

    for pid, variants in low_conf.items():
        base = variants.get("identity")
        if base is None:
            continue
        base_ok = bool(base["exact_solution"])
        identity_exact.append(float(base_ok))
        nf = _native_fraction(base)
        if nf is not None:
            native_fractions.append(nf)

        bsteps = _steps(base)
        repeat = variants.get("identity-repeat")
        if repeat is not None:
            rsteps = _steps(repeat)
            common = sorted(set(bsteps) & set(rsteps))
            if len(common) >= 2:
                repeat_taus.append(kendall_tau_b([bsteps[i] for i in common], [rsteps[i] for i in common]))
            repeat_exact_agreement.append(float(bool(repeat["exact_solution"]) == base_ok))

        # "Easy-first" is a characterization only. Compute within puzzle to avoid
        # a pooled correlation being driven by between-puzzle differences.
        if base_ok and pid in manifest_by_id:
            m = manifest_by_id[pid]
            ex, ey = [], []
            for k, count in m.get("candidate_counts", {}).items():
                i = int(k)
                if i in bsteps:
                    ex.append(float(count))
                    ey.append(float(bsteps[i]))
            if len(ex) >= 2:
                rho = spearman(ex, ey)
                if rho == rho:
                    easy_rhos.append(rho)

        ctrl = random_controls.get(pid)
        if ctrl is not None:
            cs = _steps(ctrl)
            common = sorted(set(bsteps) & set(cs))
            if len(common) >= 2:
                random_taus.append(kendall_tau_b([bsteps[i] for i in common], [cs[i] for i in common]))

        for variant_id, iso in variants.items():
            if not variant_id.startswith("iso-"):
                continue
            iso_ok = bool(iso["exact_solution"])
            iso_exact_all.append(float(iso_ok))
            solve_pair_total += 1
            if base_ok != iso_ok:
                solve_flips += 1
                if base_ok:
                    base_yes_iso_no += 1
                else:
                    base_no_iso_yes += 1
            if base_ok:
                iso_retention_when_identity_exact.append(float(iso_ok))
            if not (base_ok and iso_ok):
                continue

            t = SudokuTransform.from_dict(iso["transform"])
            isteps = _steps(iso)
            xs, ys, old_pos, new_pos, old_bd, new_bd = [], [], [], [], [], []
            for old_i, step in bsteps.items():
                new_i = t.map_index(old_i)
                if new_i not in isteps:
                    continue
                xs.append(float(step))
                ys.append(float(isteps[new_i]))
                old_pos.append(float(old_i))
                new_pos.append(float(new_i))
                old_bd.append(float(boundary_distance(old_i)))
                new_bd.append(float(boundary_distance(new_i)))
                positional_delta_x.append(float(boundary_distance(new_i) - boundary_distance(old_i)))
                positional_delta_y.append(float(isteps[new_i] - step))
            if len(xs) < 2:
                continue

            tau = kendall_tau_b(xs, ys)
            tau_surface = kendall_tau_b(old_pos, new_pos)
            tau_boundary = kendall_tau_b(old_bd, new_bd)
            pair_taus_by_pid[pid].append(tau)
            surface_null_by_pid[pid].append(tau_surface)
            boundary_null_by_pid[pid].append(tau_boundary)
            excess_surface_by_pid[pid].append(tau - tau_surface)
            excess_boundary_by_pid[pid].append(tau - tau_boundary)
            n_exact_pairs += 1
            exact_pair_puzzles.add(pid)
            nf = _native_fraction(iso)
            if nf is not None:
                native_fractions.append(nf)

    def puzzle_means(d: dict[str, list[float]]) -> list[float]:
        return [sum(v) / len(v) for v in d.values() if v]

    tau_puzzle = puzzle_means(pair_taus_by_pid)
    surface_puzzle = puzzle_means(surface_null_by_pid)
    boundary_puzzle = puzzle_means(boundary_null_by_pid)
    excess_surface_puzzle = puzzle_means(excess_surface_by_pid)
    excess_boundary_puzzle = puzzle_means(excess_boundary_by_pid)
    tau_ci = bootstrap_ci(tau_puzzle, seed=17) if tau_puzzle else (float("nan"), float("nan"))

    position_rho = spearman(positional_delta_x, positional_delta_y) if len(positional_delta_x) >= 2 else float("nan")
    min_identity = int(cfg.get("min_identity_exact_puzzles_for_interpretation", 16))
    min_order_pairs = int(cfg.get("min_both_exact_pairs_for_order_analysis", 30))
    min_order_puzzles = int(cfg.get("min_both_exact_puzzles_for_order_analysis", 12))
    repeat_floor = float(cfg.get("same_serialization_tau_floor", 0.95))

    n_identity_exact = int(sum(identity_exact))
    repeat_mean = summarize(repeat_taus)["mean"]

    return {
        "split": split,
        "n_puzzles_seen": len(low_conf),
        "n_identity_exact": n_identity_exact,
        "identity_exact_accuracy": sum(identity_exact) / len(identity_exact) if identity_exact else None,
        "isomorph_exact_accuracy_all": sum(iso_exact_all) / len(iso_exact_all) if iso_exact_all else None,
        "isomorph_exact_retention_given_identity_exact": (
            sum(iso_retention_when_identity_exact) / len(iso_retention_when_identity_exact)
            if iso_retention_when_identity_exact else None
        ),
        "solve_pair_total": solve_pair_total,
        "solve_flip_count": solve_flips,
        "solve_flip_rate": solve_flips / solve_pair_total if solve_pair_total else None,
        "solve_flip_directions": {
            "identity_correct_isomorph_wrong": base_yes_iso_no,
            "identity_wrong_isomorph_correct": base_no_iso_yes,
        },
        "n_both_exact_isomorph_pairs": n_exact_pairs,
        "n_puzzles_with_both_exact_pair": len(exact_pair_puzzles),
        "tau_iso_per_puzzle": summarize(tau_puzzle),
        "tau_iso_puzzle_cluster_bootstrap_95ci": list(tau_ci),
        "surface_order_positional_null_per_puzzle": summarize(surface_puzzle),
        "boundary_first_positional_null_per_puzzle": summarize(boundary_puzzle),
        "tau_excess_over_surface_null_per_puzzle": summarize(excess_surface_puzzle),
        "tau_excess_over_boundary_null_per_puzzle": summarize(excess_boundary_puzzle),
        "easy_first_candidate_count_spearman_per_puzzle": summarize(easy_rhos),
        "position_shift_vs_rank_shift_spearman_descriptive": position_rho,
        "same_serialization_repeat_tau": summarize(repeat_taus),
        "same_serialization_exact_outcome_agreement": summarize(repeat_exact_agreement),
        "random_remasking_tau_vs_identity": summarize(random_taus),
        "native_digit_argmax_fraction": summarize(native_fractions),
        "decision_flags": {
            "enough_identity_successes": n_identity_exact >= min_identity,
            "same_serialization_order_stable": repeat_mean is not None and repeat_mean >= repeat_floor,
            "enough_both_exact_for_order_tau": n_exact_pairs >= min_order_pairs and len(exact_pair_puzzles) >= min_order_puzzles,
        },
        "interpretation_note": (
            "Outcome flips under exact isomorphism are themselves evidence of non-equivariance; "
            "they must not be discarded merely because they cannot enter conditional order-tau analysis."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="results/g0_traces.jsonl")
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--config", default="LOCKED_CONFIG.json")
    ap.add_argument("--split", choices=["discovery", "confirmation"], default="discovery")
    ap.add_argument("--out", default="results/g0_summary.json")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    summary = analyze(read_jsonl(args.traces), read_jsonl(args.manifest), args.split, cfg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
