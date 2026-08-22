"""Frozen G1: does one shared readout identify which policy wins the same state?"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_panel import aggregate_feature_replicates, concat_feature_npz
from .panel import aggregate_success, pair_state_table, validate_panel
from .relative_probe import SharedLinearProbe, bootstrap_relative_auc, paired_relative_metrics


def _dataset(behavior, feature_panel, a: str, b: str, *, min_trials: int, rate_gap: float):
    behavior = validate_panel(behavior, min_trials=min_trials)
    rates = aggregate_success(behavior, min_trials=min_trials)
    rates = rates[rates.checkpoint.isin([a, b])]
    winners = pair_state_table(behavior, a, b, min_trials=min_trials, rate_gap=rate_gap)
    winner_map = winners.set_index("state_id")["winner"].to_dict()
    rate_map = rates.set_index(["state_id", "checkpoint"])["success_rate"].to_dict()
    hash_map = rates.set_index(["state_id", "checkpoint"])["sim_state_hash"].to_dict()

    sid, cp, target, winner, feat = [], [], [], [], []
    for i in range(len(feature_panel.state_id)):
        s = str(feature_panel.state_id[i])
        c = str(feature_panel.checkpoint[i])
        if c not in {a, b} or s not in winner_map:
            continue
        key = (s, c)
        if key not in rate_map:
            continue
        if str(feature_panel.sim_state_hash[i]) != str(hash_map[key]):
            raise ValueError(f"behavior/feature sim_state_hash mismatch for {s}, {c}")
        sid.append(s)
        cp.append(c)
        target.append(float(rate_map[key]))
        winner.append(str(winner_map[s]))
        feat.append(feature_panel.feature[i])

    sid = np.asarray(sid)
    cp = np.asarray(cp)
    target = np.asarray(target, float)
    winner = np.asarray(winner)
    feat = np.asarray(feat, float)
    for s in np.unique(sid):
        present = sorted(cp[sid == s].tolist())
        if present != sorted([a, b]):
            raise ValueError(f"incomplete feature pair for state {s}: {present}")
    return sid, cp, target, winner, feat


def _unique_winner_counts(state_ids, winners) -> tuple[int, int]:
    m = {}
    for s, w in zip(state_ids, winners, strict=True):
        s, w = str(s), str(w)
        m.setdefault(s, w)
        if m[s] != w:
            raise ValueError(f"winner mismatch within state {s}")
    return sum(w == "A" for w in m.values()), sum(w == "B" for w in m.values())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-behavior", type=Path, nargs="+", required=True)
    p.add_argument("--test-behavior", type=Path, nargs="+", required=True)
    p.add_argument("--train-features", type=Path, nargs="+", required=True)
    p.add_argument("--test-features", type=Path, nargs="+", required=True)
    p.add_argument("--checkpoint-a", required=True)
    p.add_argument("--checkpoint-b", required=True)
    p.add_argument("--min-trials", type=int, default=8)
    p.add_argument("--rate-gap", type=float, default=0.50)
    p.add_argument("--min-feature-seeds", type=int, default=4)
    p.add_argument("--min-bidirectional", type=int, default=15)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--auc-min", type=float, default=0.70)
    p.add_argument("--auc-ci-lower-min", type=float, default=0.60)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    a, b = str(args.checkpoint_a), str(args.checkpoint_b)
    train_b = pd.concat([pd.read_csv(x) for x in args.train_behavior], ignore_index=True)
    test_b = pd.concat([pd.read_csv(x) for x in args.test_behavior], ignore_index=True)
    train_f = aggregate_feature_replicates(
        concat_feature_npz(args.train_features), min_seeds=args.min_feature_seeds
    )
    test_f = aggregate_feature_replicates(
        concat_feature_npz(args.test_features), min_seeds=args.min_feature_seeds
    )

    tr = _dataset(train_b, train_f, a, b, min_trials=args.min_trials, rate_gap=args.rate_gap)
    te = _dataset(test_b, test_f, a, b, min_trials=args.min_trials, rate_gap=args.rate_gap)
    if set(tr[0]) & set(te[0]):
        raise ValueError("train/test physical states overlap")

    probe = SharedLinearProbe(alpha=args.ridge_alpha).fit(tr[4], tr[2])
    test_scores = probe.score(te[4])
    a_wins, b_wins = _unique_winner_counts(te[0], te[3])

    report = {
        "checkpoint_a": a,
        "checkpoint_b": b,
        "train_states": int(len(set(tr[0]))),
        "test_states": int(len(set(te[0]))),
        "test_robust_a_wins": int(a_wins),
        "test_robust_b_wins": int(b_wins),
        "rate_gap": float(args.rate_gap),
        "min_trials": int(args.min_trials),
        "min_feature_seeds": int(args.min_feature_seeds),
        "probe": {"type": "shared_standardized_ridge", "alpha": float(args.ridge_alpha)},
    }

    if min(a_wins, b_wins) < args.min_bidirectional:
        report["verdict"] = "KILL_CROSSOVER_NOT_REPLICATED"
    else:
        metrics = paired_relative_metrics(te[0], te[1], te[3], test_scores, a, b)
        boot = bootstrap_relative_auc(
            te[0], te[1], te[3], test_scores, a, b, n_boot=args.bootstrap, seed=0
        )
        report["confirmation"] = metrics
        report["confirmation"]["relative_auc_ci95"] = boot["ci95"]
        lo = boot["ci95"][0]
        report["verdict"] = (
            "PASS_POLICY_SPECIFIC_SUCCESS_SIGNAL"
            if metrics["relative_auc"] >= args.auc_min
            and lo is not None
            and lo > args.auc_ci_lower_min
            else "KILL_SELF_KNOWLEDGE_INTERPRETATION"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
