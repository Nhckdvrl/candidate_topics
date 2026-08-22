#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topic12.stats import (
    bootstrap_rho,
    ci,
    competence_loss_curve,
    gate_label,
    intervention_label,
    net_accuracy_drop_curve,
    relation_stats,
    safe_spearman,
)


def parse_args():
    p = argparse.ArgumentParser(description="Analyze functional necessity vs published RL leverage")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--paper-table", default=str(ROOT / "data" / "qwen3_1p7b_table13_math.csv"))
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--topk", type=int, default=5)
    return p.parse_args()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def layer_index(path: Path) -> int:
    m = re.search(r"layer_(\d+)\.jsonl$", path.name)
    if not m:
        raise ValueError(path)
    return int(m.group(1))


def task_arrays(results_dir: Path):
    baseline = read_jsonl(results_dir / "baseline.jsonl")
    layer_paths = sorted(results_dir.glob("layer_*.jsonl"), key=layer_index)
    layers = [layer_index(p) for p in layer_paths]
    if layers != list(range(len(layers))):
        raise RuntimeError(f"Expected contiguous layer scan from 0; got {layers}")
    tasks = sorted({r["task"] for r in baseline})
    base_by_task, layer_by_task, uid_order = {}, {}, {}
    for task in tasks:
        base_rows = [r for r in baseline if r["task"] == task]
        uid_order[task] = [r["uid"] for r in base_rows]
        base_by_task[task] = np.array([bool(r["correct"]) for r in base_rows], dtype=float)
        matrix = []
        for path in layer_paths:
            rows = [r for r in read_jsonl(path) if r["task"] == task]
            by_uid = {r["uid"]: r for r in rows}
            if set(by_uid) != set(uid_order[task]):
                raise RuntimeError(f"{path.name}/{task}: UID set mismatch")
            matrix.append([bool(by_uid[u]["correct"]) for u in uid_order[task]])
        layer_by_task[task] = np.asarray(matrix, dtype=float)
    return np.array(layers), base_by_task, layer_by_task


def load_paper_table(path: Path):
    df = pd.read_csv(path)
    base = df[df["setting"] == "Base"].iloc[0]
    full = df[df["setting"] == "Full"].iloc[0]
    layers = df[df["setting"].str.startswith("Layer")].copy()
    layers["layer"] = layers["layer"].astype(int)
    return df, base, full, layers.sort_values("layer")


def paper_leverage_for_task(base, full, layers, task):
    col = {"math500": "math500", "gsm8k": "gsm8k"}[task]
    denom = float(full[col] - base[col])
    if denom == 0:
        raise RuntimeError(f"Paper full-vs-base gain is zero for {task}")
    return (layers[col].to_numpy(float) - float(base[col])) / denom


def paper_leverage_for_matched_tasks(base, full, layers, tasks):
    cols = [{"math500": "math500", "gsm8k": "gsm8k"}[t] for t in tasks]
    base_score = float(np.mean([float(base[c]) for c in cols]))
    full_score = float(np.mean([float(full[c]) for c in cols]))
    layer_score = np.mean(np.column_stack([layers[c].to_numpy(float) for c in cols]), axis=1)
    denom = full_score - base_score
    if denom == 0:
        raise RuntimeError("Paper full-vs-base gain is zero on matched task aggregate")
    return (layer_score - base_score) / denom


def z(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-12)


def generation_health_by_layer(results_dir: Path, layer_ids: np.ndarray):
    fallback, truncation = [], []
    for layer in layer_ids:
        rows = read_jsonl(results_dir / f"layer_{int(layer):02d}.jsonl")
        fallback.append(float(np.mean([not bool(r["parse_ok"]) for r in rows])))
        truncation.append(float(np.mean([bool(r["truncated"]) for r in rows])))
    return np.asarray(fallback), np.asarray(truncation)


def make_figure(out_path: Path, layer_ids, necessity, leverage, net_drop, task_rows):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(layer_ids, z(necessity), marker="o", label="paired necessity I (z)")
    axes[0].plot(layer_ids, z(leverage), marker="o", label="RL leverage C (z)")
    axes[0].plot(layer_ids, z(net_drop), marker=".", alpha=0.45, label="net acc drop (z)")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("standardized curve")
    axes[0].set_title("Full depth sweep")
    axes[0].legend(fontsize=8)

    axes[1].scatter(necessity, leverage)
    for layer, x, y in zip(layer_ids, necessity, leverage):
        axes[1].annotate(str(int(layer)), (x, y), fontsize=7, alpha=0.7)
    axes[1].set_xlabel("paired functional necessity")
    axes[1].set_ylabel("published RL leverage")
    axes[1].set_title("Layer-level relation")

    names = [r["task"] for r in task_rows]
    vals = [r["rho_conditional"] for r in task_rows]
    axes[2].bar(names, vals)
    axes[2].axhline(0.0, linewidth=1)
    axes[2].set_ylim(-1, 1)
    axes[2].set_ylabel("Spearman rho")
    axes[2].set_title("Task-matched correlations")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    layer_ids, base_by_task, layer_by_task = task_arrays(results_dir)
    _, paper_base, paper_full, paper_layers = load_paper_table(Path(args.paper_table))
    if not np.array_equal(layer_ids, paper_layers["layer"].to_numpy(int)):
        raise RuntimeError("Layer ids do not match the frozen paper table")

    available_tasks = [t for t in sorted(base_by_task) if t in {"math500", "gsm8k"}]
    if set(available_tasks) != {"math500", "gsm8k"}:
        raise RuntimeError("Locked primary analysis requires both math500 and gsm8k")

    conditional_by_task, net_by_task, leverage_by_task, task_rows = {}, {}, {}, []
    for task in available_tasks:
        base = base_by_task[task]
        layer = layer_by_task[task]
        conditional = competence_loss_curve(base, layer)
        net = net_accuracy_drop_curve(base, layer)
        leverage_task = paper_leverage_for_task(paper_base, paper_full, paper_layers, task)
        conditional_by_task[task] = conditional
        net_by_task[task] = net
        leverage_by_task[task] = leverage_task
        task_rows.append({
            "task": task,
            "baseline_accuracy": float(base.mean()),
            "baseline_solved_n": int(base.sum()),
            "rho_conditional": safe_spearman(conditional, leverage_task),
            "rho_net_drop": safe_spearman(net, leverage_task),
        })

    necessity = np.mean(np.vstack([conditional_by_task[t] for t in available_tasks]), axis=0)
    net_drop = np.mean(np.vstack([net_by_task[t] for t in available_tasks]), axis=0)
    leverage = paper_leverage_for_matched_tasks(paper_base, paper_full, paper_layers, available_tasks)
    leverage_c_math = paper_layers["c_math"].to_numpy(float)

    fallback_by_layer, truncation_by_layer = generation_health_by_layer(results_dir, layer_ids)
    intervention = intervention_label(necessity, fallback_by_layer, truncation_by_layer)

    gained_by_layer = np.mean(np.vstack([
        np.mean((base_by_task[t][None, :] == 0.0) & (layer_by_task[t] == 1.0), axis=1)
        for t in available_tasks
    ]), axis=0)
    task_profile_rho = safe_spearman(conditional_by_task["math500"], conditional_by_task["gsm8k"])

    stats = relation_stats(necessity, leverage, topk=args.topk)
    net_stats = relation_stats(net_drop, leverage, topk=args.topk)
    c_math_stats = relation_stats(necessity, leverage_c_math, topk=args.topk)
    boot = bootstrap_rho(
        {t: base_by_task[t] for t in available_tasks},
        {t: layer_by_task[t] for t in available_tasks},
        leverage, n_bootstrap=args.bootstrap, seed=args.seed, metric="conditional_loss",
    )
    low, high = ci(boot, level=0.90)
    gate = gate_label(stats.spearman_rho, low, high, stats.depth_residual_spearman, intervention)

    rows = []
    for i, layer in enumerate(layer_ids):
        row = {
            "layer": int(layer),
            "necessity_conditional_mean": float(necessity[i]),
            "net_accuracy_drop_mean": float(net_drop[i]),
            "paper_c_matched_tasks": float(leverage[i]),
            "paper_c_math_4task": float(leverage_c_math[i]),
            "baseline_wrong_to_correct_rate_all_items": float(gained_by_layer[i]),
            "parser_fallback_rate": float(fallback_by_layer[i]),
            "truncation_rate": float(truncation_by_layer[i]),
        }
        for task in available_tasks:
            row[f"necessity_conditional_{task}"] = float(conditional_by_task[task][i])
            row[f"net_accuracy_drop_{task}"] = float(net_by_task[task][i])
            row[f"paper_leverage_{task}"] = float(leverage_by_task[task][i])
        rows.append(row)
    pd.DataFrame(rows).to_csv(results_dir / "layer_relation.csv", index=False)

    contract = json.loads((results_dir / "run_contract.json").read_text(encoding="utf-8"))
    metrics = {
        "gate": gate,
        "intervention_informativeness": intervention,
        "residual_scale": contract.get("residual_scale"),
        "tasks": available_tasks,
        "num_layers": len(layer_ids),
        "primary_necessity": "P(ablated_wrong | baseline_correct), equal task weight",
        "primary_target": "paper_C_on_exact_matched_tasks",
        "spearman_rho": stats.spearman_rho,
        "spearman_bootstrap_90ci": [low, high],
        "kendall_tau": stats.kendall_tau,
        "depth_residual_spearman": stats.depth_residual_spearman,
        "depth_partial_rank_diagnostic": stats.depth_partial_rank,
        "circular_shift_p": stats.circular_shift_p,
        "task_profile_rho_math500_vs_gsm8k": task_profile_rho,
        "net_accuracy_drop_robustness": {
            "spearman_rho": net_stats.spearman_rho,
            "depth_residual_spearman": net_stats.depth_residual_spearman,
            "depth_partial_rank_diagnostic": net_stats.depth_partial_rank,
        },
        "c_math_4task_robustness": {
            "spearman_rho": c_math_stats.spearman_rho,
            "kendall_tau": c_math_stats.kendall_tau,
            "depth_residual_spearman": c_math_stats.depth_residual_spearman,
            "depth_partial_rank_diagnostic": c_math_stats.depth_partial_rank,
            "circular_shift_p": c_math_stats.circular_shift_p,
            "topk_overlap": c_math_stats.topk_overlap,
        },
        "topk": args.topk,
        "topk_overlap": stats.topk_overlap,
        "topk_expected_under_random": stats.topk_expected,
        "task_matched": task_rows,
        "generation_damage_diagnostics": {
            "rho_necessity_vs_parser_fallback": safe_spearman(necessity, fallback_by_layer),
            "rho_necessity_vs_truncation": safe_spearman(necessity, truncation_by_layer),
            "max_parser_fallback_rate": float(np.max(fallback_by_layer)),
            "max_truncation_rate": float(np.max(truncation_by_layer)),
            "fraction_layers_necessity_ge_0p90": float(np.mean(necessity >= 0.90)),
        },
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
    }
    (results_dir / "relation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    make_figure(results_dir / "relation.png", layer_ids, necessity, leverage, net_drop, task_rows)

    report = f"""# Topic 12 G-0 report

**Scientific gate:** `{gate}`  
**Intervention informativeness:** `{intervention}`

## Locked primary result

- Primary necessity I(l): **P(ablated wrong | baseline correct)**, averaged equally across MATH500 and GSM8K.
- Spearman rho(I, C_matched): **{stats.spearman_rho:.3f}**
- Paired item-bootstrap 90% CI: **[{low:.3f}, {high:.3f}]**
- Kendall tau: **{stats.kendall_tau:.3f}**
- Spearman between residuals after removing predeclared quadratic depth trends: **{stats.depth_residual_spearman:.3f}**
- partial-rank depth diagnostic (descriptive): **{stats.depth_partial_rank:.3f}**
- circular-shift p-value: **{stats.circular_shift_p:.3f}**
- top-{args.topk} overlap: **{stats.topk_overlap}** (random expectation {stats.topk_expected:.2f})
- MATH500-vs-GSM8K necessity-profile rho: **{task_profile_rho:.3f}**

## Locked robustness checks

- rho using legacy net accuracy drop as I: **{net_stats.spearman_rho:.3f}**
- rho against published four-task C_math: **{c_math_stats.spearman_rho:.3f}**
- rho(necessity, parser-fallback rate): **{safe_spearman(necessity, fallback_by_layer):.3f}**
- rho(necessity, truncation rate): **{safe_spearman(necessity, truncation_by_layer):.3f}**

## Task-level checks
"""
    for row in task_rows:
        report += (
            f"- {row['task']}: baseline acc={row['baseline_accuracy']:.3f}, "
            f"baseline solved n={row['baseline_solved_n']}, "
            f"rho(conditional I_task,C_task)={row['rho_conditional']:.3f}, "
            f"rho(net drop,C_task)={row['rho_net_drop']:.3f}\n"
        )

    report += """
## How to read this without fooling ourselves

The primary necessity measure is conditional on baseline-correct items. It asks
whether a layer is required for competence that demonstrably exists before the
intervention. This prevents chance wrong->correct flips from cancelling real
damage. Net accuracy drop is still reported because it matches the older layer-
ablation literature, but it is no longer the identification target.

The locked depth-shape gate correlates deviations after each raw curve has had
a quadratic function of normalized depth removed. This deliberately asks whether
neighboring layers line up beyond a broad middle-layer profile. A separate true
partial-rank diagnostic is reported, but it is descriptive because a quadratic
model of ranks does not perfectly absorb a nonlinear U-shaped rank profile.

Ablation-induced parser failure and runaway generation are outcomes, not rows to
filter. But if >=25% of layers lose >=90% of baseline-solved items, or >=25% of
layers have >=50% parser/truncation failure, full deletion is declared too
destructive to rank functional necessity cleanly. That is a measurement failure,
not a negative scientific result; run the predeclared alpha=0.5 full sweep.

`INCONCLUSIVE_DO_NOT_TUNE` means do not search layer subsets, task weights,
metrics, or new ablation definitions. Fix only genuine engineering/protocol
mismatches, or stop. `DISSOCIATION_CANDIDATE` is not yet a paper-level law: it
needs independent-model replication because the published RL curve itself is a
finite experimental estimate.
"""
    (results_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
