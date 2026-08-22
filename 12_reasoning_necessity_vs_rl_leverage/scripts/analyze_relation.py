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

from topic12.stats import bootstrap_rho, ci, gate_label, relation_stats, safe_spearman


def parse_args():
    p = argparse.ArgumentParser(description="Analyze functional necessity vs published RL leverage")
    p.add_argument("--results-dir", required=True)
    p.add_argument(
        "--paper-table",
        default=str(ROOT / "data" / "qwen3_1p7b_table13_math.csv"),
    )
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
    base_by_task = {}
    layer_by_task = {}
    uid_order = {}

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
    layers = layers.sort_values("layer")
    return df, base, full, layers


def paper_leverage_for_task(base, full, layers, task):
    col = {"math500": "math500", "gsm8k": "gsm8k"}[task]
    denom = float(full[col] - base[col])
    if denom == 0:
        raise RuntimeError(f"Paper full-vs-base gain is zero for {task}")
    return (layers[col].to_numpy(float) - float(base[col])) / denom


def paper_leverage_for_matched_tasks(base, full, layers, tasks):
    """Apply the paper's C formula to exactly the tasks in our necessity curve."""
    cols = [{"math500": "math500", "gsm8k": "gsm8k"}[t] for t in tasks]
    base_score = float(np.mean([float(base[c]) for c in cols]))
    full_score = float(np.mean([float(full[c]) for c in cols]))
    layer_score = np.mean(
        np.column_stack([layers[c].to_numpy(float) for c in cols]),
        axis=1,
    )
    denom = full_score - base_score
    if denom == 0:
        raise RuntimeError("Paper full-vs-base gain is zero on matched task aggregate")
    return (layer_score - base_score) / denom


def z(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-12)


def generation_health_by_layer(results_dir: Path, layer_ids: np.ndarray):
    fallback = []
    truncation = []
    for layer in layer_ids:
        rows = read_jsonl(results_dir / f"layer_{int(layer):02d}.jsonl")
        fallback.append(float(np.mean([not bool(r["parse_ok"]) for r in rows])))
        truncation.append(float(np.mean([bool(r["truncated"]) for r in rows])))
    return np.asarray(fallback), np.asarray(truncation)


def make_figure(out_path: Path, layer_ids, necessity, leverage, task_rows):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(layer_ids, z(necessity), marker="o", label="necessity I (z)")
    axes[0].plot(layer_ids, z(leverage), marker="o", label="RL leverage C (z)")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("standardized curve")
    axes[0].set_title("Full depth sweep")
    axes[0].legend()

    axes[1].scatter(necessity, leverage)
    for layer, x, y in zip(layer_ids, necessity, leverage):
        axes[1].annotate(str(int(layer)), (x, y), fontsize=7, alpha=0.7)
    axes[1].set_xlabel("functional necessity (accuracy drop)")
    axes[1].set_ylabel("published RL leverage")
    axes[1].set_title("Layer-level relation")

    names = [r["task"] for r in task_rows]
    vals = [r["rho"] for r in task_rows]
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

    paper_layer_ids = paper_layers["layer"].to_numpy(int)
    if not np.array_equal(layer_ids, paper_layer_ids):
        raise RuntimeError(
            f"Layer mismatch: results={layer_ids.tolist()} paper={paper_layer_ids.tolist()}"
        )

    available_tasks = [t for t in sorted(base_by_task) if t in {"math500", "gsm8k"}]
    if not available_tasks:
        raise RuntimeError("Need at least one of math500/gsm8k")

    necessity_by_task = {}
    leverage_by_task = {}
    task_rows = []
    for task in available_tasks:
        base_acc = float(base_by_task[task].mean())
        layer_acc = layer_by_task[task].mean(axis=1)
        necessity = base_acc - layer_acc
        leverage = paper_leverage_for_task(paper_base, paper_full, paper_layers, task)
        necessity_by_task[task] = necessity
        leverage_by_task[task] = leverage
        task_rows.append(
            {
                "task": task,
                "baseline_accuracy": base_acc,
                "mean_layer_accuracy": float(layer_acc.mean()),
                "rho": safe_spearman(necessity, leverage),
            }
        )

    # Primary comparison is exactly task-matched on both sides. We apply the
    # paper's C formula to the MATH500+GSM8K columns used by our necessity curve.
    # The paper's published four-benchmark C_math remains a locked robustness
    # target, not the primary statistic.
    necessity = np.mean(
        np.vstack([necessity_by_task[t] for t in available_tasks]), axis=0
    )
    leverage = paper_leverage_for_matched_tasks(
        paper_base, paper_full, paper_layers, available_tasks
    )
    leverage_c_math = paper_layers["c_math"].to_numpy(float)

    # Diagnostics that stay descriptive: a layer can be "necessary" because its
    # bypass destroys broad generation behavior rather than mathematical
    # computation specifically. We never filter these layers post hoc; instead
    # we expose parser/truncation pathology and paired success transitions.
    fallback_by_layer, truncation_by_layer = generation_health_by_layer(
        results_dir, layer_ids
    )
    lost_by_layer = np.mean(
        np.vstack(
            [
                np.mean(
                    (base_by_task[t][None, :] == 1.0)
                    & (layer_by_task[t] == 0.0),
                    axis=1,
                )
                for t in available_tasks
            ]
        ),
        axis=0,
    )
    gained_by_layer = np.mean(
        np.vstack(
            [
                np.mean(
                    (base_by_task[t][None, :] == 0.0)
                    & (layer_by_task[t] == 1.0),
                    axis=1,
                )
                for t in available_tasks
            ]
        ),
        axis=0,
    )

    stats = relation_stats(necessity, leverage, topk=args.topk)
    c_math_stats = relation_stats(necessity, leverage_c_math, topk=args.topk)
    boot = bootstrap_rho(
        {t: base_by_task[t] for t in available_tasks},
        {t: layer_by_task[t] for t in available_tasks},
        leverage,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    low, high = ci(boot, level=0.90)
    gate = gate_label(
        stats.spearman_rho,
        low,
        high,
        stats.depth_residual_spearman,
    )

    rows = []
    for i, layer in enumerate(layer_ids):
        row = {
            "layer": int(layer),
            "necessity_mean": float(necessity[i]),
            "paper_c_matched_tasks": float(leverage[i]),
            "paper_c_math_4task": float(leverage_c_math[i]),
            "baseline_correct_to_wrong_rate": float(lost_by_layer[i]),
            "baseline_wrong_to_correct_rate": float(gained_by_layer[i]),
            "parser_fallback_rate": float(fallback_by_layer[i]),
            "truncation_rate": float(truncation_by_layer[i]),
        }
        for task in available_tasks:
            row[f"necessity_{task}"] = float(necessity_by_task[task][i])
            row[f"paper_leverage_{task}"] = float(leverage_by_task[task][i])
        rows.append(row)

    pd.DataFrame(rows).to_csv(results_dir / "layer_relation.csv", index=False)

    metrics = {
        "gate": gate,
        "tasks": available_tasks,
        "num_layers": len(layer_ids),
        "primary_target": "paper_C_on_exact_matched_tasks",
        "spearman_rho": stats.spearman_rho,
        "spearman_bootstrap_90ci": [low, high],
        "kendall_tau": stats.kendall_tau,
        "depth_residual_spearman": stats.depth_residual_spearman,
        "circular_shift_p": stats.circular_shift_p,
        "c_math_4task_robustness": {
            "spearman_rho": c_math_stats.spearman_rho,
            "kendall_tau": c_math_stats.kendall_tau,
            "depth_residual_spearman": c_math_stats.depth_residual_spearman,
            "circular_shift_p": c_math_stats.circular_shift_p,
            "topk_overlap": c_math_stats.topk_overlap,
        },
        "topk": args.topk,
        "topk_overlap": stats.topk_overlap,
        "topk_expected_under_random": stats.topk_expected,
        "task_matched": task_rows,
        "generation_damage_diagnostics": {
            "rho_necessity_vs_parser_fallback": safe_spearman(
                necessity, fallback_by_layer
            ),
            "rho_necessity_vs_truncation": safe_spearman(
                necessity, truncation_by_layer
            ),
            "max_parser_fallback_rate": float(np.max(fallback_by_layer)),
            "max_truncation_rate": float(np.max(truncation_by_layer)),
        },
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
    }
    (results_dir / "relation_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    make_figure(
        results_dir / "relation.png",
        layer_ids,
        necessity,
        leverage,
        task_rows,
    )

    report = f"""# Topic 12 G-0 report

**Gate:** `{gate}`

## Locked primary result

- Spearman rho(I, C_matched[MATH500+GSM8K]): **{stats.spearman_rho:.3f}**
- Paired item-bootstrap 90% CI: **[{low:.3f}, {high:.3f}]**
- Kendall tau: **{stats.kendall_tau:.3f}**
- rho after removing a quadratic depth trend from both curves: **{stats.depth_residual_spearman:.3f}**
- exact circular-shift p-value (depth-autocorrelation-preserving null): **{stats.circular_shift_p:.3f}**
- top-{args.topk} overlap: **{stats.topk_overlap}** (random expectation {stats.topk_expected:.2f})
- robustness rho against published four-task C_math: **{c_math_stats.spearman_rho:.3f}**
- robustness depth-residual rho against four-task C_math: **{c_math_stats.depth_residual_spearman:.3f}**
- rho(necessity, parser-fallback rate): **{safe_spearman(necessity, fallback_by_layer):.3f}**
- rho(necessity, truncation rate): **{safe_spearman(necessity, truncation_by_layer):.3f}**

The last two are diagnostics, not exclusion criteria. If necessity is almost
entirely tied to parser failure or runaway generation, describe the result as
**broad generation fragility**, not a reasoning-specific mechanism.

## Task-matched checks

"""
    for row in task_rows:
        report += (
            f"- {row['task']}: baseline acc={row['baseline_accuracy']:.3f}, "
            f"rho(I_task, C_task)={row['rho']:.3f}\n"
        )

    report += f"""
## Interpretation discipline

All external RL numbers come from Table 13 of *Is One Layer Enough?* for
Qwen3-1.7B-Base. The primary `C_matched` applies the paper's own contribution
formula to exactly the MATH500+GSM8K columns used by our necessity curve.
Published four-task `C_math` is reported as a locked robustness target. `I` is
measured here on the same base model with a frozen full-depth residual-block
bypass sweep. The primary statistic was not selected after seeing the curve.

The depth-residual statistic exists for one reason: a high raw rho can arise
because both quantities are merely broad middle-layer functions. If raw
alignment disappears after removing the predeclared quadratic depth trend, the
result is labeled `BROAD_DEPTH_ALIGNMENT_ONLY`, not a layer-specific law.

`INCONCLUSIVE_DO_NOT_TUNE` means stop. Do not rescue the topic by changing
layer subsets, task weights, correlation metrics, or ablation strength after
seeing this run. The only predeclared confirmation is the residual-scale=0.5
full sweep, and only after an interpretable G-0.
"""
    (results_dir / "REPORT.md").write_text(report, encoding="utf-8")

    print(report)


if __name__ == "__main__":
    main()
