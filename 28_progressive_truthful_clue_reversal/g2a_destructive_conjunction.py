#!/usr/bin/env python3
"""Frozen Topic 28 G2a destructive-conjunction screen."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import analyze_reversal_structure as analysis
import g0_progressive_reversal as g0
import g1_order_swap as g1


SEED = 20260825
BOOTSTRAP_REPS = 2000
EXPECTED_PANEL = 498
EXPECTED_QIDS = 415
EXPECTED_HASHES = {
    "panel.csv": "427deda33f45bb2a6c2d2caa1e64cead2b920b579412d0f7dbae1fe92f7b6f92",
    "state_outputs.csv": "ddf8eeb10e6dd4b69d1e1c09b2998e8bd6cb1a67cb1bf83f6ee2c348e4e4d11a",
    "paired_results.csv": "f3a05d65b07dc68977161fe77eb4593d9bdaf612bee4e575787a2e249750bc97",
}

MIN_VALID_FRACTION = 0.98
MIN_JOINT_SUPPORT = 100
MIN_EXACT_EVENTS = 10
MIN_EXACT_RATE = 0.03
MIN_EXACT_CI_LOW = 0.01
MIN_CLEAR_EVENTS = 5
MIN_CLEAR_QIDS = 5
MIN_CLEAR_RATE = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1-dir", type=Path, default=Path("artifacts/g1"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/g2a"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=None,
        help="Engineering only; debug subsets cannot receive a scientific verdict.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate frozen G1 inputs and write no model outputs.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_aliases(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(x) for x in value)
    loaded = json.loads(str(value))
    if not isinstance(loaded, list):
        raise ValueError("aliases must decode to a list")
    return tuple(str(x) for x in loaded)


def validate_and_load_g1(g1_dir: Path) -> tuple[pd.DataFrame, dict]:
    observed = {}
    for name, expected in EXPECTED_HASHES.items():
        path = g1_dir / name
        if not path.is_file():
            raise RuntimeError(f"Missing frozen G1 artifact: {path}")
        observed[name] = sha256(path)
        if observed[name] != expected:
            raise RuntimeError(
                f"Frozen G1 hash mismatch for {name}: {observed[name]} != {expected}"
            )

    receipt_path = g1_dir / "g1_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    required_receipt = {
        "question_revision": g1.QUESTION_REVISION,
        "model_id": g1.MODEL_ID,
        "model_revision": g1.MODEL_REVISION,
        "seed": g1.SEED,
        "max_new_tokens": g1.MAX_NEW_TOKENS,
        "decoding": "greedy",
        "debug": False,
    }
    for key, expected in required_receipt.items():
        if receipt.get(key) != expected:
            raise RuntimeError(
                f"Frozen G1 receipt mismatch for {key}: {receipt.get(key)!r} != {expected!r}"
            )

    paired = pd.read_csv(g1_dir / "paired_results.csv")
    if len(paired) != EXPECTED_PANEL or paired.qid.nunique() != EXPECTED_QIDS:
        raise RuntimeError(
            f"Frozen G1 panel mismatch: {len(paired)} rows, {paired.qid.nunique()} qids"
        )
    if paired.boundary_id.duplicated().any():
        raise RuntimeError("Frozen G1 paired results contain duplicate boundary IDs")
    if not paired[["o1_valid", "o2_valid"]].all(axis=None):
        raise RuntimeError("Frozen G1 P or P+C output is invalid")
    paired["aliases"] = paired["aliases"].map(parse_aliases)
    return paired, {"hashes": observed, "g1_receipt": receipt}


def singular_stem(token: str) -> str:
    token = token.lower()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def prediction_alias_related(prediction: object, aliases: Iterable[str]) -> bool:
    pred_norm = g0.normalize_answer(prediction)
    if not pred_norm:
        return False
    pred_tokens = analysis.content_tokens(pred_norm)
    pred_set = set(pred_tokens)
    for alias in aliases:
        alias_norm = g0.normalize_answer(alias)
        if not alias_norm:
            continue
        if pred_norm == alias_norm or pred_norm in alias_norm or alias_norm in pred_norm:
            return True
        alias_tokens = analysis.content_tokens(alias_norm)
        if pred_set & set(alias_tokens):
            return True
        for left in pred_tokens:
            for right in alias_tokens:
                if singular_stem(left) == singular_stem(right):
                    return True
                if difflib.SequenceMatcher(None, left, right).ratio() >= 0.85:
                    return True
    return False


def build_c_states(paired: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "boundary_id": paired.boundary_id,
            "qid": paired.qid,
            "state": "c",
            "user_prompt": paired.trigger_clue.map(lambda clue: g1.build_user_prompt([clue])),
            "aliases": paired.aliases,
        }
    )


def combine_results(paired: pd.DataFrame, c_states: pd.DataFrame) -> pd.DataFrame:
    if c_states.boundary_id.duplicated().any():
        raise RuntimeError("C-alone outputs contain duplicate boundary IDs")
    c = c_states.set_index("boundary_id")
    out = paired.copy().set_index("boundary_id")
    for column in (
        "prediction",
        "prediction_norm",
        "correct",
        "valid",
        "input_tokens",
        "output_tokens",
        "single_line",
    ):
        out[f"c_{column}"] = c[column]
    out["jointly_sufficient"] = out.o1_correct & out.c_correct
    out["destructive_exact"] = out.jointly_sufficient & ~out.o2_correct
    out["pc_alias_related"] = [
        prediction_alias_related(pred, aliases)
        for pred, aliases in zip(out.o2_prediction, out.aliases)
    ]
    out["combined_clear_wrong"] = ~out.o2_correct & ~out.pc_alias_related
    out["destructive_clear"] = out.destructive_exact & out.combined_clear_wrong
    out["p_equals_c"] = out.o1_prediction_norm == out.c_prediction_norm
    out["c_equals_pc"] = out.c_prediction_norm == out.o2_prediction_norm
    out["three_state_pattern"] = [
        "".join("1" if bool(x) else "0" for x in values)
        for values in zip(out.o1_correct, out.c_correct, out.o2_correct)
    ]
    return out.reset_index()


def cluster_bootstrap_conditional_rate(
    df: pd.DataFrame,
    numerator: str,
    denominator: str,
    reps: int = BOOTSTRAP_REPS,
    seed: int = SEED,
) -> tuple[float, float, float]:
    work = df[["qid", numerator, denominator]].copy()
    by_qid = work.groupby("qid")[[numerator, denominator]].sum()
    numer = by_qid[numerator].to_numpy(float)
    denom = by_qid[denominator].to_numpy(float)
    observed_denom = float(denom.sum())
    observed = float(numer.sum() / observed_denom) if observed_denom else float("nan")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(len(by_qid), size=(reps, len(by_qid)), replace=True)
    sampled_numer = numer[sampled].sum(axis=1)
    sampled_denom = denom[sampled].sum(axis=1)
    valid = sampled_denom > 0
    values = sampled_numer[valid] / sampled_denom[valid]
    lo, hi = np.quantile(values, [0.025, 0.975])
    return observed, float(lo), float(hi)


def summarize(results: pd.DataFrame, c_states: pd.DataFrame, debug: bool) -> dict:
    exact_rate, exact_lo, exact_hi = cluster_bootstrap_conditional_rate(
        results, "destructive_exact", "jointly_sufficient"
    )
    clear_rate, clear_lo, clear_hi = cluster_bootstrap_conditional_rate(
        results,
        "destructive_clear",
        "jointly_sufficient",
        seed=SEED + 1,
    )
    artifact_gates = {
        "panel_boundaries_exact": len(results) == EXPECTED_PANEL,
        "unique_qids_exact": int(results.qid.nunique()) == EXPECTED_QIDS,
        "c_valid_fraction": float(results.c_valid.mean()) >= MIN_VALID_FRACTION,
        "jointly_sufficient_support": int(results.jointly_sufficient.sum()) >= MIN_JOINT_SUPPORT,
    }
    scientific_gates = {
        "exact_event_support": int(results.destructive_exact.sum()) >= MIN_EXACT_EVENTS,
        "exact_rate": exact_rate >= MIN_EXACT_RATE,
        "exact_ci_low": exact_lo > MIN_EXACT_CI_LOW,
        "clear_event_support": int(results.destructive_clear.sum()) >= MIN_CLEAR_EVENTS,
        "clear_qid_support": int(results.loc[results.destructive_clear, "qid"].nunique()) >= MIN_CLEAR_QIDS,
        "clear_rate": clear_rate >= MIN_CLEAR_RATE,
    }
    if debug:
        verdict = "DEBUG_NO_VERDICT"
    elif not all(artifact_gates.values()):
        verdict = "STOP_G2A_MEASUREMENT"
    elif all(scientific_gates.values()):
        verdict = "GO_DESTRUCTIVE_CONJUNCTION_OBJECT"
    else:
        verdict = "STOP_DESTRUCTIVE_CONJUNCTION"

    return {
        "verdict": verdict,
        "panel_boundaries": int(len(results)),
        "unique_qids": int(results.qid.nunique()),
        "c_outputs": int(len(c_states)),
        "c_valid": int(results.c_valid.sum()),
        "c_valid_fraction": float(results.c_valid.mean()),
        "c_single_line_fraction": float(results.c_single_line.mean()),
        "p_correct": int(results.o1_correct.sum()),
        "c_correct": int(results.c_correct.sum()),
        "pc_correct": int(results.o2_correct.sum()),
        "jointly_sufficient": int(results.jointly_sufficient.sum()),
        "destructive_exact": int(results.destructive_exact.sum()),
        "destructive_exact_qids": int(results.loc[results.destructive_exact, "qid"].nunique()),
        "destructive_exact_rate": exact_rate,
        "destructive_exact_ci95": [exact_lo, exact_hi],
        "destructive_clear": int(results.destructive_clear.sum()),
        "destructive_clear_qids": int(results.loc[results.destructive_clear, "qid"].nunique()),
        "destructive_clear_rate": clear_rate,
        "destructive_clear_ci95": [clear_lo, clear_hi],
        "p_equals_c": int(results.p_equals_c.sum()),
        "c_equals_pc": int(results.c_equals_pc.sum()),
        "artifact_gates": artifact_gates,
        "scientific_gates": scientific_gates,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paired, input_receipt = validate_and_load_g1(args.g1_dir)
    full_run = args.debug_limit is None
    if args.debug_limit is not None:
        paired = paired.head(args.debug_limit).copy()

    preflight = {
        **input_receipt,
        "expected_panel": EXPECTED_PANEL,
        "expected_qids": EXPECTED_QIDS,
        "loaded_boundaries": int(len(paired)),
        "loaded_qids": int(paired.qid.nunique()),
        "debug": not full_run,
        "preflight_only": bool(args.preflight_only),
    }
    if args.preflight_only:
        (args.out_dir / "g2a_receipt.json").write_text(json.dumps(preflight, indent=2) + "\n")
        print(json.dumps(preflight, indent=2))
        return

    c_states = build_c_states(paired)
    c_states = g1.run_inference(c_states, args.batch_size, args.device)
    results = combine_results(paired, c_states)
    summary = summarize(results, c_states, debug=not full_run)

    c_write = c_states.copy()
    c_write["aliases"] = c_write.aliases.map(json.dumps)
    c_write.to_csv(args.out_dir / "c_alone_outputs.csv", index=False)
    result_write = results.copy()
    result_write["aliases"] = result_write.aliases.map(json.dumps)
    result_write.to_csv(args.out_dir / "g2a_cases.csv", index=False)
    results.three_state_pattern.value_counts().rename_axis(
        "three_state_pattern"
    ).reset_index(name="n").sort_values("three_state_pattern").to_csv(
        args.out_dir / "three_state_cells.csv", index=False
    )
    for column in ("specificity_half", "category", "t"):
        grouped = (
            results.groupby(column, dropna=False)
            .agg(
                n=("boundary_id", "size"),
                jointly_sufficient=("jointly_sufficient", "sum"),
                destructive_exact=("destructive_exact", "sum"),
                destructive_clear=("destructive_clear", "sum"),
            )
            .reset_index()
        )
        grouped.to_csv(args.out_dir / f"summary_by_{column}.csv", index=False)

    receipt = {
        **preflight,
        "preflight_only": False,
        "model_id": g1.MODEL_ID,
        "model_revision": g1.MODEL_REVISION,
        "seed": SEED,
        "decoding": "greedy",
        "max_new_tokens": g1.MAX_NEW_TOKENS,
        "torch": str(c_states.torch_version.iloc[0]),
        "transformers": str(c_states.transformers_version.iloc[0]),
        "device": args.device,
    }
    (args.out_dir / "g2a_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (args.out_dir / "g2a_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
