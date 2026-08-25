#!/usr/bin/env python3
"""Artifact-only preflight for cross-model shared truthful-clue traps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SEED = 20260825
NULL_REPS = 1000
INPUT_SHA256 = "9b001242576a1ca7d15411e7872a9725fef3a1cd740e8eb9b082dd3006ccbc7e"
EXPECTED_ROWS = 120353
EXPECTED_REVERSALS = 8102
EXPECTED_CONFIGS = 93
EXPECTED_FAMILIES = 19
EXPECTED_BOUNDARIES = 2241
EXPECTED_QIDS = 782
MIN_WELL_SUPPORTED_BOUNDARIES = 2000

MIN_RISK_CONFIGS = 20
MIN_RISK_FAMILIES = 8
MIN_BOUNDARY_REVERSALS = 8
MIN_BOUNDARY_HAZARD = 0.20
MIN_FAMILY_HITS = 4
FAMILY_HIT_RATE = 0.25
MIN_TOP_WRONG_COUNT = 5
MIN_TOP_WRONG_SHARE = 0.50

MIN_FAMILY_OVERLAP_RATIO = 1.25
MAX_PERMUTATION_P = 0.001
MIN_TRAPS = 20
MIN_TRAP_QIDS = 20
MIN_TRAP_CATEGORIES = 5
MIN_TRAP_NULL_RATIO = 3.0
MIN_CONSENSUS_EVENTS = 100


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_input = (
        repo_root
        / "28_progressive_truthful_clue_reversal"
        / "artifacts"
        / "analysis1"
        / "eligible_transition_features.csv"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("artifacts/shared_truthful_clue_traps")
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--debug-permutations",
        type=int,
        default=None,
        help="Engineering timing only; reduced null runs cannot receive a verdict.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_family(config: str) -> str:
    s = config.lower()
    rules = [
        (s.startswith("rag-bm25"), "rag_bm25"),
        (s.startswith("rag-contriever"), "rag_contriever"),
        (s.startswith("rag-grit"), "rag_grit"),
        (s.startswith("bm25_"), "retrieval_bm25"),
        (s.startswith("contriever_"), "retrieval_contriever"),
        (s.startswith("grit_"), "retrieval_grit"),
        (s.startswith(("t0", "flan-", "ul2_")), "t5_t0_ul2"),
        (s.startswith("falcon"), "falcon"),
        (s.startswith("gemma-"), "gemma"),
        (s.startswith("google-gemini"), "gemini"),
        (s.startswith("cohere-"), "cohere"),
        (s.startswith("gpt-neo"), "gpt_neo"),
        (s.startswith(("llama-", "meta-llama")), "llama"),
        (s.startswith(("mistral", "mixtral")), "mistral_mixtral"),
        (s.startswith("openai-"), "openai"),
        (s.startswith("opt-"), "opt"),
        (s.startswith("phi-"), "phi"),
        (s.startswith("pythia-"), "pythia"),
        (s.startswith("vicuna-"), "vicuna"),
    ]
    matches = [family for matched, family in rules if matched]
    if len(matches) != 1:
        raise ValueError(f"Config must match exactly one frozen family rule: {config}")
    return matches[0]


def load_input(path: Path) -> tuple[pd.DataFrame, dict]:
    observed_sha = sha256(path)
    if observed_sha != INPUT_SHA256:
        raise RuntimeError(f"Input SHA mismatch: {observed_sha} != {INPUT_SHA256}")
    columns = [
        "config",
        "qid",
        "prev_clue_idx",
        "clue_idx",
        "category",
        "subcategory",
        "reversal",
        "prediction_norm",
        "strict_alias_correct",
        "new_clue_text",
        "new_clue_specificity",
    ]
    df = pd.read_csv(path, usecols=columns)
    df["config"] = df.config.astype(str)
    df["qid"] = df.qid.astype(str)
    df["category"] = df.category.fillna("UNKNOWN").astype(str)
    df["reversal"] = df.reversal.astype(bool)
    df["strict_alias_correct"] = df.strict_alias_correct.astype(bool)
    df["prediction_norm"] = df.prediction_norm.fillna("").astype(str).str.strip()
    df["family"] = df.config.map(config_family)
    df["boundary_id"] = (
        df.qid
        + "_"
        + df.prev_clue_idx.astype(int).astype(str)
        + "_"
        + df.clue_idx.astype(int).astype(str)
    )
    if df.duplicated(["config", "boundary_id"]).any():
        raise RuntimeError("Duplicate config-boundary cells in frozen input")

    boundary_consistency = df.groupby("boundary_id").agg(
        qids=("qid", "nunique"),
        from_indices=("prev_clue_idx", "nunique"),
        to_indices=("clue_idx", "nunique"),
        categories=("category", "nunique"),
        clues=("new_clue_text", "nunique"),
        specificities=("new_clue_specificity", "nunique"),
    )
    if (boundary_consistency > 1).any(axis=None):
        raise RuntimeError("Boundary metadata is inconsistent across configs")

    receipt = {
        "input": str(path.resolve()),
        "input_sha256": observed_sha,
        "rows": int(len(df)),
        "reversals": int(df.reversal.sum()),
        "configs": int(df.config.nunique()),
        "families": int(df.family.nunique()),
        "boundaries": int(df.boundary_id.nunique()),
        "qids": int(df.qid.nunique()),
    }
    return df, receipt


def build_family_inventory(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["config", "family"]]
        .drop_duplicates()
        .sort_values(["family", "config"])
        .reset_index(drop=True)
    )


def encode_analysis(df: pd.DataFrame) -> dict:
    work = df.copy()
    boundaries = sorted(work.boundary_id.unique())
    families = sorted(work.family.unique())
    boundary_pos = {value: i for i, value in enumerate(boundaries)}
    family_pos = {value: i for i, value in enumerate(families)}
    work["b"] = work.boundary_id.map(boundary_pos).astype(int)
    work["f"] = work.family.map(family_pos).astype(int)

    strict_wrong = work.reversal & ~work.strict_alias_correct & work.prediction_norm.ne("")
    wrong_values = sorted(work.loc[strict_wrong, "prediction_norm"].unique())
    wrong_pos = {value: i for i, value in enumerate(wrong_values)}
    wrong_code = np.full(len(work), -1, dtype=np.int32)
    positions = np.flatnonzero(strict_wrong.to_numpy())
    wrong_code[positions] = work.loc[strict_wrong, "prediction_norm"].map(wrong_pos).to_numpy(np.int32)

    b = work.b.to_numpy(np.int32)
    f = work.f.to_numpy(np.int16)
    bf = b.astype(np.int64) * len(families) + f.astype(np.int64)
    reversal = work.reversal.to_numpy(bool)
    risk_bf = np.bincount(bf, minlength=len(boundaries) * len(families)).reshape(
        len(boundaries), len(families)
    )
    risk_b = risk_bf.sum(axis=1)
    risk_families = (risk_bf > 0).sum(axis=1)
    pair_opportunities = float(np.sum(risk_families * (risk_families - 1) / 2))
    config_pair_opportunities = float(np.sum(risk_b * (risk_b - 1) / 2))

    strata = work.groupby(["config", "category", "clue_idx"], sort=True).indices
    stratum_indices = [np.asarray(index, dtype=np.int32) for index in strata.values()]

    boundary_meta = (
        work.sort_values(["boundary_id", "config"])
        .drop_duplicates("boundary_id")
        [[
            "boundary_id",
            "qid",
            "prev_clue_idx",
            "clue_idx",
            "category",
            "subcategory",
            "new_clue_text",
            "new_clue_specificity",
        ]]
        .set_index("boundary_id")
        .loc[boundaries]
        .reset_index()
    )
    boundary_meta["specificity_quartile"] = pd.qcut(
        boundary_meta.new_clue_specificity,
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="raise",
    ).astype(str)

    return {
        "work": work,
        "boundaries": boundaries,
        "families": families,
        "wrong_values": wrong_values,
        "b": b,
        "f": f,
        "bf": bf,
        "reversal": reversal,
        "wrong_code": wrong_code,
        "risk_bf": risk_bf,
        "risk_b": risk_b,
        "risk_families": risk_families,
        "pair_opportunities": pair_opportunities,
        "config_pair_opportunities": config_pair_opportunities,
        "stratum_indices": stratum_indices,
        "boundary_meta": boundary_meta,
    }


def compute_metrics(encoded: dict, reversal: np.ndarray, wrong_code: np.ndarray) -> dict:
    n_boundaries = len(encoded["boundaries"])
    n_families = len(encoded["families"])
    flip_bf = np.bincount(
        encoded["bf"][reversal], minlength=n_boundaries * n_families
    ).reshape(n_boundaries, n_families)
    risk_bf = encoded["risk_bf"]
    family_rates = np.divide(
        flip_bf,
        risk_bf,
        out=np.zeros_like(flip_bf, dtype=float),
        where=risk_bf > 0,
    )
    family_pair_sum = np.sum(
        (family_rates.sum(axis=1) ** 2 - (family_rates**2).sum(axis=1)) / 2
    )
    family_overlap = float(family_pair_sum / encoded["pair_opportunities"])

    flips_b = flip_bf.sum(axis=1)
    raw_pair_sum = np.sum(flips_b * (flips_b - 1) / 2)
    config_overlap = float(raw_pair_sum / encoded["config_pair_opportunities"])
    hazard = np.divide(
        flips_b,
        encoded["risk_b"],
        out=np.zeros(n_boundaries, dtype=float),
        where=encoded["risk_b"] > 0,
    )
    family_hits = (family_rates >= FAMILY_HIT_RATE).sum(axis=1)

    top_wrong_count = np.zeros(n_boundaries, dtype=np.int32)
    top_wrong_code = np.full(n_boundaries, -1, dtype=np.int32)
    consensus_rows = reversal & (wrong_code >= 0)
    if consensus_rows.any():
        n_wrong = len(encoded["wrong_values"])
        pairs = encoded["b"][consensus_rows].astype(np.int64) * n_wrong + wrong_code[consensus_rows]
        unique_pairs, counts = np.unique(pairs, return_counts=True)
        pair_boundaries = (unique_pairs // n_wrong).astype(np.int32)
        pair_wrong = (unique_pairs % n_wrong).astype(np.int32)
        order = np.lexsort((pair_wrong, -counts, pair_boundaries))
        sorted_boundaries = pair_boundaries[order]
        first = np.concatenate(([True], sorted_boundaries[1:] != sorted_boundaries[:-1]))
        chosen = order[first]
        top_wrong_count[pair_boundaries[chosen]] = counts[chosen]
        top_wrong_code[pair_boundaries[chosen]] = pair_wrong[chosen]
    top_wrong_share = np.divide(
        top_wrong_count,
        flips_b,
        out=np.zeros(n_boundaries, dtype=float),
        where=flips_b > 0,
    )

    trap = (
        (encoded["risk_b"] >= MIN_RISK_CONFIGS)
        & (encoded["risk_families"] >= MIN_RISK_FAMILIES)
        & (flips_b >= MIN_BOUNDARY_REVERSALS)
        & (hazard >= MIN_BOUNDARY_HAZARD)
        & (family_hits >= MIN_FAMILY_HITS)
        & (top_wrong_count >= MIN_TOP_WRONG_COUNT)
        & (top_wrong_share >= MIN_TOP_WRONG_SHARE)
    )
    return {
        "family_overlap": family_overlap,
        "config_overlap": config_overlap,
        "flip_bf": flip_bf,
        "flips_b": flips_b,
        "hazard": hazard,
        "family_hits": family_hits,
        "top_wrong_count": top_wrong_count,
        "top_wrong_code": top_wrong_code,
        "top_wrong_share": top_wrong_share,
        "trap": trap,
        "trap_count": int(trap.sum()),
        "consensus_events": int(top_wrong_count[trap].sum()),
    }


def permute_payload(
    reversal: np.ndarray,
    wrong_code: np.ndarray,
    strata: Iterable[np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    perm_reversal = reversal.copy()
    perm_wrong = wrong_code.copy()
    for index in strata:
        if len(index) <= 1:
            continue
        order = rng.permutation(len(index))
        source = index[order]
        perm_reversal[index] = reversal[source]
        perm_wrong[index] = wrong_code[source]
    return perm_reversal, perm_wrong


def plus_one_p(null: np.ndarray, observed: float) -> float:
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def safe_ratio(observed: float, expected: float) -> float:
    if expected == 0:
        return float("inf") if observed > 0 else 1.0
    return float(observed / expected)


def artifact_gates(receipt: dict, well_supported: int) -> dict:
    return {
        "input_sha": receipt["input_sha256"] == INPUT_SHA256,
        "rows": receipt["rows"] == EXPECTED_ROWS,
        "reversals": receipt["reversals"] == EXPECTED_REVERSALS,
        "configs": receipt["configs"] == EXPECTED_CONFIGS,
        "families": receipt["families"] == EXPECTED_FAMILIES,
        "boundaries": receipt["boundaries"] == EXPECTED_BOUNDARIES,
        "qids": receipt["qids"] == EXPECTED_QIDS,
        "well_supported_boundaries": well_supported >= MIN_WELL_SUPPORTED_BOUNDARIES,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df, receipt = load_input(args.input)
    inventory = build_family_inventory(df)
    encoded = encode_analysis(df)
    well_supported = int(
        (
            (encoded["risk_b"] >= MIN_RISK_CONFIGS)
            & (encoded["risk_families"] >= MIN_RISK_FAMILIES)
        ).sum()
    )
    a_gates = artifact_gates(receipt, well_supported)
    preflight_receipt = {
        **receipt,
        "seed": SEED,
        "null_reps": NULL_REPS,
        "strata": "config x category x to_clue_idx",
        "n_strata": len(encoded["stratum_indices"]),
        "well_supported_boundaries": well_supported,
        "artifact_gates": a_gates,
        "preflight_only": bool(args.preflight_only),
    }
    inventory.to_csv(args.out_dir / "family_inventory.csv", index=False)
    if args.preflight_only:
        preflight_receipt["verdict"] = "PREFLIGHT_ONLY_NO_SCIENTIFIC_OUTPUT"
        (args.out_dir / "receipt.json").write_text(
            json.dumps(preflight_receipt, indent=2) + "\n"
        )
        print(json.dumps(preflight_receipt, indent=2))
        return

    debug = args.debug_permutations is not None
    reps = int(args.debug_permutations) if debug else NULL_REPS
    if reps <= 0:
        raise ValueError("Permutation count must be positive")

    observed = compute_metrics(encoded, encoded["reversal"], encoded["wrong_code"])
    rng = np.random.default_rng(SEED)
    null_rows = []
    for rep in range(reps):
        perm_reversal, perm_wrong = permute_payload(
            encoded["reversal"],
            encoded["wrong_code"],
            encoded["stratum_indices"],
            rng,
        )
        metrics = compute_metrics(encoded, perm_reversal, perm_wrong)
        null_rows.append(
            {
                "rep": rep,
                "family_overlap": metrics["family_overlap"],
                "config_overlap": metrics["config_overlap"],
                "trap_count": metrics["trap_count"],
                "consensus_events": metrics["consensus_events"],
            }
        )
        if (rep + 1) % 25 == 0 or rep + 1 == reps:
            print(f"[null] {rep + 1}/{reps}", flush=True)
    null = pd.DataFrame(null_rows)

    family_null_mean = float(null.family_overlap.mean())
    family_null_sd = float(null.family_overlap.std(ddof=1))
    family_ratio = safe_ratio(observed["family_overlap"], family_null_mean)
    family_z = (
        float((observed["family_overlap"] - family_null_mean) / family_null_sd)
        if family_null_sd > 0
        else float("inf")
    )
    family_p = plus_one_p(null.family_overlap.to_numpy(), observed["family_overlap"])
    trap_null_mean = float(null.trap_count.mean())
    trap_ratio = safe_ratio(observed["trap_count"], trap_null_mean)
    trap_p = plus_one_p(null.trap_count.to_numpy(), observed["trap_count"])

    boundary = encoded["boundary_meta"].copy()
    boundary["risk_configs"] = encoded["risk_b"]
    boundary["risk_families"] = encoded["risk_families"]
    boundary["reversals"] = observed["flips_b"]
    boundary["hazard"] = observed["hazard"]
    boundary["family_hits"] = observed["family_hits"]
    boundary["top_wrong_count"] = observed["top_wrong_count"]
    boundary["top_wrong_share"] = observed["top_wrong_share"]
    boundary["top_wrong_prediction"] = [
        encoded["wrong_values"][code] if code >= 0 else ""
        for code in observed["top_wrong_code"]
    ]
    boundary["shared_trap"] = observed["trap"]

    trap_table = boundary.loc[boundary.shared_trap].sort_values(
        ["top_wrong_count", "reversals", "hazard"], ascending=False
    )
    trap_qids = int(trap_table.qid.nunique())
    trap_categories = int(trap_table.category.nunique())

    specificity = (
        boundary.groupby("specificity_quartile", observed=False)
        .agg(
            boundaries=("boundary_id", "size"),
            mean_hazard=("hazard", "mean"),
            mean_family_hits=("family_hits", "mean"),
            traps=("shared_trap", "sum"),
            consensus_events=(
                "top_wrong_count",
                lambda x: int(x[boundary.loc[x.index, "shared_trap"]].sum()),
            ),
        )
        .reset_index()
    )
    specificity["trap_rate"] = specificity.traps / specificity.boundaries

    s_gates = {
        "family_overlap_ratio": family_ratio >= MIN_FAMILY_OVERLAP_RATIO,
        "family_overlap_permutation_p": family_p <= MAX_PERMUTATION_P,
        "trap_boundaries": observed["trap_count"] >= MIN_TRAPS,
        "trap_qids": trap_qids >= MIN_TRAP_QIDS,
        "trap_categories": trap_categories >= MIN_TRAP_CATEGORIES,
        "trap_permutation_p": trap_p <= MAX_PERMUTATION_P,
        "trap_null_ratio": trap_ratio >= MIN_TRAP_NULL_RATIO,
        "consensus_events": observed["consensus_events"] >= MIN_CONSENSUS_EVENTS,
    }
    if debug:
        verdict = "DEBUG_NO_VERDICT"
    elif not all(a_gates.values()):
        verdict = "STOP_SHARED_TRAP_ARTIFACT"
    elif all(s_gates.values()):
        verdict = "GO_SHARED_TRUTHFUL_CLUE_TRAPS"
    else:
        verdict = "STOP_SHARED_TRAP_ROUTE"

    summary = {
        "verdict": verdict,
        "null_reps": reps,
        "family_overlap_observed": observed["family_overlap"],
        "family_overlap_null_mean": family_null_mean,
        "family_overlap_null_sd": family_null_sd,
        "family_overlap_ratio": family_ratio,
        "family_overlap_z": family_z,
        "family_overlap_permutation_p": family_p,
        "config_overlap_observed": observed["config_overlap"],
        "config_overlap_null_mean": float(null.config_overlap.mean()),
        "shared_traps": observed["trap_count"],
        "shared_trap_qids": trap_qids,
        "shared_trap_categories": trap_categories,
        "shared_trap_null_mean": trap_null_mean,
        "shared_trap_null_q99": float(null.trap_count.quantile(0.99)),
        "shared_trap_null_ratio": trap_ratio,
        "shared_trap_permutation_p": trap_p,
        "consensus_events_in_traps": observed["consensus_events"],
        "artifact_gates": a_gates,
        "scientific_gates": s_gates,
    }
    receipt_out = {
        **preflight_receipt,
        "preflight_only": False,
        "debug": debug,
        "actual_null_reps": reps,
    }
    boundary.to_csv(args.out_dir / "boundary_observed.csv", index=False)
    trap_table.to_csv(args.out_dir / "shared_trap_boundaries.csv", index=False)
    null.to_csv(args.out_dir / "null_distribution.csv", index=False)
    specificity.to_csv(args.out_dir / "summary_by_specificity.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out_dir / "receipt.json").write_text(json.dumps(receipt_out, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
