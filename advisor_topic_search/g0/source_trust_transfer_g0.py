#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_UPSTREAM_COMMIT = "87dd466f10a76ea1cadc21a552d423d2d60c0cce"
FROZEN_MODEL = "google/gemma-3-4b-it"
FROZEN_SEED = 20260824
FROZEN_N_ITEMS = 128
EXPECTED_SYSTEM_MESSAGE = (
    "The following are multiple choice questions. You are provided tabular data from sources. "
    "Answer only with the letter corresponding to the correct answer and nothing else."
)


def norm_values(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        v = str(value).strip()
        key = v.casefold()
        if not v or key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def field_candidates(entity: Any, require_conflict: bool) -> list[str]:
    fields: list[str] = []
    for field, values in entity.attributes.items():
        vals = norm_values(values)
        if not vals:
            continue
        # Structural filter fixed before outcomes: avoid pathological long cells.
        if max(len(v) for v in vals[:2]) > 160:
            continue
        if require_conflict and len(vals) < 2:
            continue
        fields.append(field)
    return fields


def git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def bootstrap_ci(values: list[float], seed: int, b: int = 10_000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    means = np.empty(b, dtype=float)
    for i in range(b):
        means[i] = rng.choice(arr, size=arr.size, replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def history_entries(repeated_source: str, reverse_order: bool) -> list[tuple[str, str]]:
    """Return source/claim assignments with exact matched global marginals.

    In both conditions:
      * S1 appears exactly twice;
      * S2 appears exactly twice;
      * H1 appears exactly three times;
      * H2 appears exactly once.

    Only the source associated with the duplicated claim changes.
    """
    if repeated_source == "S1":
        entries = [("S1", "H1"), ("S2", "H1"), ("S1", "H1"), ("S2", "H2")]
    elif repeated_source == "S2":
        entries = [("S1", "H1"), ("S2", "H1"), ("S1", "H2"), ("S2", "H1")]
    else:
        raise ValueError(repeated_source)
    return list(reversed(entries)) if reverse_order else entries


def history_report(
    source: Any,
    entity: Any,
    field: str,
    value: str,
    report_idx: int,
    format_source_info,
    to_markdown_table,
) -> str:
    src = format_source_info(source, show_circulation=False)
    field_display = field.replace("_", " ").title()
    table = {"Name": entity.name, field_display: str(value)}
    return f"Prior Report {report_idx} (Source: {src}):\n{to_markdown_table(table)}"


def build_prompt(
    *,
    repeated_source: str,
    reverse_history: bool,
    swap_target_tables: bool,
    swap_answer_order: bool,
    source1: Any,
    source2: Any,
    hist1_entity: Any,
    hist1_field: str,
    hist1_value: str,
    hist2_entity: Any,
    hist2_field: str,
    hist2_value: str,
    target_entity: Any,
    target_field: str,
    target_value_s1: str,
    target_value_s2: str,
    question: str,
    format_source_info,
    to_markdown_table,
    create_entity_table,
    format_answer_choices,
) -> tuple[str, int]:
    source_map = {"S1": source1, "S2": source2}
    claim_map = {
        "H1": (hist1_entity, hist1_field, hist1_value),
        "H2": (hist2_entity, hist2_field, hist2_value),
    }

    history_parts: list[str] = []
    for idx, (source_key, claim_key) in enumerate(
        history_entries(repeated_source, reverse_history), start=1
    ):
        entity, field, value = claim_map[claim_key]
        history_parts.append(
            history_report(
                source_map[source_key], entity, field, value, idx,
                format_source_info, to_markdown_table,
            )
        )

    target_s1 = create_entity_table(target_entity, target_field, target_value_s1)
    target_s2 = create_entity_table(target_entity, target_field, target_value_s2)
    s1_display = format_source_info(source1, show_circulation=False)
    s2_display = format_source_info(source2, show_circulation=False)

    if not swap_target_tables:
        table_a = f"Current Table A (Source: {s1_display}):\n{to_markdown_table(target_s1)}"
        table_b = f"Current Table B (Source: {s2_display}):\n{to_markdown_table(target_s2)}"
    else:
        table_a = f"Current Table A (Source: {s2_display}):\n{to_markdown_table(target_s2)}"
        table_b = f"Current Table B (Source: {s1_display}):\n{to_markdown_table(target_s1)}"

    if not swap_answer_order:
        answer_block = format_answer_choices(target_value_s1, target_value_s2)
        s1_answer_index = 0
    else:
        answer_block = format_answer_choices(target_value_s2, target_value_s1)
        s1_answer_index = 1

    prompt = (
        "You previously saw the following reports. They concern other facts, not the current question.\n\n"
        + "\n\n".join(history_parts)
        + "\n\nCurrent reports:\n"
        + table_a
        + "\n\n"
        + table_b
        + "\n\n"
        + question
        + "\n"
        + answer_block
    )
    return prompt, s1_answer_index


def make_items(entities: list[Any], n_items: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[Any]] = defaultdict(list)
    for entity in entities:
        if field_candidates(entity, require_conflict=False):
            groups[entity.entity_class.value].append(entity)

    targets = [
        entity
        for entity in entities
        if field_candidates(entity, require_conflict=True)
        and len(groups[entity.entity_class.value]) >= 3
    ]
    rng.shuffle(targets)

    items: list[dict[str, Any]] = []
    for target in targets:
        same_class = [e for e in groups[target.entity_class.value] if e.id != target.id]
        if len(same_class) < 2:
            continue
        hist1, hist2 = rng.sample(same_class, 2)

        target_field = rng.choice(field_candidates(target, require_conflict=True))
        hist1_field = rng.choice(field_candidates(hist1, require_conflict=False))
        hist2_field = rng.choice(field_candidates(hist2, require_conflict=False))

        target_vals = norm_values(target.attributes[target_field])
        hist1_vals = norm_values(hist1.attributes[hist1_field])
        hist2_vals = norm_values(hist2.attributes[hist2_field])
        if len(target_vals) < 2 or not hist1_vals or not hist2_vals:
            continue

        items.append(
            {
                "target_entity": target,
                "target_field": target_field,
                "target_value_s1": target_vals[0],
                "target_value_s2": target_vals[1],
                "hist1_entity": hist1,
                "hist1_field": hist1_field,
                "hist1_value": hist1_vals[0],
                "hist2_entity": hist2,
                "hist2_field": hist2_field,
                "hist2_value": hist2_vals[0],
            }
        )
        if len(items) == n_items:
            break

    if len(items) != n_items:
        raise RuntimeError(f"Only {len(items)} structurally eligible items found; need {n_items}")
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--upstream-repo", required=True,
        help="Path to the locked JaSchuste/llm-source-preference checkout",
    )
    ap.add_argument(
        "--data-dir", default=None,
        help="Decrypted official data directory; defaults to <upstream-repo>/data",
    )
    ap.add_argument("--model", default=FROZEN_MODEL)
    ap.add_argument("--n-items", type=int, default=FROZEN_N_ITEMS)
    ap.add_argument("--seed", type=int, default=FROZEN_SEED)
    ap.add_argument("--outdir", default="artifacts/source_trust_transfer_g0")
    args = ap.parse_args()

    # Frozen scientific contract: these are not search knobs.
    if args.model != FROZEN_MODEL:
        raise SystemExit(f"STOP: frozen model is {FROZEN_MODEL}")
    if args.n_items != FROZEN_N_ITEMS:
        raise SystemExit(f"STOP: frozen n-items is {FROZEN_N_ITEMS}")
    if args.seed != FROZEN_SEED:
        raise SystemExit(f"STOP: frozen seed is {FROZEN_SEED}")

    upstream = Path(args.upstream_repo).resolve()
    head = git_head(upstream)
    if head != EXPECTED_UPSTREAM_COMMIT:
        raise SystemExit(
            f"STOP: upstream HEAD must be {EXPECTED_UPSTREAM_COMMIT}; got {head}"
        )

    sys.path.insert(0, str(upstream))
    from helpers.config import DEFAULT_SYSTEM_MESSAGE
    from helpers.data_loader import (
        load_circulation_values,
        load_entities,
        load_government_templates,
        load_names,
        load_question_cache,
        load_source_templates,
        load_timeline_locations,
    )
    from helpers.model_inference import get_answer_probabilities, load_model
    from helpers.prompt_builder import (
        create_entity_table,
        format_answer_choices,
        format_source_info,
        generate_question,
        to_markdown_table,
    )
    from helpers.source_generator import SourceGenerator

    if DEFAULT_SYSTEM_MESSAGE != EXPECTED_SYSTEM_MESSAGE:
        raise SystemExit("STOP: upstream system-message contract changed")

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir).resolve() if args.data_dir else upstream / "data"
    entities = load_entities(str(data_dir))
    question_cache = load_question_cache(str(data_dir))
    source_gen = SourceGenerator(
        source_templates=load_source_templates(str(data_dir)),
        timeline_locations=load_timeline_locations(str(data_dir)),
        government_templates=load_government_templates(str(data_dir)),
        names_data=load_names(str(data_dir)),
        circulation_values=load_circulation_values(str(data_dir)),
    )

    items = make_items(entities, args.n_items, args.seed)
    tokenizer, model, batch_size = load_model(args.model)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    records_path = outdir / "records.jsonl"
    audit_path = outdir / "prompt_audit.jsonl"

    prompt_jobs: list[dict[str, Any]] = []
    item_meta: list[dict[str, Any]] = []

    for item_id, item in enumerate(items):
        source1 = source_gen.generate_social_media_user(popularity_tier=None)
        source2 = source_gen.generate_social_media_user(popularity_tier=None)
        while source2.username == source1.username:
            source2 = source_gen.generate_social_media_user(popularity_tier=None)

        question = generate_question(
            item["target_entity"], item["target_field"], question_cache,
            model=None, tokenizer=None,
        )
        item_meta.append(
            {
                "item_id": item_id,
                "target_entity_id": item["target_entity"].id,
                "target_entity_name": item["target_entity"].name,
                "entity_class": item["target_entity"].entity_class.value,
                "target_field": item["target_field"],
                "target_value_s1": item["target_value_s1"],
                "target_value_s2": item["target_value_s2"],
                "hist1_entity_id": item["hist1_entity"].id,
                "hist1_field": item["hist1_field"],
                "hist1_value": item["hist1_value"],
                "hist2_entity_id": item["hist2_entity"].id,
                "hist2_field": item["hist2_field"],
                "hist2_value": item["hist2_value"],
                "source1": source1.username,
                "source2": source2.username,
            }
        )

        for repeated_source in ("S1", "S2"):
            for reverse_history in (False, True):
                for swap_target_tables in (False, True):
                    for swap_answer_order in (False, True):
                        prompt, s1_answer_index = build_prompt(
                            repeated_source=repeated_source,
                            reverse_history=reverse_history,
                            swap_target_tables=swap_target_tables,
                            swap_answer_order=swap_answer_order,
                            source1=source1,
                            source2=source2,
                            hist1_entity=item["hist1_entity"],
                            hist1_field=item["hist1_field"],
                            hist1_value=item["hist1_value"],
                            hist2_entity=item["hist2_entity"],
                            hist2_field=item["hist2_field"],
                            hist2_value=item["hist2_value"],
                            target_entity=item["target_entity"],
                            target_field=item["target_field"],
                            target_value_s1=item["target_value_s1"],
                            target_value_s2=item["target_value_s2"],
                            question=question,
                            format_source_info=format_source_info,
                            to_markdown_table=to_markdown_table,
                            create_entity_table=create_entity_table,
                            format_answer_choices=format_answer_choices,
                        )
                        prompt_jobs.append(
                            {
                                "item_id": item_id,
                                "repeated_source": repeated_source,
                                "reverse_history": reverse_history,
                                "swap_target_tables": swap_target_tables,
                                "swap_answer_order": swap_answer_order,
                                "s1_answer_index": s1_answer_index,
                                "prompt": prompt,
                            }
                        )

    # Before inference, prove the paired causal contrast is structurally exact.
    paired: dict[tuple[int, bool, bool, bool], dict[str, str]] = defaultdict(dict)
    for job in prompt_jobs:
        key = (
            job["item_id"], job["reverse_history"],
            job["swap_target_tables"], job["swap_answer_order"],
        )
        paired[key][job["repeated_source"]] = job["prompt"]

    for key, pair in paired.items():
        if set(pair) != {"S1", "S2"}:
            raise RuntimeError(f"Missing paired condition for {key}")
        p1, p2 = pair["S1"], pair["S2"]
        marker = "Current reports:\n"
        target1 = p1.split(marker, 1)[1]
        target2 = p2.split(marker, 1)[1]
        if target1 != target2:
            raise RuntimeError(f"Target section mismatch for {key}")
        if len(p1) != len(p2):
            raise RuntimeError(
                f"Paired prompt length mismatch for {key}: {len(p1)} vs {len(p2)}"
            )

    probabilities: list[tuple[float, float]] = []
    for start in range(0, len(prompt_jobs), batch_size):
        chunk = prompt_jobs[start : start + batch_size]
        probs = get_answer_probabilities(
            model,
            tokenizer,
            [job["prompt"] for job in chunk],
            DEFAULT_SYSTEM_MESSAGE,
            neutral=False,
            alttok=False,
        )
        probabilities.extend((float(a), float(b)) for a, b in probs)

    if len(probabilities) != len(prompt_jobs):
        raise RuntimeError("Inference result count mismatch")

    by_item: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    raw_rows: list[dict[str, Any]] = []
    cell_map: dict[tuple[int, bool, bool, bool], dict[str, float]] = defaultdict(dict)

    for job, (prob_a, prob_b) in zip(prompt_jobs, probabilities):
        p_s1 = prob_a if job["s1_answer_index"] == 0 else prob_b
        row = {k: v for k, v in job.items() if k != "prompt"}
        row.update({"prob_a": prob_a, "prob_b": prob_b, "p_source1_target": p_s1})
        raw_rows.append(row)
        by_item[job["item_id"]][job["repeated_source"]].append(p_s1)
        key = (
            job["item_id"], job["reverse_history"],
            job["swap_target_tables"], job["swap_answer_order"],
        )
        cell_map[key][job["repeated_source"]] = p_s1

    factor_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (_, reverse_history, swap_tables, swap_answers), vals in cell_map.items():
        delta = vals["S1"] - vals["S2"]
        factor_values["history_order"]["reversed" if reverse_history else "forward"].append(delta)
        factor_values["target_table_order"]["swapped" if swap_tables else "canonical"].append(delta)
        factor_values["answer_order"]["swapped" if swap_answers else "canonical"].append(delta)

    item_rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    follow_flip_count = 0
    opposite_flip_count = 0

    for meta in item_meta:
        item_id = meta["item_id"]
        p_when_s1_repeated = float(np.mean(by_item[item_id]["S1"]))
        p_when_s2_repeated = float(np.mean(by_item[item_id]["S2"]))
        delta = p_when_s1_repeated - p_when_s2_repeated
        follows = p_when_s1_repeated > 0.5 and p_when_s2_repeated < 0.5
        opposite = p_when_s1_repeated < 0.5 and p_when_s2_repeated > 0.5
        deltas.append(delta)
        follow_flip_count += int(follows)
        opposite_flip_count += int(opposite)
        item_rows.append(
            {
                **meta,
                "mean_p_source1_when_s1_repeated": p_when_s1_repeated,
                "mean_p_source1_when_s2_repeated": p_when_s2_repeated,
                "transfer_delta": delta,
                "choice_follows_repeated_source": follows,
                "opposite_flip": opposite,
            }
        )

    ci_low, ci_high = bootstrap_ci(deltas, seed=args.seed + 17)
    mean_delta = float(np.mean(deltas))
    positive_fraction = float(np.mean(np.asarray(deltas) > 0))
    factor_means = {
        factor: {level: float(np.mean(values)) for level, values in levels.items()}
        for factor, levels in factor_values.items()
    }
    all_counterbalances_positive = all(
        mean > 0
        for levels in factor_means.values()
        for mean in levels.values()
    )

    # Frozen before running the new hypothesis. 5 pp is ~1/6 of the reproduced
    # 30.55 pp immediate social-media repetition shift: large enough to matter
    # as cross-claim reputation transfer rather than a tiny residual.
    gate = {
        "n_items_eq_128": len(item_rows) == FROZEN_N_ITEMS,
        "mean_transfer_delta_ge_0.05": mean_delta >= 0.05,
        "bootstrap_95_lower_gt_0": ci_low > 0.0,
        "positive_item_fraction_ge_0.60": positive_fraction >= 0.60,
        "all_counterbalance_level_means_gt_0": all_counterbalances_positive,
        # Independent target entities for a later discrete mechanism cell.
        "choice_following_repetition_flips_ge_12": follow_flip_count >= 12,
    }

    if all(gate.values()):
        verdict = "GO_REGISTER_TOPIC23_SOURCE_LEVEL_TRANSFER"
    elif ci_high < 0.03:
        verdict = "KILL_SOURCE_LEVEL_TRANSFER_PAPER_SCALE"
    elif ci_low > 0.0 and mean_delta < 0.05:
        verdict = "WEAK_POSITIVE_BELOW_PAPER_SCALE_DO_NOT_REGISTER"
    else:
        verdict = "INCONCLUSIVE_DO_NOT_TUNE"

    summary = {
        "scientific_question": "Does repetition transfer across claims through source-level trust?",
        "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
        "model": args.model,
        "seed": args.seed,
        "n_items": len(item_rows),
        "prompt_variations_per_condition_per_item": 8,
        "total_prompts": len(prompt_jobs),
        "mean_transfer_delta_probability": mean_delta,
        "mean_transfer_delta_percentage_points": 100.0 * mean_delta,
        "bootstrap_95_ci_probability": [ci_low, ci_high],
        "bootstrap_95_ci_percentage_points": [100.0 * ci_low, 100.0 * ci_high],
        "positive_item_fraction": positive_fraction,
        "choice_following_repetition_flip_count": follow_flip_count,
        "opposite_flip_count": opposite_flip_count,
        "counterbalance_level_mean_deltas": factor_means,
        "gate": gate,
        "verdict": verdict,
        "interpretation_boundary": (
            "The paired contrast holds source identities, source exposure counts, the global "
            "history-claim multiset, target conflict, target values, target entity, and target "
            "question fixed. It changes only which source is associated with the duplicate "
            "historical claim. A positive effect therefore supports cross-claim source-level "
            "transfer rather than simple claim frequency or source-name exposure."
        ),
    }

    with records_path.open("w", encoding="utf-8") as f:
        for row in item_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Save only a small human-audit sample of full prompts.
    with audit_path.open("w", encoding="utf-8") as f:
        for job in prompt_jobs[:32]:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")

    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
