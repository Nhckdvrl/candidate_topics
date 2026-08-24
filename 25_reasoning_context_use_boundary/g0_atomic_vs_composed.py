#!/usr/bin/env python3
"""Frozen Topic 25 G0: matched atomic evidence use vs composed integration.

The script deliberately imports prompt construction, Qwen3 think/no-think API
calls, answer extraction, document placement, and Exact Match from the pinned
Weakest-Link repository. Our code owns only the new matched query manipulation,
case eligibility, preregistered statistics, and gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PINNED_UPSTREAM_COMMIT = "9b01abaad354208a6a8fb26c58eb5c330036fb94"
SOURCE_DATASET = "Shahar6000/MoreDocsSameLen"
SOURCE_DATASET_REVISION = "6053179334f456f9490457edcc91ea3196fd54d7"
SOURCE_SPLIT = "train"
MODEL_SHORT = "Qwen3-8B"
SELECTION_SEED = 20260825
GENERATION_SEED = 42
N_ITEMS = 256
BUCKETS = ("beginning", "midsection", "tail")
DISTANCE = 1
PROMPT_ID = 22
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_CI = 0.90
_PLACEHOLDER = re.compile(r"#(\d+)")


def _normalize_ws(text: Any) -> str:
    return " ".join(str(text or "").split())


def _stable_rank(item_id: str, seed: int = SELECTION_SEED) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).hexdigest()


def _render_atomic_question(decomposition: Sequence[Dict[str, Any]], index: int) -> str:
    """Resolve only references to already released earlier intermediate answers."""
    if not 0 <= index < len(decomposition):
        raise IndexError(index)
    question = str(decomposition[index].get("question", "")).strip()
    if not question:
        raise ValueError("empty atomic question")

    def repl(match: re.Match[str]) -> str:
        ref = int(match.group(1)) - 1
        if ref < 0 or ref >= index:
            raise ValueError(
                f"atomic step {index} contains non-earlier placeholder {match.group(0)}"
            )
        answer = str(decomposition[ref].get("answer", "")).strip()
        if not answer:
            raise ValueError(f"missing released intermediate answer for step {ref}")
        return answer

    rendered = _PLACEHOLDER.sub(repl, question)
    if _PLACEHOLDER.search(rendered):
        raise ValueError(f"unresolved placeholder after rendering: {rendered}")
    return rendered


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty percentile input")
    if not 0.0 <= q <= 1.0:
        raise ValueError(q)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{lineno}")
            rows.append(obj)
    return rows


def _load_source_rows() -> List[Dict[str, Any]]:
    # Lazy import keeps helper unit tests independent of HF/datasets.
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(
        SOURCE_DATASET,
        split=SOURCE_SPLIT,
        revision=SOURCE_DATASET_REVISION,
    )
    return [dict(row) for row in ds]


def _source_indices(
    rows: Iterable[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    by_question: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        rid = str(row.get("id", ""))
        if rid:
            if rid in by_id:
                raise ValueError(f"duplicate source id: {rid}")
            by_id[rid] = row
        q = _normalize_ws(row.get("question", ""))
        if q:
            by_question.setdefault(q, []).append(row)
    return by_id, by_question


def _match_source_row(
    bank_row: Dict[str, Any],
    by_id: Dict[str, Dict[str, Any]],
    by_question: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Optional[Dict[str, Any]], str]:
    qid = str(bank_row.get("question_id", bank_row.get("id", "")))
    if qid and qid in by_id:
        return by_id[qid], "id"
    q = _normalize_ws(bank_row.get("question", ""))
    candidates = by_question.get(q, [])
    if len(candidates) == 1:
        return candidates[0], "exact_question"
    if not candidates:
        return None, "no_match"
    return None, "ambiguous_question"


def _build_eligible_case(
    bank_row: Dict[str, Any], source_row: Dict[str, Any], match_mode: str
) -> Tuple[Optional[Dict[str, Any]], str]:
    decomposition = source_row.get("question_decomposition") or []
    if len(decomposition) != 2:
        return None, "not_exactly_2hop"

    paragraphs = source_row.get("paragraphs") or []
    gold_docs = bank_row.get("gold_docs") or []
    if len(gold_docs) != 2:
        return None, "bank_not_two_gold"

    gold_texts = [_normalize_ws(doc.get("text", "")) for doc in gold_docs]
    if not all(gold_texts) or gold_texts[0] == gold_texts[1]:
        return None, "bad_bank_gold_texts"

    support_gold_indices: List[int] = []
    atomic_questions: List[str] = []
    atomic_answers: List[str] = []

    for step_idx, step in enumerate(decomposition):
        try:
            support_idx = int(step.get("paragraph_support_idx"))
        except (TypeError, ValueError):
            return None, "bad_support_idx"
        if not 0 <= support_idx < len(paragraphs):
            return None, "support_idx_out_of_range"
        support_text = _normalize_ws(paragraphs[support_idx].get("paragraph_text", ""))
        hits = [i for i, gold_text in enumerate(gold_texts) if support_text == gold_text]
        if len(hits) != 1:
            return None, "support_not_unique_bank_gold"
        support_gold_indices.append(hits[0])

        try:
            atomic_question = _render_atomic_question(decomposition, step_idx)
        except (ValueError, IndexError):
            return None, "placeholder_resolution_failed"
        atomic_answer = str(step.get("answer", "")).strip()
        if not atomic_answer:
            return None, "missing_atomic_answer"
        atomic_questions.append(atomic_question)
        atomic_answers.append(atomic_answer)

    if set(support_gold_indices) != {0, 1}:
        return None, "atomic_steps_do_not_cover_both_gold_docs"

    item_id = str(bank_row.get("question_id", bank_row.get("id", "")))
    if not item_id:
        return None, "missing_item_id"

    source_id = str(source_row.get("id", ""))
    composed_question = str(bank_row.get("question", "")).strip()
    answers = bank_row.get("answers") or []
    if not composed_question or not answers or not str(answers[0]).strip():
        return None, "missing_composed_fields"

    return (
        {
            "item_id": item_id,
            "source_id": source_id,
            "match_mode": match_mode,
            "bank": bank_row,
            "atomic_questions": atomic_questions,
            "atomic_answers": atomic_answers,
            "support_gold_indices": support_gold_indices,
            "composed_question": composed_question,
            "composed_answer": str(answers[0]).strip(),
        },
        "ok",
    )


def _verify_upstream(upstream_repo: Path) -> str:
    actual = subprocess.check_output(
        ["git", "-C", str(upstream_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != PINNED_UPSTREAM_COMMIT:
        raise RuntimeError(
            f"upstream commit mismatch: expected {PINNED_UPSTREAM_COMMIT}, got {actual}"
        )
    return actual


def _import_upstream(upstream_repo: Path) -> Dict[str, Any]:
    root = str(upstream_repo)
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.evaluate.metrics import compute_exact  # type: ignore
    from src.infer.entity.common import (  # type: ignore
        assemble_documents_spread,
        build_musique_prompt,
        call_llm_api,
        extract_answer_from_response,
    )
    from src.utils.model_name_mapping import get_full_name  # type: ignore

    return {
        "compute_exact": compute_exact,
        "assemble_documents_spread": assemble_documents_spread,
        "build_musique_prompt": build_musique_prompt,
        "call_llm_api": call_llm_api,
        "extract_answer_from_response": extract_answer_from_response,
        "get_full_name": get_full_name,
    }


def _task_key(
    item_id: str, bucket: str, query_type: str, enable_thinking: bool
) -> str:
    return f"{item_id}\t{bucket}\t{query_type}\t{int(enable_thinking)}"


def _load_existing_records(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    rows = _read_jsonl(path)
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = _task_key(
            str(row["item_id"]),
            str(row["bucket"]),
            str(row["query_type"]),
            bool(row["enable_thinking"]),
        )
        if key in out:
            raise ValueError(f"duplicate completed G0 record key: {key}")
        out[key] = row
    return out


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("empty mean")
    return sum(values) / len(values)


def _bootstrap_interaction_ci(
    item_contrib: Dict[str, float], seed: int = SELECTION_SEED
) -> Tuple[float, float]:
    ids = sorted(item_contrib)
    vals = [item_contrib[i] for i in ids]
    if not vals:
        raise ValueError("no item contributions")
    rng = random.Random(seed + 7001)
    reps: List[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        reps.append(_mean([vals[rng.randrange(len(vals))] for _ in range(len(vals))]))
    reps.sort()
    alpha = (1.0 - BOOTSTRAP_CI) / 2.0
    return _percentile(reps, alpha), _percentile(reps, 1.0 - alpha)


def _summarize(
    records: Dict[str, Dict[str, Any]], selected_ids: List[str], error_count: int
) -> Dict[str, Any]:
    expected_keys = {
        _task_key(item_id, bucket, query_type, think)
        for item_id in selected_ids
        for bucket in BUCKETS
        for query_type in ("atomic_0", "atomic_1", "composed")
        for think in (False, True)
    }
    actual_keys = set(records)
    complete = expected_keys == actual_keys and error_count == 0

    paired_rows: List[Dict[str, Any]] = []
    if complete:
        for item_id in selected_ids:
            for bucket in BUCKETS:
                def corr(query_type: str, think: bool) -> int:
                    row = records[_task_key(item_id, bucket, query_type, think)]
                    return 1 if bool(row["correct"]) else 0

                atomic_no = int(corr("atomic_0", False) and corr("atomic_1", False))
                atomic_think = int(corr("atomic_0", True) and corr("atomic_1", True))
                composed_no = corr("composed", False)
                composed_think = corr("composed", True)
                paired_rows.append(
                    {
                        "item_id": item_id,
                        "bucket": bucket,
                        "atomic_no": atomic_no,
                        "atomic_think": atomic_think,
                        "composed_no": composed_no,
                        "composed_think": composed_think,
                        "d_atomic": atomic_think - atomic_no,
                        "d_composed": composed_think - composed_no,
                        "interaction": (composed_think - composed_no)
                        - (atomic_think - atomic_no),
                    }
                )

    def metrics(rows: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        if not rows:
            return None
        atomic_no = _mean([float(r["atomic_no"]) for r in rows])
        atomic_think = _mean([float(r["atomic_think"]) for r in rows])
        composed_no = _mean([float(r["composed_no"]) for r in rows])
        composed_think = _mean([float(r["composed_think"]) for r in rows])
        d_atomic = atomic_think - atomic_no
        d_composed = composed_think - composed_no
        return {
            "atomic_no": atomic_no,
            "atomic_think": atomic_think,
            "composed_no": composed_no,
            "composed_think": composed_think,
            "d_atomic": d_atomic,
            "d_composed": d_composed,
            "interaction": d_composed - d_atomic,
        }

    pooled = metrics(paired_rows)
    by_bucket = {
        bucket: metrics([r for r in paired_rows if r["bucket"] == bucket])
        for bucket in BUCKETS
    }

    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    if complete:
        by_item: Dict[str, List[float]] = {item_id: [] for item_id in selected_ids}
        for row in paired_rows:
            by_item[row["item_id"]].append(float(row["interaction"]))
        item_contrib = {item_id: _mean(vals) for item_id, vals in by_item.items()}
        ci_low, ci_high = _bootstrap_interaction_ci(item_contrib)

    positive_buckets = sum(
        1
        for value in by_bucket.values()
        if value is not None and value["interaction"] > 0.0
    )

    gates: Dict[str, bool] = {
        "selected_items_eq_256": len(selected_ids) == N_ITEMS,
        "all_expected_calls_complete": complete,
        "no_think_atomic_ge_0p30": bool(pooled and pooled["atomic_no"] >= 0.30),
        "no_think_composed_ge_0p15": bool(pooled and pooled["composed_no"] >= 0.15),
        "composed_thinking_gain_ge_0p08": bool(
            pooled and pooled["d_composed"] >= 0.08
        ),
        "atomic_thinking_gain_le_0p03": bool(pooled and pooled["d_atomic"] <= 0.03),
        "interaction_ge_0p08": bool(pooled and pooled["interaction"] >= 0.08),
        "bootstrap_90ci_lower_gt_0": bool(ci_low is not None and ci_low > 0.0),
        "positive_interaction_in_ge_2_buckets": positive_buckets >= 2,
    }

    verdict = "GO_MATCHED_BOUNDARY" if all(gates.values()) else "STOP_MATCHED_BOUNDARY"
    if not complete:
        verdict = "INCOMPLETE_ENGINEERING"

    strong_sign_reversal = bool(
        pooled and pooled["d_atomic"] < 0.0 and pooled["d_composed"] > 0.0
    )

    return {
        "verdict": verdict,
        "selected_items": len(selected_ids),
        "expected_calls": len(expected_keys),
        "completed_calls": len(actual_keys & expected_keys),
        "unexpected_calls": len(actual_keys - expected_keys),
        "missing_calls": len(expected_keys - actual_keys),
        "error_count_this_run": error_count,
        "pooled": pooled,
        "by_bucket": by_bucket,
        "paired_bootstrap_90ci_interaction": [ci_low, ci_high],
        "positive_interaction_buckets": positive_buckets,
        "strong_matched_sign_reversal_diagnostic": strong_sign_reversal,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-repo", required=True)
    parser.add_argument("--bank-file", required=True)
    parser.add_argument("--api-url", default="http://localhost:8000/v1")
    parser.add_argument("--output-dir", default="artifacts/g0")
    parser.add_argument("--num-threads", type=int, default=12)
    args = parser.parse_args()

    upstream_repo = Path(args.upstream_repo).expanduser().resolve()
    bank_file = Path(args.bank_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_upstream = _verify_upstream(upstream_repo)
    upstream = _import_upstream(upstream_repo)

    bank_rows = _read_jsonl(bank_file)
    source_rows = _load_source_rows()
    by_id, by_question = _source_indices(source_rows)

    eligible: List[Dict[str, Any]] = []
    reject_counts: Dict[str, int] = {}
    match_counts: Dict[str, int] = {}

    for bank_row in bank_rows:
        source_row, match_mode = _match_source_row(bank_row, by_id, by_question)
        match_counts[match_mode] = match_counts.get(match_mode, 0) + 1
        if source_row is None:
            reject_counts[match_mode] = reject_counts.get(match_mode, 0) + 1
            continue
        case, reason = _build_eligible_case(bank_row, source_row, match_mode)
        if case is None:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
            continue
        eligible.append(case)

    eligible.sort(key=lambda c: _stable_rank(str(c["item_id"])))
    if len(eligible) < N_ITEMS:
        raise RuntimeError(
            f"Only {len(eligible)} eligible matched 2-hop cases; frozen G0 requires {N_ITEMS}. "
            f"Rejects: {reject_counts}"
        )
    selected = eligible[:N_ITEMS]
    selected_ids = [str(c["item_id"]) for c in selected]
    if len(set(selected_ids)) != N_ITEMS:
        raise RuntimeError("selected item IDs are not unique")

    selected_path = output_dir / "selected_cases.jsonl"
    with selected_path.open("w", encoding="utf-8") as f:
        for case in selected:
            compact = {
                "item_id": case["item_id"],
                "source_id": case["source_id"],
                "match_mode": case["match_mode"],
                "atomic_questions": case["atomic_questions"],
                "atomic_answers": case["atomic_answers"],
                "support_gold_indices": case["support_gold_indices"],
                "composed_question": case["composed_question"],
                "composed_answer": case["composed_answer"],
                "selection_rank": _stable_rank(str(case["item_id"])),
            }
            f.write(json.dumps(compact, ensure_ascii=False) + "\n")

    contract = {
        "upstream_commit": actual_upstream,
        "source_dataset": SOURCE_DATASET,
        "source_dataset_revision": SOURCE_DATASET_REVISION,
        "source_split": SOURCE_SPLIT,
        "model": "Qwen/Qwen3-8B",
        "selection_seed": SELECTION_SEED,
        "generation_seed": GENERATION_SEED,
        "n_items": N_ITEMS,
        "buckets": list(BUCKETS),
        "distance": DISTANCE,
        "prompt_id": PROMPT_ID,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": {"no_think": 3000, "think": 10000},
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_ci": BOOTSTRAP_CI,
        "bank_n": len(bank_rows),
        "eligible_n": len(eligible),
        "match_counts": match_counts,
        "reject_counts": reject_counts,
    }
    (output_dir / "run_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8"
    )

    full_model_name = upstream["get_full_name"](MODEL_SHORT)
    records_path = output_dir / "records.jsonl"
    existing = _load_existing_records(records_path)

    tasks: List[Dict[str, Any]] = []
    for case in selected:
        for bucket in BUCKETS:
            documents, gold_positions = upstream["assemble_documents_spread"](
                case["bank"], bucket_name=bucket, distance=DISTANCE
            )
            query_defs = [
                (
                    "atomic_0",
                    case["atomic_questions"][0],
                    case["atomic_answers"][0],
                    gold_positions[case["support_gold_indices"][0]],
                ),
                (
                    "atomic_1",
                    case["atomic_questions"][1],
                    case["atomic_answers"][1],
                    gold_positions[case["support_gold_indices"][1]],
                ),
                (
                    "composed",
                    case["composed_question"],
                    case["composed_answer"],
                    None,
                ),
            ]
            for query_type, question, gold_answer, atomic_support_position in query_defs:
                for enable_thinking in (False, True):
                    key = _task_key(
                        str(case["item_id"]), bucket, query_type, enable_thinking
                    )
                    if key in existing:
                        continue
                    tasks.append(
                        {
                            "key": key,
                            "item_id": str(case["item_id"]),
                            "bucket": bucket,
                            "query_type": query_type,
                            "enable_thinking": enable_thinking,
                            "question": question,
                            "gold_answer": gold_answer,
                            "documents": documents,
                            "gold_positions": gold_positions,
                            "atomic_support_position": atomic_support_position,
                        }
                    )

    def worker(task: Dict[str, Any]) -> Dict[str, Any]:
        prompt = upstream["build_musique_prompt"](
            task["question"], task["documents"], prompt_id=PROMPT_ID
        )
        think = bool(task["enable_thinking"])
        response_content, reasoning_content = upstream["call_llm_api"](
            prompt,
            api_base=args.api_url,
            model_name=full_model_name,
            temperature=0.0,
            max_tokens=10000 if think else 3000,
            top_p=1.0,
            enable_thinking=think,
            seed=GENERATION_SEED,
            timeout=300.0,
        )
        pred = upstream["extract_answer_from_response"](response_content)
        model_answer = str(pred.get("answer_content", "")).strip()
        correct = bool(upstream["compute_exact"](task["gold_answer"], model_answer))
        return {
            "item_id": task["item_id"],
            "bucket": task["bucket"],
            "distance": DISTANCE,
            "query_type": task["query_type"],
            "enable_thinking": think,
            "question": task["question"],
            "gold_answer": task["gold_answer"],
            "model_answer": model_answer,
            "correct": correct,
            "gold_positions": task["gold_positions"],
            "atomic_support_position": task["atomic_support_position"],
            "response_content": response_content,
            "reasoning_content": reasoning_content,
            "prompt_id": PROMPT_ID,
            "generation_seed": GENERATION_SEED,
        }

    errors: List[Dict[str, str]] = []
    if tasks:
        with records_path.open("a", encoding="utf-8", buffering=1) as fout:
            with ThreadPoolExecutor(max_workers=args.num_threads) as pool:
                future_to_task = {pool.submit(worker, task): task for task in tasks}
                for idx, future in enumerate(as_completed(future_to_task), 1):
                    task = future_to_task[future]
                    try:
                        rec = future.result()
                    except Exception as exc:  # network/runtime errors are not scientific rows
                        errors.append({"key": task["key"], "error": repr(exc)})
                        print(f"ERROR {task['key']}: {exc}", file=sys.stderr)
                        continue
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    existing[task["key"]] = rec
                    if idx % 64 == 0 or idx == len(tasks):
                        print(f"completed {idx}/{len(tasks)} new calls")

    if errors:
        (output_dir / "errors.json").write_text(
            json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # Reload from disk so the summary is based on persisted state, not futures.
    completed = _load_existing_records(records_path)
    summary = _summarize(completed, selected_ids, len(errors))
    summary["contract"] = contract
    summary["selected_cases_path"] = str(selected_path)
    summary["records_path"] = str(records_path)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
