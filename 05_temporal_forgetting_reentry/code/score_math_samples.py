#!/usr/bin/env python3
"""Score generated math responses, aligned with the Temporal Forgetting stack.

Primary recommended mode is `--method hybrid`:
  1. use the official PRIME/MATH rule+sympy scorer from a local clone of
     uw-nsl/Temporal_Forgetting;
  2. optionally send rule-negative cases with an extractable answer to a
     Qwen2.5-32B-Instruct judge, matching the seed repository's documented
     fallback philosophy.

For fast G-1 when the judge is unavailable, use `--method prime` and later
re-score all primary forgotten/control candidates with the judge before freezing
F/N/S membership.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import read_jsonl, write_jsonl, last_boxed_content


def load_prime_compute_score(repo: str):
    root = Path(repo).resolve()
    pkg = root / "Temperal_sampling" / "prime_math"
    if not pkg.exists():
        raise FileNotFoundError(f"Expected official prime_math at {pkg}")
    sys.path.insert(0, str(root / "Temperal_sampling"))
    from prime_math import compute_score  # type: ignore
    return compute_score


def judge_false_cases(rows: list[dict], model_name: str, tp: int) -> dict[int, bool]:
    from vllm import LLM, SamplingParams

    targets = []
    indices = []
    for i, r in enumerate(rows):
        if r.get("prime_correct"):
            continue
        pred = r.get("extracted_answer")
        if pred is None:
            continue
        prompt = (
            "Given a math problem, its correct answer, and a model-generated final answer, "
            "decide mathematical correctness. Reply only True or False.\n"
            f"Problem: {r.get('problem', r.get('prompt',''))}\n"
            f"Correct answer: {r.get('gold_answer')}\n"
            f"Generated answer: {pred}\nJudgement:"
        )
        targets.append(prompt)
        indices.append(i)
    if not targets:
        return {}

    llm = LLM(model=model_name, tensor_parallel_size=tp, dtype="bfloat16", trust_remote_code=True)
    tok = llm.get_tokenizer()
    texts = [
        tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in targets
    ]
    outs = llm.generate(texts, SamplingParams(temperature=0.0, max_tokens=8))
    result = {}
    for i, out in zip(indices, outs):
        txt = out.outputs[0].text.strip().lower()
        if txt.startswith("true") or " true" in txt:
            result[i] = True
        elif txt.startswith("false") or " false" in txt:
            result[i] = False
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--temporal-repo", required=True)
    ap.add_argument("--method", choices=["prime", "hybrid"], default="prime")
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--judge-tp", type=int, default=4)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--checkpoint-order", type=int, default=None)
    args = ap.parse_args()

    compute_score = load_prime_compute_score(args.temporal_repo)
    rows = read_jsonl(args.input)
    out = []
    for r in rows:
        response = str(r.get("response", ""))
        gold = str(r.get("gold_answer", ""))
        try:
            score = compute_score(response, gold)
            if isinstance(score, tuple):
                prime_correct = bool(score[0])
                extracted = score[2] if len(score) >= 3 else last_boxed_content(response)
            else:
                prime_correct = bool(score)
                extracted = last_boxed_content(response)
        except Exception as exc:
            prime_correct = False
            extracted = last_boxed_content(response)
            r = {**r, "prime_score_error": repr(exc)}
        out.append({**r, "prime_correct": prime_correct, "extracted_answer": extracted})

    judge = {}
    if args.method == "hybrid":
        judge = judge_false_cases(out, args.judge_model, args.judge_tp)

    final = []
    for i, r in enumerate(out):
        correct = bool(r["prime_correct"] or judge.get(i, False))
        row = {**r, "correct": correct, "score_method": args.method}
        if i in judge:
            row["judge_correct"] = judge[i]
        if args.checkpoint is not None:
            row["checkpoint"] = args.checkpoint
        if args.checkpoint_order is not None:
            row["checkpoint_order"] = args.checkpoint_order
        final.append(row)
    write_jsonl(args.output, final)
    print(f"rows={len(final)} correct={sum(bool(r['correct']) for r in final)} method={args.method}")


if __name__ == "__main__":
    main()
