#!/usr/bin/env python3
"""Best-effort adapter for the seed repository's unzipped 64-response release."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from common import write_jsonl

STEP_RE = re.compile(r"step[_-]?(\d+)", re.I)


def infer_step(path: Path) -> int:
    for part in reversed(path.parts):
        m = STEP_RE.search(part)
        if m:
            return int(m.group(1))
    raise ValueError(f"Cannot infer checkpoint step from {path}")


def task_from_judge(path: Path) -> str:
    name = path.stem
    prefix = "llm_answer_check_result_"
    if not name.startswith(prefix):
        raise ValueError(path)
    return name[len(prefix) :]


def find_samples(directory: Path, task: str) -> Path:
    candidates = sorted(directory.glob(f"samples_{task}*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No samples_{task}*.jsonl under {directory}")
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple sample files for {task}: {candidates}")
    return candidates[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    judge_files = sorted(p for p in root.rglob("llm_answer_check_result_*") if p.is_file())
    if not judge_files:
        raise SystemExit("No official judgement files found. Unzip sampling_64_responses.zip first.")

    all_steps = sorted({infer_step(p) for p in judge_files})
    order_of = {step: i for i, step in enumerate(all_steps)}
    out = []

    for jf in judge_files:
        step = infer_step(jf)
        task = task_from_judge(jf)
        sf = find_samples(jf.parent, task)
        judged = json.loads(jf.read_text(encoding="utf-8"))
        sample_rows = [json.loads(x) for x in sf.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(judged) != len(sample_rows):
            raise RuntimeError(f"Problem-count mismatch: {jf} vs {sf}")

        for i, (jrow, srow) in enumerate(zip(judged, sample_rows)):
            resps = srow.get("resps", [[]])[0]
            jresps = jrow.get("responses", [])
            if len(resps) != len(jresps):
                raise RuntimeError(f"Response-count mismatch task={task} problem={i} step={step}")
            doc = srow.get("doc", {})
            problem = doc.get("problem") or doc.get("question") or jrow.get("problem")
            answer = doc.get("answer") or jrow.get("answer")
            pid = f"{task}_{i:04d}"
            for ridx, (resp, jr) in enumerate(zip(resps, jresps)):
                flag = jr.get("llm_check_result")
                if flag is None:
                    continue
                out.append(
                    {
                        "problem_id": pid,
                        "task": task,
                        "checkpoint": f"step_{step}",
                        "checkpoint_order": order_of[step],
                        "prompt": problem,
                        "problem": problem,
                        "gold_answer": answer,
                        "sample_index": ridx,
                        "response": resp,
                        "correct": bool(flag),
                        "source": "official_64_release",
                    }
                )

    write_jsonl(args.output, out)
    print(f"steps={all_steps} rows={len(out)}")


if __name__ == "__main__":
    main()
