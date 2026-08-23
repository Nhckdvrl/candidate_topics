#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TargetProgram:
    sample_id: int
    code: str
    input_x: int
    output: list[int]
    lexical_line: str
    trace: list[dict]


@dataclass
class ContextPair:
    start: str
    middle: str
    start_tokens: int
    middle_tokens: int
    start_target_center_fraction: float
    middle_target_center_fraction: float
    distractor_sha256: str
    local_neighbor_sha256: str


def build_target_program(rng: random.Random, sample_id: int, n_steps: int) -> TargetProgram:
    arr = [0] * n_steps
    x = rng.randint(10, 99)
    order = list(range(n_steps))
    rng.shuffle(order)
    lines = [f"def target_{sample_id}(x):", f"    arr = {arr!r}"]
    prev, prev_ref, trace, assigns = x, "x", [], []
    for step, idx in enumerate(order):
        delta = rng.choice([d for d in range(-99, 100) if d])
        value = prev + delta
        op = "+" if delta > 0 else "-"
        line = f"    arr[{idx}] = {prev_ref} {op} {abs(delta)}"
        lines.append(line)
        assigns.append(line)
        arr[idx] = value
        trace.append({"step": step, "target_index": idx, "source": prev_ref, "delta": delta, "value": value})
        prev, prev_ref = value, f"arr[{idx}]"
    lines.append("    return arr")
    return TargetProgram(sample_id, "\n".join(lines), x, arr, assigns[len(assigns) // 2], trace)


def build_distractor(rng: random.Random, idx: int, n_lines: int = 12) -> str:
    lines = [f"def helper_{idx}(x):", f"    v = x + {rng.randint(-20, 20)}"]
    for j in range(n_lines):
        k = rng.randint(1, 97)
        if j % 3 == 0:
            lines.append(f"    v = (v * {rng.randint(2, 9)} + {k}) % 100003")
        elif j % 3 == 1:
            lines.append(f"    v = v - {k}")
        else:
            lines.append(f"    v = v + {k}")
    lines.append("    return v")
    return "\n".join(lines)


def token_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False).input_ids)


def _compose_parts(tok, header: str, parts: list[str], target_idx: int) -> tuple[str, int, float]:
    text = header + "\n\n".join(parts)
    if target_idx:
        prefix = header + "\n\n".join(parts[:target_idx]) + "\n\n"
    else:
        prefix = header
    target = parts[target_idx]
    target_start = token_len(tok, prefix)
    target_end = token_len(tok, prefix + target)
    total = token_len(tok, text)
    center = ((target_start + target_end) / 2) / max(1, total)
    return text, total, center


def make_context_pair(tok, target: str, rng: random.Random, target_tokens: int) -> ContextPair:
    header = "# Synthetic code repository\n\n"
    budget = max(512, target_tokens - token_len(tok, target) - token_len(tok, header) - 128)
    blocks, used, i = [], 0, 0
    while used < budget:
        block = build_distractor(rng, i)
        n_tok = token_len(tok, block + "\n\n")
        if used + n_tok > budget and blocks:
            break
        blocks.append(block)
        used += n_tok
        i += 1

    if len(blocks) < 4:
        raise RuntimeError("Context budget produced fewer than four distractor blocks")

    # Preserve the target's immediate lexical neighborhood across positions.
    # Only the amount of distant prefix/suffix context changes.
    guard_before, guard_after = blocks[0], blocks[1]
    mobile = blocks[2:]
    package = [guard_before, target, guard_after]

    start_parts = package + mobile
    start_text, start_tokens, start_center = _compose_parts(tok, header, start_parts, target_idx=1)

    candidates = []
    for split in range(len(mobile) + 1):
        parts = mobile[:split] + package + mobile[split:]
        target_idx = split + 1
        candidates.append(_compose_parts(tok, header, parts, target_idx) + (split,))
    middle_text, middle_tokens, middle_center, _ = min(candidates, key=lambda x: abs(x[2] - 0.5))

    distractor_digest = hashlib.sha256("\n\n".join(blocks).encode("utf-8")).hexdigest()
    neighbor_digest = hashlib.sha256((guard_before + "\n\n" + guard_after).encode("utf-8")).hexdigest()
    return ContextPair(
        start=start_text,
        middle=middle_text,
        start_tokens=start_tokens,
        middle_tokens=middle_tokens,
        start_target_center_fraction=start_center,
        middle_target_center_fraction=middle_center,
        distractor_sha256=distractor_digest,
        local_neighbor_sha256=neighbor_digest,
    )


def chat(tok, user: str) -> str:
    messages = [
        {"role": "system", "content": "Controlled code experiment. Follow output format exactly; no explanation."},
        {"role": "user", "content": user},
    ]
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return messages[0]["content"] + "\n\n" + user + "\nAnswer:"


@torch.inference_mode()
def generate(model, tok, prompt: str, max_new_tokens: int) -> str:
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to(model.device)
    mask = enc["attention_mask"].to(model.device)
    out = model.generate(
        input_ids=ids,
        attention_mask=mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def normalize_line(text: str) -> str:
    lines = [x.strip(" `\t") for x in text.splitlines() if x.strip()]
    cand = next((x for x in lines if re.search(r"arr\[\d+\]\s*=", x)), "")
    return re.sub(r"\s+", " ", cand.strip().rstrip(";"))


def parse_list(text: str) -> list[int] | None:
    for match in re.findall(r"\[[^\[\]\n]*\]", text):
        try:
            value = ast.literal_eval(match)
        except Exception:
            continue
        if isinstance(value, list) and all(type(x) is int for x in value):
            return value
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260823)
    p.add_argument("--context-tokens", type=int, default=8192)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--outdir", default="artifacts/g0")
    a = p.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()

    recs = []
    record_path = outdir / "records.jsonl"
    record_path.unlink(missing_ok=True)

    for sid in tqdm(range(a.n), desc="SemTrace paired G0"):
        prog = build_target_program(random.Random(a.seed + sid), sid, a.steps)
        pair = make_context_pair(tok, prog.code, random.Random(a.seed * 1000003 + sid), a.context_tokens)
        contexts = {"start": pair.start, "middle": pair.middle}
        row = {
            "sample_id": sid,
            "input_x": prog.input_x,
            "expected_output": prog.output,
            "lexical_line": prog.lexical_line,
            "trace": prog.trace,
            "context_contract": {
                "start_tokens": pair.start_tokens,
                "middle_tokens": pair.middle_tokens,
                "token_length_delta": abs(pair.start_tokens - pair.middle_tokens),
                "start_target_center_fraction": pair.start_target_center_fraction,
                "middle_target_center_fraction": pair.middle_target_center_fraction,
                "distractor_sha256": pair.distractor_sha256,
                "local_neighbor_sha256": pair.local_neighbor_sha256,
                "local_neighbors_preserved": True,
            },
            "conditions": {},
        }
        idx = re.search(r"arr\[(\d+)\]", prog.lexical_line).group(1)
        for pos, ctx in contexts.items():
            semantic_prompt = chat(
                tok,
                f"{ctx}\n\nEvaluate target_{sid}({prog.input_x}) exactly. Return ONLY the resulting Python list of integers.",
            )
            lexical_prompt = chat(
                tok,
                f"{ctx}\n\nInside target_{sid}, copy exactly the assignment line whose left-hand side is arr[{idx}]. Return ONLY that one assignment line.",
            )
            semantic_text = generate(model, tok, semantic_prompt, 64)
            lexical_text = generate(model, tok, lexical_prompt, 48)
            parsed = parse_list(semantic_text)
            normalized_lex = normalize_line(lexical_text)
            row["conditions"][pos] = {
                "semantic_output": semantic_text,
                "semantic_parsed": parsed,
                "semantic_valid": parsed is not None and len(parsed) == len(prog.output),
                "semantic_correct": parsed == prog.output,
                "lexical_output": lexical_text,
                "lexical_valid": bool(normalized_lex),
                "lexical_correct": normalized_lex == normalize_line(prog.lexical_line),
            }

        c = row["conditions"]
        contract = row["context_contract"]
        row["context_contract_ok"] = (
            contract["local_neighbors_preserved"]
            and contract["token_length_delta"] <= 16
            and contract["start_target_center_fraction"] <= 0.12
            and 0.40 <= contract["middle_target_center_fraction"] <= 0.60
        )
        row["eligible"] = bool(
            row["context_contract_ok"]
            and c["start"]["semantic_correct"]
            and c["start"]["lexical_correct"]
            and c["middle"]["lexical_correct"]
            and c["middle"]["semantic_valid"]
        )
        row["critical_cell"] = bool(row["eligible"] and not c["middle"]["semantic_correct"])
        recs.append(row)
        with record_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(recs)
    eligible = [r for r in recs if r["eligible"]]
    critical = [r for r in recs if r["critical_cell"]]

    def acc(pos: str, key: str) -> float:
        return sum(bool(r["conditions"][pos][key]) for r in recs) / max(1, n)

    start_sem = acc("start", "semantic_correct")
    middle_sem = acc("middle", "semantic_correct")
    start_lex = acc("start", "lexical_correct")
    middle_lex = acc("middle", "lexical_correct")
    middle_invalid = 1.0 - acc("middle", "semantic_valid")
    critical_rate = len(critical) / max(1, len(eligible))
    contract_rate = sum(r["context_contract_ok"] for r in recs) / max(1, n)

    gate = {
        "context_contract_rate_eq_1": contract_rate == 1.0,
        "start_semantic_acc_ge_0.50": start_sem >= 0.50,
        "start_lexical_acc_ge_0.80": start_lex >= 0.80,
        "middle_lexical_acc_ge_0.80": middle_lex >= 0.80,
        "middle_semantic_invalid_rate_le_0.10": middle_invalid <= 0.10,
        "semantic_drop_ge_0.15": start_sem - middle_sem >= 0.15,
        "critical_count_ge_16": len(critical) >= 16,
        "critical_rate_ge_0.20": critical_rate >= 0.20,
    }
    summary = {
        "model": a.model,
        "n": n,
        "start_semantic_accuracy": start_sem,
        "middle_semantic_accuracy": middle_sem,
        "semantic_drop": start_sem - middle_sem,
        "start_lexical_accuracy": start_lex,
        "middle_lexical_accuracy": middle_lex,
        "middle_semantic_invalid_rate": middle_invalid,
        "context_contract_rate": contract_rate,
        "eligible_count": len(eligible),
        "critical_count": len(critical),
        "critical_rate_among_eligible": critical_rate,
        "gate": gate,
        "verdict": "GO_MECHANISM" if all(gate.values()) else "STOP_MECHANISM_OBJECT",
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
