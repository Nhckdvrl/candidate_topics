#!/usr/bin/env python3
"""Frozen G0 for Topic 26: temporal-scope interference and reinstatement.

The script deliberately separates panel construction from model inference.
Panel construction uses only released ChronoScope metadata and deterministic
selection; no model output is consulted when deciding eligibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SEED = 20260825
TARGET_N = 512
PRESENT_YEAR = 2025
MODEL = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers factual questions.\n"
    "Important:\n"
    "- Use the conversation context.\n"
    "- If a prior turn establishes a time period, keep that same time scope unless it is explicitly changed.\n"
    "- Answer with only the entity or value, no extra explanation.\n"
)

PID_SURFACE = {
    "P35": "head of state", "P6": "head of government", "P39": "position held",
    "P102": "political party", "P108": "employer", "P463": "membership",
    "P127": "owner", "P169": "chief executive officer", "P488": "chairperson",
    "P69": "educational institution", "P106": "occupation", "P101": "field of work",
    "P131": "administrative entity", "P17": "country", "P276": "location",
    "P54": "sports team", "P286": "head coach", "P31": "type", "P279": "subclass",
}

_WS = re.compile(r"\s+")

def norm(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.strip().lower().replace("’", "'").replace("“", '"').replace("”", '"')
    return _WS.sub(" ", s)

def relaxed_match(pred: str, gold: str) -> bool:
    p = re.sub(r"[^\w\s]", "", norm(pred).split("\n", 1)[0]).strip()
    g = re.sub(r"[^\w\s]", "", norm(gold)).strip()
    return bool(p and g and (p == g or g in p or p in g))

def read_jsonl(path: Path) -> List[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def present_value(turn: dict) -> Optional[str]:
    # Current public Stage-3 writer stores this at TURN level.
    v = turn.get("present_day_answer")
    return str(v) if v not in (None, "") else None

@dataclass(frozen=True)
class Fact:
    subject: str
    pid: str
    surface: str
    answer: str
    present: str
    stable: bool
    year: Optional[int]

@dataclass
class PanelItem:
    item_id: str
    chain_id: str
    year: int
    subject: str
    target_pid: str
    anchor_question: str
    anchor_answer: str
    probe_question: str
    historical_answer: str
    present_answer: str
    stable_fact: Dict[str, Any]
    neutral_facts: List[Dict[str, Any]]


def iter_facts(chains: Sequence[dict]) -> Iterable[Fact]:
    for c in chains:
        for t in c.get("turns", []) or []:
            subject = t.get("subject_label")
            pid = t.get("pid")
            ans = t.get("answer")
            pres = present_value(t)
            if not subject or not pid or ans in (None, "") or pres in (None, ""):
                continue
            yield Fact(
                subject=str(subject), pid=str(pid), surface=PID_SURFACE.get(str(pid), str(pid)),
                answer=str(ans), present=str(pres), stable=(norm(ans) == norm(pres)), year=t.get("year"),
            )


def _stable_key(f: Fact) -> Tuple[str, str, str]:
    return (norm(f.subject), f.pid, norm(f.answer))


def build_panel(chains: Sequence[dict], n: int = TARGET_N, seed: int = SEED) -> Tuple[List[PanelItem], dict]:
    facts = list(iter_facts(chains))
    stable_by_subject: Dict[str, List[Fact]] = defaultdict(list)
    stable_by_pid: Dict[str, List[Fact]] = defaultdict(list)
    seen = set()
    for f in facts:
        if not f.stable:
            continue
        k = _stable_key(f)
        if k in seen:
            continue
        seen.add(k)
        stable_by_subject[norm(f.subject)].append(f)
        stable_by_pid[f.pid].append(f)

    candidates: List[PanelItem] = []
    skip = defaultdict(int)
    for c in chains:
        turns = c.get("turns", []) or []
        if c.get("family") != "carryover" or c.get("truth_type") != "temporal" or len(turns) != 2:
            skip["not_target_family"] += 1
            continue
        a, p = turns
        pa = present_value(p)
        if not pa or norm(pa) == norm(p.get("answer")):
            skip["probe_not_drift_eligible"] += 1
            continue
        subject = p.get("subject_label") or a.get("subject_label")
        target_pid = p.get("pid")
        year = a.get("year") or c.get("snapshot_year")
        if not subject or not target_pid or not isinstance(year, int):
            skip["missing_metadata"] += 1
            continue

        sf = [f for f in stable_by_subject.get(norm(subject), []) if f.pid != target_pid]
        if not sf:
            skip["no_same_entity_stable_fact"] += 1
            continue
        # deterministic donor: shortest lexical form first, then PID/value.
        sf.sort(key=lambda f: (len(f.subject) + len(f.answer), f.pid, norm(f.answer)))
        stable = sf[0]

        neutral_pool = [f for f in stable_by_pid.get(stable.pid, []) if norm(f.subject) != norm(subject)]
        if len(neutral_pool) < 4:
            skip["insufficient_other_entity_controls"] += 1
            continue
        # Match answer/entity lexical length to reduce prompt-length differences.
        neutral_pool.sort(key=lambda f: (abs(len(f.subject) - len(subject)) + abs(len(f.answer) - len(stable.answer)), norm(f.subject)))
        neutrals = neutral_pool[:4]

        cid = str(c.get("chain_id") or "")
        raw_id = f"{cid}|{subject}|{target_pid}|{year}"
        item_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]
        candidates.append(PanelItem(
            item_id=item_id, chain_id=cid, year=year, subject=str(subject), target_pid=str(target_pid),
            anchor_question=str(a["question"]), anchor_answer=str(a["answer"]),
            probe_question=str(p["question"]), historical_answer=str(p["answer"]), present_answer=str(pa),
            stable_fact=asdict(stable), neutral_facts=[asdict(x) for x in neutrals],
        ))

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:n]
    report = {
        "seed": seed, "target_n": n, "eligible_n": len(candidates), "selected_n": len(selected),
        "hard_gate_pass": len(selected) >= n, "skip_counts": dict(skip),
    }
    return selected, report


def fact_sentence(f: Dict[str, Any], present_cue: bool = False) -> str:
    subj, surface, answer = f["subject"], f["surface"], f["answer"]
    if present_cue:
        # The fact is intentionally temporally STABLE. The only new temporal signal is the present-day cue.
        return f"As of {PRESENT_YEAR}, {subj}'s {surface} is {answer}."
    return f"In a separate note, {subj}'s {surface} is {answer}."


def aside_exchange(sentence: str) -> List[Dict[str, str]]:
    return [
        {"role": "user", "content": "This is a separate aside and does not change the time frame of the main discussion. " + sentence},
        {"role": "assistant", "content": "Acknowledged."},
    ]


def build_messages(item: PanelItem, condition: str) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item.anchor_question},
        {"role": "assistant", "content": item.anchor_answer},
    ]
    if condition == "baseline":
        pass
    elif condition.startswith("neutral_"):
        k = int(condition.split("_", 1)[1])
        if k not in {1, 2, 4}:
            raise ValueError(condition)
        for f in item.neutral_facts[:k]:
            msgs.extend(aside_exchange(fact_sentence(f, present_cue=False)))
    elif condition == "same_entity_semantic":
        msgs.extend(aside_exchange(fact_sentence(item.stable_fact, present_cue=False)))
    elif condition in {"bounded_present", "bounded_present_reinstate"}:
        msgs.extend(aside_exchange(fact_sentence(item.stable_fact, present_cue=True)))
        if condition == "bounded_present_reinstate":
            msgs.extend([
                {"role": "user", "content": "Return to the earlier time frame from the original question."},
                {"role": "assistant", "content": "Understood."},
            ])
    else:
        raise ValueError(f"unknown condition: {condition}")
    msgs.append({"role": "user", "content": item.probe_question})
    return msgs

CONDITIONS = [
    "baseline", "neutral_1", "neutral_2", "neutral_4",
    "same_entity_semantic", "bounded_present", "bounded_present_reinstate",
]


def write_panel(panel: Sequence[PanelItem], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in panel:
            row = asdict(item)
            row["messages"] = {c: build_messages(item, c) for c in CONDITIONS}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def bootstrap_delta(rows: Sequence[dict], a: str, b: str, n_boot: int = 5000, seed: int = SEED) -> dict:
    # delta = accuracy(a) - accuracy(b), paired by item.
    pairs = [(int(r["conditions"][a]["correct"]), int(r["conditions"][b]["correct"])) for r in rows]
    if not pairs:
        return {"n": 0, "delta": None, "ci95": [None, None]}
    diffs = [x-y for x,y in pairs]
    delta = sum(diffs) / len(diffs)
    rng = random.Random(seed + sum(map(ord, a+b)))
    boots = []
    n = len(diffs)
    for _ in range(n_boot):
        boots.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    lo = boots[int(0.025 * (n_boot-1))]
    hi = boots[int(0.975 * (n_boot-1))]
    return {"n": n, "delta": delta, "ci95": [lo, hi]}


def summarize(rows: Sequence[dict]) -> dict:
    acc = {}
    drift = {}
    for c in CONDITIONS:
        vals = [int(r["conditions"][c]["correct"]) for r in rows]
        dvals = [int(r["conditions"][c]["present_drift"]) for r in rows]
        acc[c] = sum(vals)/len(vals) if vals else None
        drift[c] = sum(dvals)/len(dvals) if dvals else None
    contrasts = {
        "neutral_decay_1_to_4": bootstrap_delta(rows, "neutral_1", "neutral_4"),
        "same_entity_penalty": bootstrap_delta(rows, "neutral_1", "same_entity_semantic"),
        "present_cue_penalty": bootstrap_delta(rows, "same_entity_semantic", "bounded_present"),
        "reinstatement_gain": bootstrap_delta(rows, "bounded_present_reinstate", "bounded_present"),
    }
    supported = []
    for name, x in contrasts.items():
        if x["delta"] is not None and x["delta"] >= 0.05 and x["ci95"][0] > 0:
            supported.append(name)
    verdict = "+".join(supported).upper() if supported else "NO_LARGE_CONTROLLED_EFFECT"
    return {"n": len(rows), "accuracy": acc, "present_drift_rate": drift, "contrasts": contrasts, "verdict": verdict}


def messages_to_prompt(tokenizer, messages: List[Dict[str,str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    chunks = []
    for m in messages:
        chunks.append(f"{m['role'].title()}: {m['content']}")
    chunks.append("Assistant:")
    return "\n".join(chunks)


def run_model(panel_path: Path, out_path: Path, model_name: str, batch_size: int, max_new_tokens: int) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    panel_rows = read_jsonl(panel_path)
    if len(panel_rows) < TARGET_N:
        raise SystemExit(f"HARD STOP: panel has {len(panel_rows)} items; frozen G0 requires {TARGET_N}.")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()

    jobs = []
    for i, row in enumerate(panel_rows):
        for c in CONDITIONS:
            jobs.append((i, c, messages_to_prompt(tok, row["messages"][c])))

    preds: Dict[int, Dict[str, str]] = defaultdict(dict)
    prompt_tokens: Dict[int, Dict[str, int]] = defaultdict(dict)
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start:start+batch_size]
        prompts = [x[2] for x in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
        input_len = enc.input_ids.shape[1]
        with torch.inference_mode():
            gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
        texts = tok.batch_decode(gen[:, input_len:], skip_special_tokens=True)
        lengths = enc.attention_mask.sum(dim=1).tolist()
        for (idx, c, _), text, plen in zip(batch, texts, lengths):
            preds[idx][c] = text.strip()
            prompt_tokens[idx][c] = int(plen)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_rows = []
    for i, row in enumerate(panel_rows):
        cond = {}
        for c in CONDITIONS:
            pred = preds[i][c]
            ok = relaxed_match(pred, row["historical_answer"])
            pd = (not ok) and relaxed_match(pred, row["present_answer"])
            cond[c] = {"prediction": pred, "correct": ok, "present_drift": pd,
                       "prompt_tokens": prompt_tokens[i][c]}
        result_rows.append({"item_id": row["item_id"], "chain_id": row["chain_id"], "conditions": cond})
    with out_path.open("w", encoding="utf-8") as f:
        for r in result_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Token-balance audit for the primary neutral/semantic/present contrast.
    gaps = []
    for r in result_rows:
        lens = [r["conditions"][c]["prompt_tokens"] for c in ["neutral_1", "same_entity_semantic", "bounded_present"]]
        gaps.append(max(lens)-min(lens))
    if max(gaps) > 16:
        raise SystemExit(f"MEASUREMENT STOP: max primary prompt-token gap={max(gaps)} > frozen 16-token ceiling")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--panel", type=Path, default=Path("results/g0_panel.jsonl"))
    p.add_argument("--report", type=Path, default=Path("results/g0_preflight.json"))
    r = sub.add_parser("run")
    r.add_argument("--panel", type=Path, required=True)
    r.add_argument("--out", type=Path, default=Path("results/g0_raw.jsonl"))
    r.add_argument("--model", default=MODEL)
    r.add_argument("--batch-size", type=int, default=16)
    r.add_argument("--max-new-tokens", type=int, default=24)
    s = sub.add_parser("summarize")
    s.add_argument("--raw", type=Path, required=True)
    s.add_argument("--out", type=Path, default=Path("results/g0_summary.json"))
    args = ap.parse_args()

    if args.cmd == "prepare":
        chains = read_jsonl(args.data)
        panel, report = build_panel(chains)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not report["hard_gate_pass"]:
            raise SystemExit(f"HARD STOP: only {report['selected_n']} exact eligible items; need {TARGET_N}.")
        write_panel(panel, args.panel)
        print(json.dumps(report, indent=2))
    elif args.cmd == "run":
        run_model(args.panel, args.out, args.model, args.batch_size, args.max_new_tokens)
    else:
        rows = read_jsonl(args.raw)
        summary = summarize(rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
