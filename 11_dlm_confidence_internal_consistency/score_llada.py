#!/usr/bin/env python3
"""Score Topic-11 v3 with LLaDA final-forward confidence.

Primary identification is retroactive: internal consistency is changed only in a
suffix *after* the trajectory, while confidence is measured on unchanged
trajectory result tokens before that suffix.

No diffusion generation is needed. The protocol matches the seed paper's final
teacher-forced forward score and includes quantitative arithmetic and semantic-
alias prerequisites before the factorial can be interpreted.
"""
from __future__ import annotations

import argparse, json, random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
import torch


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def hamming(a: list[int], b: list[int]) -> int:
    if len(a) != len(b):
        return 10**9
    return sum(x != y for x, y in zip(a, b))


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False,
    )
    return list(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def encode_with_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = [tuple(map(int, x)) for x in enc["offset_mapping"]]
    if len(ids) != len(offsets):
        raise RuntimeError("token ids / offset mapping length mismatch")
    return ids, offsets


def token_positions_for_char_spans(offsets, spans) -> list[int]:
    spans = [(int(s), int(e)) for s, e in spans]
    out = []
    for i, (s, e) in enumerate(offsets):
        s, e = int(s), int(e)
        if e <= s:
            continue
        if any(s < be and e > bs for bs, be in spans):
            out.append(i)
    return out


def encode_sample(tokenizer, row: dict[str, Any]) -> dict[str, Any]:
    prompt_ids = encode_prompt(tokenizer, row["prompt"])
    traj_ids, traj_offsets = encode_with_offsets(tokenizer, row["trajectory_text"])
    check_ids = list(tokenizer(row["check_text"], add_special_tokens=False)["input_ids"])
    groups_rel = [token_positions_for_char_spans(traj_offsets, [span]) for span in row["result_char_spans"]]
    if len(groups_rel) != 4 or any(not g for g in groups_rel):
        raise RuntimeError(f"invalid result spans pair={row['pair_id']} cell={row['cell']}")
    p, t = len(prompt_ids), len(traj_ids)
    groups_abs = [[p+i for i in g] for g in groups_rel]
    return {
        **row,
        "prompt_ids": prompt_ids,
        "trajectory_ids": traj_ids,
        "check_ids": check_ids,
        "input_ids": prompt_ids + traj_ids + check_ids,
        "prompt_len": p,
        "trajectory_len": t,
        "check_len": len(check_ids),
        "output_len": t + len(check_ids),
        "result_position_groups": groups_abs,
        "result_positions": sorted({i for g in groups_abs for i in g}),
    }


def eligible_orientation(cells: dict[str, dict[str, Any]], max_intervention_tokens: int) -> tuple[bool, str]:
    if set(cells) != {"CC", "IC", "CW", "IW"}:
        return False, "missing_cells"
    cc, ic, cw, iw = (cells[k] for k in ("CC", "IC", "CW", "IW"))
    if cc["prompt_ids"] != ic["prompt_ids"] or cw["prompt_ids"] != iw["prompt_ids"]:
        return False, "prompt_not_fixed_within_correctness"
    if not (cc["trajectory_ids"] == ic["trajectory_ids"] == cw["trajectory_ids"] == iw["trajectory_ids"]):
        return False, "trajectory_not_identical"
    if cc["check_ids"] != cw["check_ids"] or ic["check_ids"] != iw["check_ids"]:
        return False, "check_not_fixed_within_consistency"
    dp = hamming(cc["prompt_ids"], cw["prompt_ids"])
    dc = hamming(cc["check_ids"], ic["check_ids"])
    if dp < 1 or dp > max_intervention_tokens:
        return False, f"prompt_token_diff={dp}"
    if dc < 1 or dc > max_intervention_tokens:
        return False, f"check_token_diff={dc}"
    if len({len(x["input_ids"]) for x in cells.values()}) != 1:
        return False, "total_length_mismatch"
    if len({x["prompt_len"] for x in cells.values()}) != 1:
        return False, "prompt_length_mismatch"
    if len({x["check_len"] for x in cells.values()}) != 1:
        return False, "check_length_mismatch"
    if not (cc["result_position_groups"] == ic["result_position_groups"] ==
            cw["result_position_groups"] == iw["result_position_groups"]):
        return False, "result_positions_not_identical"
    return True, "ok"


def select_eligible_pairs(rows: list[dict[str, Any]], max_intervention_tokens: int):
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        grouped[(int(r["pair_id"]), int(r["orientation"]))][r["cell"]] = r
    status = {}; reasons: dict[str, int] = defaultdict(int)
    for key, cells in grouped.items():
        status[key] = eligible_orientation(cells, max_intervention_tokens)
        reasons[status[key][1]] += 1
    pids = sorted({p for p, _ in grouped})
    pair_ok = {p: all(status.get((p, o), (False, "missing"))[0] for o in (0, 1)) for p in pids}
    kept = [r for r in rows if pair_ok.get(int(r["pair_id"]), False)]
    reasons["eligible_pairs"] = sum(pair_ok.values()); reasons["total_pairs"] = len(pair_ok)
    return kept, dict(reasons)


def exact_length_batches(records: list[dict[str, Any]], batch_size: int):
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        buckets[len(r["input_ids"])].append(r)
    for n in sorted(buckets):
        bucket = buckets[n]
        for i in range(0, len(bucket), batch_size):
            batch = bucket[i:i+batch_size]
            assert len({len(x["input_ids"]) for x in batch}) == 1
            yield batch


def observed_token_probs(model, input_ids: torch.Tensor) -> torch.Tensor:
    """P(x_i | fully observed sequence) at each position, per seed-paper final forward."""
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits
        target = torch.gather(logits, -1, input_ids.unsqueeze(-1)).squeeze(-1).float()
        log_z = torch.logsumexp(logits.float(), dim=-1)
        return torch.exp(target - log_z)


def mean_at(probs: torch.Tensor, row: int, positions) -> float:
    pos = list(positions)
    if not pos:
        raise RuntimeError("empty scoring position set")
    idx = torch.tensor(pos, dtype=torch.long, device=probs.device)
    return float(probs[row].index_select(0, idx).mean().item())


def score_batch(model, batch: list[dict[str, Any]], device: str) -> list[dict[str, Any]]:
    ids = torch.tensor([x["input_ids"] for x in batch], dtype=torch.long, device=device)
    probs = observed_token_probs(model, ids)
    out = []
    for i, r in enumerate(batch):
        p = int(r["prompt_len"]); t = int(r["trajectory_len"]); n = len(r["input_ids"])
        groups = r["result_position_groups"]
        middle = groups[1] + groups[2]
        all_results = [x for g in groups for x in g]
        out.append({
            "pair_id": int(r["pair_id"]), "orientation": int(r["orientation"]), "cell": r["cell"],
            "internal_consistent": bool(r["internal_consistent"]),
            "externally_correct": bool(r["externally_correct"]),
            "branch_anchor": int(r["branch_anchor"]), "alternate_anchor": int(r["alternate_anchor"]),
            "prompt_anchor": int(r["prompt_anchor"]), "check_anchor": int(r["check_anchor"]),
            "reported_final": int(r["reported_final"]), "true_final": int(r["true_final"]),
            "confidence_result_middle": mean_at(probs, i, middle),
            "confidence_result_first": mean_at(probs, i, groups[0]),
            "confidence_result_final": mean_at(probs, i, groups[-1]),
            "confidence_result_all": mean_at(probs, i, all_results),
            "confidence_trajectory": mean_at(probs, i, range(p, p+t)),
            "confidence_full": mean_at(probs, i, range(p, n)),
            "confidence_check": mean_at(probs, i, range(p+t, n)),
            "prompt_len": p, "trajectory_len": t, "check_len": int(r["check_len"]), "input_len": n,
        })
    return out


def response_record(tokenizer, prompt: str, response: str, result_text: str):
    pids = encode_prompt(tokenizer, prompt)
    rids, offsets = encode_with_offsets(tokenizer, response)
    start = response.rfind(result_text)
    if start < 0:
        raise AssertionError("result missing")
    rel = token_positions_for_char_spans(offsets, [[start, start+len(result_text)]])
    if not rel:
        return None
    return {"input_ids": pids+rids, "result_positions": [len(pids)+i for i in rel]}


def build_protocol_probe(tokenizer, n_pairs: int, seed: int):
    rng = random.Random(seed); prompt = "Read the arithmetic equation below."; rows = []; attempts = 0
    while len(rows) < n_pairs and attempts < n_pairs*1000:
        attempts += 1
        # LLaDA's tokenizer is digit-level for multi-digit numerals. Keep the
        # prerequisite target to one digit so the locked one-token probe gate
        # tests arithmetic discrimination rather than tokenizer fragmentation.
        a, b = rng.randint(1, 4), rng.randint(1, 4); correct = a+b
        wrong = correct + rng.choice([-2, -1, 1, 2])
        if wrong <= 0 or wrong > 9: continue
        prefix = f"{a} + {b} = "
        c = response_record(tokenizer, prompt, prefix+str(correct), str(correct))
        w = response_record(tokenizer, prompt, prefix+str(wrong), str(wrong))
        if c is None or w is None: continue
        if len(c["input_ids"]) != len(w["input_ids"]) or hamming(c["input_ids"], w["input_ids"]) != 1: continue
        if len(c["result_positions"]) != 1 or len(w["result_positions"]) != 1: continue
        rows.append({"probe_id": len(rows), "a": a, "b": b, "correct_result": correct,
                     "wrong_result": wrong, "correct": c, "wrong": w})
    if len(rows) != n_pairs: raise RuntimeError(f"only built {len(rows)}/{n_pairs} protocol probes")
    return rows


def build_alias_probe(tokenizer, n_pairs: int, seed: int):
    """Verify that arithmetic aliases are understood on an unchanged target token."""
    rng = random.Random(seed); rows = []; attempts = 0; prompt = "Read the arithmetic equality below."
    while len(rows) < n_pairs and attempts < n_pairs*2000:
        attempts += 1; target = rng.randint(4, 9); base = rng.choice([1, 2, 3])
        good_r = target-base
        if good_r <= 0: continue
        bad_r = good_r + rng.choice([-7,-6,-5,-4,-3,3,4,5,6,7])
        if bad_r <= 0: continue
        g = response_record(tokenizer, prompt, f"Equality: {base} + {good_r} = {target}", str(target))
        b = response_record(tokenizer, prompt, f"Equality: {base} + {bad_r} = {target}", str(target))
        if g is None or b is None: continue
        if len(g["input_ids"]) != len(b["input_ids"]) or hamming(g["input_ids"], b["input_ids"]) != 1: continue
        if g["result_positions"] != b["result_positions"] or len(g["result_positions"]) != 1: continue
        rows.append({"probe_id": len(rows), "target": target, "base": base,
                     "correct_residual": good_r, "wrong_residual": bad_r, "correct": g, "wrong": b})
    if len(rows) != n_pairs: raise RuntimeError(f"only built {len(rows)}/{n_pairs} semantic-alias probes")
    return rows


def _score_paired_probe(model, pairs, batch_size: int, device: str, probe_type: str):
    flat = []
    for q in pairs:
        for label in ("correct", "wrong"):
            r = dict(q[label]); r.update({"probe_id": q["probe_id"], "label": label}); flat.append(r)
    scores = {}
    for batch in exact_length_batches(flat, batch_size):
        ids = torch.tensor([x["input_ids"] for x in batch], dtype=torch.long, device=device)
        probs = observed_token_probs(model, ids)
        for i, r in enumerate(batch):
            scores[(int(r["probe_id"]), r["label"])] = mean_at(probs, i, r["result_positions"])
    out = []
    for q in pairs:
        pid = int(q["probe_id"]); c = scores[(pid,"correct")]; w = scores[(pid,"wrong")]
        meta = {k:v for k,v in q.items() if k not in ("correct","wrong")}
        out.append({**meta, "probe_type": probe_type, "confidence_correct": c,
                    "confidence_wrong": w, "gap": c-w})
    return out


def score_protocol_probes(model, tokenizer, n_pairs: int, seed: int, batch_size: int, device: str):
    arithmetic = build_protocol_probe(tokenizer, n_pairs, seed)
    aliases = build_alias_probe(tokenizer, n_pairs, seed + 101)
    return (_score_paired_probe(model, arithmetic, batch_size, device, "arithmetic_result") +
            _score_paired_probe(model, aliases, batch_size, device, "semantic_alias"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True); p.add_argument("--output", type=Path)
    p.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct"); p.add_argument("--revision", default="main")
    p.add_argument("--batch-size", type=int, default=8); p.add_argument("--dtype", choices=["bfloat16","float16","float32"], default="bfloat16")
    p.add_argument("--device", default="cuda"); p.add_argument("--shard-id", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--min-pairs", type=int, default=128); p.add_argument("--max-intervention-tokens", type=int, default=1)
    p.add_argument("--audit-only", action="store_true"); p.add_argument("--protocol-probe-output", type=Path)
    p.add_argument("--protocol-probe-pairs", type=int, default=100); p.add_argument("--protocol-probe-seed", type=int, default=20260823)
    p.add_argument("--runtime-output", type=Path); a = p.parse_args()
    if not (0 <= a.shard_id < a.num_shards): raise SystemExit("require 0 <= shard-id < num-shards")
    if not a.audit_only and a.output is None: raise SystemExit("--output required unless --audit-only")

    raw = read_jsonl(a.input)
    from transformers import AutoModel, AutoTokenizer, __version__ as transformers_version
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision, trust_remote_code=True)
    encoded = [encode_sample(tokenizer, r) for r in raw]
    eligible, stats = select_eligible_pairs(encoded, a.max_intervention_tokens)
    n_pairs = int(stats["eligible_pairs"])
    probe_preview = build_protocol_probe(tokenizer, a.protocol_probe_pairs, a.protocol_probe_seed)
    alias_preview = build_alias_probe(tokenizer, a.protocol_probe_pairs, a.protocol_probe_seed + 101)
    audit = {"factorial_tokenization": stats, "protocol_probe_pairs_each": len(probe_preview),
             "semantic_alias_probe_pairs": len(alias_preview), "model": a.model, "revision_requested": a.revision}
    print("tokenization audit:", json.dumps(audit, sort_keys=True))
    if n_pairs < a.min_pairs: raise SystemExit(f"Only {n_pairs} eligible mirrored pairs; need >= {a.min_pairs}.")
    if a.audit_only: return

    eligible = [r for r in eligible if int(r["pair_id"]) % a.num_shards == a.shard_id]
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[a.dtype]
    model = AutoModel.from_pretrained(a.model, revision=a.revision, trust_remote_code=True, torch_dtype=dtype).to(a.device).eval()
    if a.runtime_output is not None:
        runtime = {"model": a.model, "revision_requested": a.revision,
                   "model_commit": getattr(model.config, "_commit_hash", None),
                   "tokenizer_commit": getattr(tokenizer, "_commit_hash", None),
                   "torch": torch.__version__, "transformers": transformers_version,
                   "dtype": a.dtype, "device": a.device, "shard_id": a.shard_id,
                   "num_shards": a.num_shards, "tokenization_audit": stats}
        a.runtime_output.parent.mkdir(parents=True, exist_ok=True)
        a.runtime_output.write_text(json.dumps(runtime, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    if a.protocol_probe_output is not None:
        rows = score_protocol_probes(model, tokenizer, a.protocol_probe_pairs, a.protocol_probe_seed, a.batch_size, a.device)
        write_jsonl(a.protocol_probe_output, rows); print(f"scored {len(rows)} protocol probes -> {a.protocol_probe_output}")
    assert a.output is not None; a.output.parent.mkdir(parents=True, exist_ok=True); n = 0
    with a.output.open("w", encoding="utf-8") as f:
        for batch in exact_length_batches(eligible, a.batch_size):
            for r in score_batch(model, batch, a.device):
                f.write(json.dumps(r, sort_keys=True)+"\n"); n += 1
    print(f"scored {n} factorial samples on shard {a.shard_id}/{a.num_shards} -> {a.output}")


if __name__ == "__main__":
    main()
