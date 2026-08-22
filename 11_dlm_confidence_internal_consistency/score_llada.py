#!/usr/bin/env python3
"""Score Topic-11 trajectories with the final-forward LLaDA confidence protocol.

The seed paper's primary DLLM confidence is obtained by one teacher-forced
forward pass on the fully specified sequence and by reading the probability of
each observed output token at its own position. This script computes:

* confidence_full: all prescribed output tokens (paper-compatible score)
* confidence_tail: all unchanged downstream continuation tokens
* confidence_result: unchanged downstream arithmetic result tokens only
* confidence_announcement: manipulated announcement line (diagnostic only)

It also runs a cheap arithmetic positive-control probe on shard 0. If the same
scorer cannot assign higher result-token confidence to correct equations than to
single-token incorrect equations, the factorial result must not be interpreted.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def hamming(a: list[int], b: list[int]) -> int:
    if len(a) != len(b):
        return 10**9
    return sum(x != y for x, y in zip(a, b))


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    # The chat template already includes BOS/header special tokens.
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


def token_positions_for_char_spans(
    offsets: Iterable[tuple[int, int] | list[int]], spans: Iterable[tuple[int, int] | list[int]]
) -> list[int]:
    spans = [(int(s), int(e)) for s, e in spans]
    out: list[int] = []
    for i, pair in enumerate(offsets):
        s, e = int(pair[0]), int(pair[1])
        if e <= s:  # special/empty token span
            continue
        if any(s < span_e and e > span_s for span_s, span_e in spans):
            out.append(i)
    return out


def encode_text_with_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = list(enc["input_ids"])
    offsets = [tuple(map(int, x)) for x in enc["offset_mapping"]]
    if len(ids) != len(offsets):
        raise RuntimeError("Tokenizer returned mismatched ids/offsets")
    return ids, offsets


def encode_sample(tokenizer, row: dict[str, Any]) -> dict[str, Any]:
    prompt_ids = encode_prompt(tokenizer, row["prompt"])
    announce_ids = tokenizer(row["announcement_text"], add_special_tokens=False)["input_ids"]
    tail_ids, tail_offsets = encode_text_with_offsets(tokenizer, row["continuation_text"])
    result_groups_rel = [token_positions_for_char_spans(tail_offsets, [span]) for span in row["result_char_spans"]]
    if any(not g for g in result_groups_rel):
        raise RuntimeError(f"A result span has no tokens for pair={row['pair_id']} cell={row['cell']}")
    result_rel = sorted({i for group in result_groups_rel for i in group})

    input_ids = prompt_ids + announce_ids + tail_ids
    p = len(prompt_ids)
    a = len(announce_ids)
    result_abs = [p + a + i for i in result_rel]
    result_groups_abs = [[p + a + i for i in group] for group in result_groups_rel]
    return {
        **row,
        "prompt_ids": list(prompt_ids),
        "announce_ids": list(announce_ids),
        "tail_ids": list(tail_ids),
        "input_ids": list(input_ids),
        "prompt_len": p,
        "announce_len": a,
        "tail_len": len(tail_ids),
        "output_len": a + len(tail_ids),
        "result_positions": result_abs,
        "result_position_groups": result_groups_abs,
        "result_token_count": len(result_abs),
    }


def eligible_orientation(cells: dict[str, dict[str, Any]], max_intervention_tokens: int) -> tuple[bool, str]:
    if set(cells) != {"CC", "IC", "CW", "IW"}:
        return False, "missing_cells"
    cc, ic, cw, iw = (cells[k] for k in ("CC", "IC", "CW", "IW"))

    if cc["prompt_ids"] != ic["prompt_ids"] or cw["prompt_ids"] != iw["prompt_ids"]:
        return False, "prompt_not_fixed_within_correctness"
    if not (cc["tail_ids"] == ic["tail_ids"] == cw["tail_ids"] == iw["tail_ids"]):
        return False, "tail_not_common_across_factorial"
    if cc["announce_ids"] != cw["announce_ids"] or ic["announce_ids"] != iw["announce_ids"]:
        return False, "output_not_fixed_within_correctness"
    if not (cc["result_positions"] == ic["result_positions"] == cw["result_positions"] == iw["result_positions"]):
        return False, "result_token_positions_not_common"
    if not (cc["result_position_groups"] == ic["result_position_groups"] == cw["result_position_groups"] == iw["result_position_groups"]):
        return False, "result_token_groups_not_common"
    if any(len(r["result_position_groups"]) != 4 for r in cells.values()):
        return False, "unexpected_result_group_count"

    d_announce = hamming(cc["announce_ids"], ic["announce_ids"])
    d_prompt = hamming(cc["prompt_ids"], cw["prompt_ids"])
    if d_announce < 1 or d_announce > max_intervention_tokens:
        return False, f"announcement_token_diff={d_announce}"
    if d_prompt < 1 or d_prompt > max_intervention_tokens:
        return False, f"prompt_token_diff={d_prompt}"
    if len({r["output_len"] for r in cells.values()}) != 1:
        return False, "output_length_mismatch"
    if len({len(r["input_ids"]) for r in cells.values()}) != 1:
        return False, "total_length_mismatch"
    if min(r["result_token_count"] for r in cells.values()) < 1:
        return False, "missing_result_tokens"
    return True, "ok"


def select_eligible_pairs(
    rows: list[dict[str, Any]], max_intervention_tokens: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        grouped[(int(r["pair_id"]), int(r["orientation"]))][r["cell"]] = r

    orientation_status: dict[tuple[int, int], tuple[bool, str]] = {}
    reason_counts: dict[str, int] = defaultdict(int)
    for key, cells in grouped.items():
        status = eligible_orientation(cells, max_intervention_tokens)
        orientation_status[key] = status
        reason_counts[status[1]] += 1

    pair_ok: dict[int, bool] = {}
    pair_ids = sorted({p for p, _ in grouped})
    for pair_id in pair_ids:
        pair_ok[pair_id] = all(orientation_status.get((pair_id, o), (False, "missing"))[0] for o in (0, 1))

    kept = [r for r in rows if pair_ok.get(int(r["pair_id"]), False)]
    reason_counts["eligible_pairs"] = sum(pair_ok.values())
    reason_counts["total_pairs"] = len(pair_ok)
    return kept, dict(reason_counts)


def exact_length_batches(records: list[dict[str, Any]], batch_size: int):
    """Batch only equal-length sequences; avoids padding/attention-mask confounds."""
    by_len: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_len[len(record["input_ids"])].append(record)
    for length in sorted(by_len):
        bucket = by_len[length]
        for i in range(0, len(bucket), batch_size):
            batch = bucket[i : i + batch_size]
            assert len({len(r["input_ids"]) for r in batch}) == 1
            yield batch


def _mean_at(probs: torch.Tensor, row_index: int, positions: list[int] | range) -> float:
    pos = list(positions)
    if not pos:
        raise RuntimeError("Cannot average an empty token set")
    idx = torch.tensor(pos, dtype=torch.long, device=probs.device)
    return float(probs[row_index].index_select(0, idx).mean().item())


def forward_observed_token_probs(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Probability assigned to each observed token at its own position."""
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits.float()
        target_logits = torch.gather(logits, -1, input_ids.unsqueeze(-1)).squeeze(-1)
        log_z = torch.logsumexp(logits, dim=-1)
        probs = torch.exp(target_logits - log_z)
    return probs


def score_design_batch(model, batch: list[dict[str, Any]], device: str) -> list[dict[str, Any]]:
    input_ids = torch.tensor([r["input_ids"] for r in batch], dtype=torch.long, device=device)
    probs = forward_observed_token_probs(model, input_ids)
    results: list[dict[str, Any]] = []
    for i, r in enumerate(batch):
        p = int(r["prompt_len"])
        a = int(r["announce_len"])
        n = len(r["input_ids"])
        results.append(
            {
                "pair_id": int(r["pair_id"]),
                "orientation": int(r["orientation"]),
                "cell": r["cell"],
                "internal_consistent": bool(r["internal_consistent"]),
                "externally_correct": bool(r["externally_correct"]),
                "branch_anchor": int(r["branch_anchor"]),
                "alternate_anchor": int(r["alternate_anchor"]),
                "prompt_anchor": int(r["prompt_anchor"]),
                "announced_anchor": int(r["announced_anchor"]),
                "reported_final": int(r["reported_final"]),
                "true_final": int(r["true_final"]),
                "confidence_full": _mean_at(probs, i, range(p, n)),
                "confidence_announcement": _mean_at(probs, i, range(p, p + a)),
                "confidence_tail": _mean_at(probs, i, range(p + a, n)),
                "confidence_result": _mean_at(probs, i, r["result_positions"]),
                "confidence_result_first": _mean_at(probs, i, r["result_position_groups"][0]),
                "confidence_result_late": _mean_at(
                    probs, i, [x for group in r["result_position_groups"][1:] for x in group]
                ),
                "confidence_final": _mean_at(probs, i, r["result_position_groups"][-1]),
                "prompt_len": p,
                "output_len": int(r["output_len"]),
                "announce_len": a,
                "tail_len": int(r["tail_len"]),
                "result_token_count": int(r["result_token_count"]),
                "input_len": n,
            }
        )
    return results


def _response_record(tokenizer, prompt: str, response: str, result_text: str) -> dict[str, Any] | None:
    prompt_ids = encode_prompt(tokenizer, prompt)
    response_ids, offsets = encode_text_with_offsets(tokenizer, response)
    start = response.rfind(result_text)
    if start < 0:
        raise AssertionError("result text missing from response")
    rel = token_positions_for_char_spans(offsets, [[start, start + len(result_text)]])
    if not rel:
        return None
    return {
        "input_ids": prompt_ids + response_ids,
        "prompt_len": len(prompt_ids),
        "result_positions": [len(prompt_ids) + i for i in rel],
        "response_ids": response_ids,
    }


def build_protocol_probe(tokenizer, n_pairs: int, seed: int) -> list[dict[str, Any]]:
    """Build the seed-paper arithmetic correct-vs-wrong positive control."""
    rng = random.Random(seed)
    prompt = "Read the arithmetic equation below."
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < n_pairs and attempts < n_pairs * 500:
        attempts += 1
        a = rng.randint(11, 89)
        b = rng.randint(11, 89)
        correct = a + b
        delta = rng.choice([-9, -8, -7, -6, -5, -4, -3, 3, 4, 5, 6, 7, 8, 9])
        wrong = correct + delta
        if wrong <= 0:
            continue
        prefix = f"{a} + {b} = "
        correct_response = prefix + str(correct)
        wrong_response = prefix + str(wrong)
        c = _response_record(tokenizer, prompt, correct_response, str(correct))
        w = _response_record(tokenizer, prompt, wrong_response, str(wrong))
        if c is None or w is None:
            continue
        # Reproduce the paper's minimal result substitution as literally as possible.
        if len(c["input_ids"]) != len(w["input_ids"]):
            continue
        if hamming(c["input_ids"], w["input_ids"]) != 1:
            continue
        if len(c["result_positions"]) != 1 or len(w["result_positions"]) != 1:
            continue
        rows.append(
            {
                "probe_id": len(rows),
                "a": a,
                "b": b,
                "correct_result": correct,
                "wrong_result": wrong,
                "correct": c,
                "wrong": w,
            }
        )
    if len(rows) < n_pairs:
        raise RuntimeError(f"Could only build {len(rows)}/{n_pairs} one-token protocol probes")
    return rows


def score_protocol_probe(model, tokenizer, n_pairs: int, seed: int, batch_size: int, device: str) -> list[dict[str, Any]]:
    pairs = build_protocol_probe(tokenizer, n_pairs, seed)
    flat: list[dict[str, Any]] = []
    for pair in pairs:
        for label in ("correct", "wrong"):
            rec = dict(pair[label])
            rec.update({"probe_id": pair["probe_id"], "label": label})
            flat.append(rec)

    scored: dict[tuple[int, str], float] = {}
    for batch in exact_length_batches(flat, batch_size):
        ids = torch.tensor([r["input_ids"] for r in batch], dtype=torch.long, device=device)
        probs = forward_observed_token_probs(model, ids)
        for i, r in enumerate(batch):
            scored[(int(r["probe_id"]), r["label"])] = _mean_at(probs, i, r["result_positions"])

    out: list[dict[str, Any]] = []
    for pair in pairs:
        pid = int(pair["probe_id"])
        c = scored[(pid, "correct")]
        w = scored[(pid, "wrong")]
        out.append(
            {
                "probe_id": pid,
                "a": pair["a"],
                "b": pair["b"],
                "correct_result": pair["correct_result"],
                "wrong_result": pair["wrong_result"],
                "confidence_correct": c,
                "confidence_wrong": w,
                "gap": c - w,
            }
        )
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--revision", default="main")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--max-intervention-tokens", type=int, default=1)
    p.add_argument("--min-pairs", type=int, default=128)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--protocol-probe-output", type=Path)
    p.add_argument("--protocol-probe-pairs", type=int, default=100)
    p.add_argument("--protocol-probe-seed", type=int, default=20260823)
    p.add_argument("--runtime-output", type=Path)
    args = p.parse_args()
    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit("Require 0 <= shard-id < num-shards")
    if not args.audit_only and args.output is None:
        raise SystemExit("--output is required unless --audit-only is used")

    raw = read_jsonl(args.input)
    from transformers import AutoModel, AutoTokenizer, __version__ as transformers_version

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    encoded = [encode_sample(tokenizer, r) for r in raw]
    eligible, stats = select_eligible_pairs(encoded, args.max_intervention_tokens)
    n_pairs = int(stats["eligible_pairs"])
    probe = build_protocol_probe(tokenizer, args.protocol_probe_pairs, args.protocol_probe_seed)
    audit = {
        "factorial_tokenization": stats,
        "protocol_probe_pairs": len(probe),
        "model": args.model,
        "revision_requested": args.revision,
    }
    print("tokenization audit:", json.dumps(audit, sort_keys=True))
    if n_pairs < args.min_pairs:
        raise SystemExit(f"Only {n_pairs} eligible mirrored pairs; need >= {args.min_pairs}.")
    if args.audit_only:
        return

    eligible = [r for r in eligible if int(r["pair_id"]) % args.num_shards == args.shard_id]
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(args.device).eval()

    if args.runtime_output is not None:
        runtime = {
            "model": args.model,
            "revision_requested": args.revision,
            "model_commit": getattr(model.config, "_commit_hash", None),
            "tokenizer_commit": getattr(tokenizer, "_commit_hash", None),
            "torch": torch.__version__,
            "transformers": transformers_version,
            "dtype": args.dtype,
            "device": args.device,
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "tokenization_audit": stats,
        }
        args.runtime_output.parent.mkdir(parents=True, exist_ok=True)
        args.runtime_output.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.protocol_probe_output is not None:
        probe_rows = score_protocol_probe(
            model, tokenizer, args.protocol_probe_pairs, args.protocol_probe_seed, args.batch_size, args.device
        )
        write_jsonl(args.protocol_probe_output, probe_rows)
        print(f"scored {len(probe_rows)} protocol-control pairs -> {args.protocol_probe_output}")

    if not eligible:
        assert args.output is not None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("", encoding="utf-8")
        print("no factorial records assigned to this shard")
        return

    assert args.output is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.output.open("w", encoding="utf-8") as f:
        for batch in exact_length_batches(eligible, args.batch_size):
            for result in score_design_batch(model, batch, args.device):
                f.write(json.dumps(result, sort_keys=True) + "\n")
                n += 1
    print(f"scored {n} factorial samples on shard {args.shard_id}/{args.num_shards} -> {args.output}")


if __name__ == "__main__":
    main()
