from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from decision_state import classify_state, transition


TARGET_PREFIX_STATES = {"PROPOSED", "TENTATIVE", "CONDITIONAL"}
FINALITY_BLOCK = re.compile(
    r"\b(?:decid(?:e|ed|ing)|agree(?:d|ment)?|chose|chosen|determined|go ahead|"
    r"we(?:'re| are) going (?:with|for)|we can continue|out of discussion|"
    r"forget about|can(?:not|'t) use)\b",
    flags=re.IGNORECASE,
)
CONTENT_STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "we", "i", "it",
    "is", "are", "was", "were", "be", "that", "this", "with", "have", "has", "do",
    "so", "um", "uh", "yeah", "think", "maybe", "could", "would", "should", "will",
}


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.casefold()) if len(token) > 2 and token not in CONTENT_STOP}


def content_recall(source: str, summary: str) -> float:
    source_tokens = _tokens(source)
    if not source_tokens:
        return 0.0
    return len(source_tokens & _tokens(summary)) / len(source_tokens)


def load_candidates(root: Path, limit: int | None = None) -> list[dict]:
    candidates = []
    for path in sorted(root.rglob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, list):
            continue
        for item in obj:
            abstract = item.get("abstractive", {})
            if str(abstract.get("type", "")).casefold() != "decisions":
                continue
            turns = [turn for turn in item.get("extractive", []) if isinstance(turn, dict) and turn.get("text")]
            turns.sort(key=lambda turn: float(turn.get("starttime", 0)))
            if len(turns) < 2:
                continue
            chosen = None
            # Use the latest still-explicitly-nonfinal prefix. This maximizes
            # context while withholding at least the final linked contribution.
            for cut in range(len(turns) - 1, 0, -1):
                prefix_text = " ".join(str(turn["text"]) for turn in turns[:cut])
                # Candidate purity matters more than count: discard prefixes
                # containing even broad finality language that the conservative
                # state classifier intentionally does not otherwise trust.
                if FINALITY_BLOCK.search(prefix_text):
                    continue
                parsed = classify_state(prefix_text, genre="source")
                terminal_cue = any(state in parsed.matched for state in ("DECIDED", "REJECTED", "OPEN"))
                if (
                    parsed.state in TARGET_PREFIX_STATES
                    and parsed.explicit
                    and not parsed.ambiguous
                    and not terminal_cue
                ):
                    chosen = (cut, prefix_text, parsed)
                    break
            if chosen is None:
                continue
            cut, prefix_text, parsed = chosen
            full_text = " ".join(str(turn["text"]) for turn in turns)
            candidate_id = str(abstract.get("id") or f"{path.stem}:{len(candidates)}")
            candidates.append(
                {
                    "id": candidate_id,
                    "meeting": path.stem,
                    "decision_abstract": abstract.get("text"),
                    "prefix_state": parsed.state,
                    "cut_turn": cut,
                    "n_turns": len(turns),
                    "prefix_text": prefix_text,
                    "full_text": full_text,
                    "prefix_transcript": "\n".join(f"{turn.get('speaker', '?')}: {turn['text']}" for turn in turns[:cut]),
                    "full_transcript": "\n".join(f"{turn.get('speaker', '?')}: {turn['text']}" for turn in turns),
                }
            )
    candidates.sort(key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())
    return candidates[:limit] if limit else candidates


def _prompts(rows: list[dict], field: str, preserve_state: bool = False) -> list[str]:
    instruction = (
        "Write exactly one concise meeting-minutes sentence summarizing the following meeting excerpt. "
        "Report its main outcome or plan in plain language. Do not add explanation or bullet points.\n\n"
    )
    if preserve_state:
        instruction = (
            "Write exactly one concise meeting-minutes sentence summarizing the following meeting excerpt. "
            "Preserve whether the main item is only proposed, tentative, conditional, rejected, or finally decided; "
            "never state that a decision was made unless the excerpt explicitly makes it final. "
            "Do not add explanation or bullet points.\n\n"
        )
    return [instruction + row[field] for row in rows]


def generate(model_path: str, prompts: list[str], batch_size: int) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    )
    outputs = []
    for start in range(0, len(prompts), batch_size):
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts[start:start + batch_size]]
        rendered = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
        inputs = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return [output.strip().replace("\n", " ") for output in outputs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    rows = load_candidates(args.root, args.limit)
    if not rows:
        raise RuntimeError("no explicit nonfinal temporal-prefix candidates found")
    prefix_summaries = generate(args.model, _prompts(rows, "prefix_transcript"), args.batch_size)
    preserving_summaries = generate(
        args.model, _prompts(rows, "prefix_transcript", preserve_state=True), args.batch_size
    )
    full_summaries = generate(args.model, _prompts(rows, "full_transcript"), args.batch_size)

    for row, prefix_summary, preserving_summary, full_summary in zip(
        rows, prefix_summaries, preserving_summaries, full_summaries
    ):
        row["prefix_summary"] = prefix_summary
        row["preserving_summary"] = preserving_summary
        row["full_summary"] = full_summary
        row["prefix_transition"] = transition(row["prefix_text"], prefix_summary)
        row["preserving_transition"] = transition(row["prefix_text"], preserving_summary)
        row["full_summary_parse"] = classify_state(full_summary, genre="summary").__dict__
        row["prefix_content_recall"] = content_recall(row["prefix_text"], prefix_summary)

    scorable = [row for row in rows if row["prefix_transition"]["source_scorable"]]
    content_grounded = [row for row in scorable if row["prefix_content_recall"] >= 0.05]
    by_state = {}
    for state in sorted(TARGET_PREFIX_STATES):
        subset = [row for row in content_grounded if row["prefix_state"] == state]
        by_state[state] = {
            "n": len(subset),
            "upgrade_rate": sum(row["prefix_transition"]["upgrade"] for row in subset) / max(len(subset), 1),
        }
    result = {
        "model": args.model,
        "n_candidates": len(rows),
        "n_source_scorable": len(scorable),
        "n_content_grounded": len(content_grounded),
        "unsupported_upgrade_rate_all_scorable": sum(row["prefix_transition"]["upgrade"] for row in scorable) / max(len(scorable), 1),
        "unsupported_upgrade_rate_content_grounded": sum(row["prefix_transition"]["upgrade"] for row in content_grounded) / max(len(content_grounded), 1),
        "state_preserving_prompt_upgrade_rate_same_candidates": sum(
            row["preserving_transition"]["upgrade"] for row in content_grounded
        ) / max(len(content_grounded), 1),
        "full_summary_decided_rate": sum(row["full_summary_parse"]["state"] == "DECIDED" for row in rows) / len(rows),
        "by_prefix_state": by_state,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.joinpath("g0_temporal_prefix_records.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    args.out_dir.joinpath("g0_temporal_prefix_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
