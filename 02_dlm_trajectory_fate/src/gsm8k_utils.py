from __future__ import annotations

import re


def extract_number_strict(text: str) -> str | None:
    m = re.search(r"####\s*([+-]?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    return None


def extract_number(text: str) -> str | None:
    strict = extract_number_strict(text)
    if strict is not None:
        return strict
    nums = re.findall(r"[+-]?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def extract_boxed_number(text: str) -> str | None:
    # MidTruth GSM8K prompts ask for \boxed{...}; take the last numeric boxed answer.
    matches = re.findall(r"\\boxed\{\s*([+-]?[\d,]+(?:\.\d+)?)\s*\}", text)
    return matches[-1].replace(",", "") if matches else None


def gold_number(answer_text: str) -> str:
    m = re.search(r"####\s*([+-]?[\d,]+(?:\.\d+)?)", answer_text)
    if not m:
        raise ValueError(f"No GSM8K gold answer found in: {answer_text}")
    return m.group(1).replace(",", "")


def numerically_equal(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return float(pred) == float(gold)
    except ValueError:
        return pred.strip() == gold.strip()
