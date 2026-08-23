#!/usr/bin/env python3
"""Static artifact audit for the EACL-2026 Numeracy-Probing synthetic data.

Seed repository:
    VCY019/Numeracy-Probing
Seed paper:
    "LLMs Know More About Numbers than They Can Say" (EACL 2026)

Purpose
-------
Before investing GPU time in a causal-access experiment, audit whether the
released synthetic construction itself creates label artifacts:

1. scientific-notation rounding changes the underlying ordering;
2. rounding creates displayed ties;
3. answer position is badly imbalanced;
4. the paper's hard regime |log2(a/b)| < 0.1 is too sparse to support an
   instance-level critical-cell experiment.

The code below mirrors the released `src/construct_data.py` logic.  By default
it uses the released seed=0.  `--audit-seeds` can be used to inspect additional
seeds before creating an independent confirmation set.

Important discovered edge case
------------------------------
The released dec-sci generator does not reject a == b before formatting.  The
published seed=0 contains no ties, but some other seeds can.  Therefore any
new confirmation set should reject exact ties explicitly rather than blindly
reusing the generator with another seed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from decimal import Decimal, getcontext
from pathlib import Path

import pandas as pd

getcontext().prec = 28


def to_scientific(num):
    if num == 0:
        return "0"
    exponent = math.floor(math.log10(abs(num)))
    base = num / (10 ** exponent)
    return f"{base:.5g} × 10^{exponent}"


def parse_num(text: str) -> float:
    return float(eval(text.replace("×", "*").replace("^", "**").replace(",", "")))


def add_row(rows, split, digit, a_orig, b_orig, a_str, b_str, notation_a, notation_b):
    a_disp, b_disp = parse_num(a_str), parse_num(b_str)
    order_orig = (a_orig > b_orig) - (a_orig < b_orig)
    order_disp = (a_disp > b_disp) - (a_disp < b_disp)
    rows.append(
        {
            "split": split,
            "digit": digit,
            "a_orig": float(a_orig),
            "b_orig": float(b_orig),
            "a": a_str,
            "b": b_str,
            "a_disp": a_disp,
            "b_disp": b_disp,
            "notation_a": notation_a,
            "notation_b": notation_b,
            "order_orig": order_orig,
            "order_disp": order_disp,
            "answer_pos_disp": "a" if a_disp > b_disp else ("b" if b_disp > a_disp else "tie"),
            "abs_log2_ratio": abs(math.log2(a_disp / b_disp)) if a_disp > 0 and b_disp > 0 else None,
        }
    )


def generate_int_sci(seed: int) -> pd.DataFrame:
    random.seed(seed)
    rows = []
    for digit in range(2, 10):
        digit_data = []
        while len(digit_data) < 1400:
            rg = range(10 ** (digit - 1), 10**digit)
            a, b = random.choice(rg), random.choice(rg)
            if a == b:
                continue
            if random.random() < 0.5:
                a_str, b_str, notations = to_scientific(a), str(b), ("sci", "int")
            else:
                a_str, b_str, notations = str(a), to_scientific(b), ("int", "sci")
            digit_data.append((a, b, a_str, b_str, notations))
        random.shuffle(digit_data)
        for i, (a, b, a_str, b_str, notations) in enumerate(digit_data):
            split = "train" if i < 1000 else ("val" if i < 1200 else "test")
            add_row(rows, split, digit, a, b, a_str, b_str, *notations)
    return pd.DataFrame(rows)


def generate_dec_sci(seed: int) -> pd.DataFrame:
    random.seed(seed)
    rows = []
    for digit in range(2, 10):
        digit_data = []
        while len(digit_data) < 1400:
            int_a = random.randint(10 ** (digit - 1), 10**digit - 1)
            int_b = random.randint(10 ** (digit - 1), 10**digit - 1)
            dec_len_a = random.randint(0, 4)
            dec_len_b = random.randint(0, 4)
            dec_a = (
                Decimal(str(random.randint(0, 10**dec_len_a - 1))) / Decimal(10**dec_len_a)
                if dec_len_a > 0
                else Decimal("0")
            )
            dec_b = (
                Decimal(str(random.randint(0, 10**dec_len_b - 1))) / Decimal(10**dec_len_b)
                if dec_len_b > 0
                else Decimal("0")
            )
            a, b = Decimal(int_a) + dec_a, Decimal(int_b) + dec_b
            if random.random() < 0.5:
                a_str, b_str, notations = to_scientific(float(a)), str(b), ("sci", "dec")
            else:
                a_str, b_str, notations = str(a), to_scientific(float(b)), ("dec", "sci")
            digit_data.append((a, b, a_str, b_str, notations))
        random.shuffle(digit_data)
        for i, (a, b, a_str, b_str, notations) in enumerate(digit_data):
            split = "train" if i < 1000 else ("val" if i < 1200 else "test")
            add_row(rows, split, digit, a, b, a_str, b_str, *notations)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    out = {
        "n": int(len(df)),
        "display_ties": int((df.order_disp == 0).sum()),
        "original_ties": int((df.order_orig == 0).sum()),
        "ordering_changes": int((df.order_orig != df.order_disp).sum()),
    }
    for split in ("train", "val", "test"):
        s = df[df.split == split]
        close = s[s.abs_log2_ratio < 0.1]
        out[split] = {
            "n": int(len(s)),
            "display_ties": int((s.order_disp == 0).sum()),
            "ordering_changes": int((s.order_orig != s.order_disp).sum()),
            "answer_a_rate": float((s.answer_pos_disp == "a").mean()),
            "hard_abs_log2_lt_0p1_n": int(len(close)),
            "hard_abs_log2_lt_0p1_rate": float(len(close) / len(s)),
            "hard_answer_a_rate": float((close.answer_pos_disp == "a").mean()) if len(close) else None,
            "min_abs_log2_ratio": float(s.abs_log2_ratio.min()),
            "median_abs_log2_ratio": float(s.abs_log2_ratio.median()),
        }
    return out


def rounding_summary(df: pd.DataFrame) -> dict:
    errors = []
    for row in df.itertuples(index=False):
        if row.notation_a == "sci":
            errors.append(abs(row.a_disp - row.a_orig) / abs(row.a_orig))
        if row.notation_b == "sci":
            errors.append(abs(row.b_disp - row.b_orig) / abs(row.b_orig))
    s = pd.Series(errors)
    return {
        "max_relative_error": float(s.max()),
        "p95_relative_error": float(s.quantile(0.95)),
        "median_relative_error": float(s.median()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0, help="Published generator seed.")
    p.add_argument("--audit-seeds", type=int, default=1, help="Audit seeds [0, N).")
    p.add_argument("--out", type=Path, default=Path("numeracy_data_audit.json"))
    args = p.parse_args()

    all_seed_results = []
    published = None
    for seed in range(args.audit_seeds):
        int_df = generate_int_sci(seed)
        dec_df = generate_dec_sci(seed)
        payload = {
            "seed": seed,
            "int_sci": summarize(int_df),
            "dec_sci": summarize(dec_df),
            "int_sci_rounding": rounding_summary(int_df),
            "dec_sci_rounding": rounding_summary(dec_df),
        }
        all_seed_results.append(payload)
        if seed == args.seed:
            published = payload

    if published is None:
        int_df = generate_int_sci(args.seed)
        dec_df = generate_dec_sci(args.seed)
        published = {
            "seed": args.seed,
            "int_sci": summarize(int_df),
            "dec_sci": summarize(dec_df),
            "int_sci_rounding": rounding_summary(int_df),
            "dec_sci_rounding": rounding_summary(dec_df),
        }

    result = {
        "published_seed_audit": published,
        "multi_seed_audit": all_seed_results,
        "warning": (
            "The released dec-sci generator does not reject exact a==b. "
            "Published seed=0 is clean, but independent confirmation seeds must reject ties explicitly."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
