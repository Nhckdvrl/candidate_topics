"""Run the locked ordinary 9x9 competence decode for one G1/v4 checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data", default="data/g1_v4/test.jsonl", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-new-tokens", type=int, default=172)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda().eval()

    predictions = []
    for i, row in enumerate(rows, 1):
        inputs = tokenizer(row["prompt"], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.diffusion_generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=args.max_new_tokens,
                steps=args.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                alg="entropy",
                alg_temp=0.0,
                return_dict_in_generate=True,
            )
        continuation = output.sequences[0, inputs.input_ids.shape[1] :]
        text = tokenizer.decode(continuation, skip_special_tokens=False)
        predictions.append({"id": row["id"], "prediction": text})
        print(f"completed {i}/{len(rows)}", flush=True)

    args.out.write_text("\n".join(json.dumps(x) for x in predictions) + "\n")


if __name__ == "__main__":
    main()
