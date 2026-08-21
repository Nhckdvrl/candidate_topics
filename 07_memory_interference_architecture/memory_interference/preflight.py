from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from transformers import AutoConfig, AutoTokenizer

from .data import build_episode, load_pool, select_query_keys, validate_pool
from .prompts import candidate_values, render_prompt
from .scoring import longest_common_prefix


def tokenizer_fingerprint(tokenizer) -> str:
    probes = [
        "bird: emu\n",
        "visual art: illuminated manuscript\n",
        'What was the INITIAL value of "bird"?\nANSWER:\n',
        'What was the LAST (most recent) value of "bird"?\nANSWER:\n',
    ]
    ids = [tokenizer(x, add_special_tokens=True)["input_ids"] for x in probes]
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def run(config_path: str) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    pool = load_pool(cfg["data_file"])
    levels = [int(x) for x in cfg["num_updates"]]
    validate_pool(pool, int(cfg["num_keys"]), max(levels))

    report = {"config": config_path, "models": []}
    fingerprints = set()
    for spec in cfg["models"]:
        if spec.get("requires_fla", False):
            import fla  # noqa: F401

        model_id = spec["model_id"]
        config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=spec.get("trust_remote_code", False)
        )
        tokenizer = AutoTokenizer.from_pretrained(
            spec.get("tokenizer_id", model_id),
            trust_remote_code=spec.get("trust_remote_code", False),
        )
        fp = tokenizer_fingerprint(tokenizer)
        fingerprints.add(fp)
        max_context = spec.get("max_context") or getattr(config, "max_position_embeddings", None)

        max_prompt_tokens = 0
        max_observed_boundary_shift = 0
        for n in levels:
            ep = build_episode(
                pool,
                episode_id=0,
                num_keys=int(cfg["num_keys"]),
                num_updates=n,
                seed=int(cfg["seed"]),
            )
            for key in select_query_keys(
                ep, min(2, int(cfg["queries_per_episode"])), int(cfg["seed"])
            ):
                for condition in ("RI", "PI"):
                    prompt = render_prompt(ep, key, condition)
                    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
                    max_prompt_tokens = max(max_prompt_tokens, len(prompt_ids))
                    for candidate in candidate_values(ep, key):
                        full_ids = tokenizer(prompt + candidate, add_special_tokens=True)["input_ids"]
                        shift = len(prompt_ids) - longest_common_prefix(prompt_ids, full_ids)
                        max_observed_boundary_shift = max(max_observed_boundary_shift, shift)

        allowed_shift = int(cfg.get("max_boundary_shift", 0))
        if max_observed_boundary_shift > allowed_shift:
            raise RuntimeError(
                f"{spec['name']}: tokenizer boundary shift {max_observed_boundary_shift} exceeds "
                f"frozen maximum {allowed_shift}"
            )

        safety = int(cfg.get("context_safety_margin", 32))
        if isinstance(max_context, int) and max_prompt_tokens + safety >= max_context:
            raise RuntimeError(
                f"{spec['name']}: pilot prompt length {max_prompt_tokens} is too close to/exceeds "
                f"configured context {max_context} with safety margin {safety}"
            )
        report["models"].append(
            {
                "name": spec["name"],
                "model_id": model_id,
                "model_type": getattr(config, "model_type", None),
                "max_context": max_context,
                "vocab_size": getattr(config, "vocab_size", None),
                "tokenizer_class": tokenizer.__class__.__name__,
                "tokenizer_fingerprint": fp,
                "max_sampled_prompt_tokens": max_prompt_tokens,
                "max_sampled_boundary_shift": max_observed_boundary_shift,
            }
        )

    if cfg.get("require_shared_tokenizer", False) and len(fingerprints) != 1:
        raise RuntimeError(
            f"primary model family does not share one tokenizer fingerprint: {sorted(fingerprints)}"
        )
    print(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
