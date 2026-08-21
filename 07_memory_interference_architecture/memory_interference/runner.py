from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import yaml

from .data import build_episode, load_pool, select_query_keys, validate_pool
from .modeling import load_model
from .prompts import candidate_values, render_prompt, target_value
from .scoring import score_candidates


def _json_dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _append_jsonl(row: dict, path: Path):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(
    config_path: str,
    model_filter: List[str] | None = None,
    *,
    output_root: str | None = None,
    device: str | None = None,
) -> Path:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if output_root is not None:
        cfg["output_root"] = output_root
    if device is not None:
        cfg["models"] = [{**spec, "device": device} for spec in cfg["models"]]

    pool = load_pool(cfg["data_file"])
    levels = [int(x) for x in cfg["num_updates"]]
    validate_pool(pool, int(cfg["num_keys"]), max(levels))

    run_name = cfg.get("run_name", time.strftime("pilot_%Y%m%d_%H%M%S"))
    out_dir = Path(cfg.get("output_root", "outputs")) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(cfg, out_dir / "resolved_config.json")

    model_specs = cfg["models"]
    if model_filter:
        wanted = set(model_filter)
        model_specs = [m for m in model_specs if m["name"] in wanted]
        missing = wanted - {m["name"] for m in model_specs}
        if missing:
            raise ValueError(f"unknown model names: {sorted(missing)}")

    results_path = out_dir / "results.jsonl"
    if results_path.exists() and not cfg.get("resume", False):
        results_path.unlink()

    for spec in model_specs:
        loaded = load_model(spec)
        model, tokenizer, device = loaded.model, loaded.tokenizer, loaded.device
        model_name = spec["name"]
        print(f"[model] {model_name}: {spec['model_id']} on {device}")

        for n_updates in levels:
            for episode_id in range(int(cfg["episodes_per_level"])):
                episode = build_episode(
                    pool,
                    episode_id=episode_id,
                    num_keys=int(cfg["num_keys"]),
                    num_updates=n_updates,
                    seed=int(cfg["seed"]),
                )
                qkeys = select_query_keys(
                    episode, int(cfg["queries_per_episode"]), int(cfg["seed"])
                )
                for query_key in qkeys:
                    candidates = candidate_values(episode, query_key)
                    for condition in ("RI", "PI"):
                        prompt = render_prompt(episode, query_key, condition)
                        prompt_tokens = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
                        safety = int(cfg.get("context_safety_margin", 32))
                        if loaded.max_context is not None and prompt_tokens + safety >= loaded.max_context:
                            row = {
                                "model": model_name,
                                "architecture": spec.get("architecture", "unknown"),
                                "model_id": spec["model_id"],
                                "episode_id": episode_id,
                                "num_updates": n_updates,
                                "query_key": query_key,
                                "condition": condition,
                                "target": target_value(episode, query_key, condition),
                                "skipped": True,
                                "skip_reason": f"prompt_tokens={prompt_tokens} exceeds safe context {loaded.max_context - safety}",
                                "prompt_tokens": prompt_tokens,
                            }
                            _append_jsonl(row, results_path)
                            continue

                        scores = score_candidates(
                            model,
                            tokenizer,
                            prompt,
                            candidates,
                            device=device,
                            batch_size=int(cfg.get("candidate_batch_size", 16)),
                            max_boundary_shift=int(cfg.get("max_boundary_shift", 0)),
                        )
                        metric = cfg.get("candidate_metric", "mean_logprob")
                        if metric not in ("mean_logprob", "sum_logprob"):
                            raise ValueError("candidate_metric must be mean_logprob or sum_logprob")
                        ranked = sorted(scores, key=lambda x: getattr(x, metric), reverse=True)
                        target = target_value(episode, query_key, condition)
                        predicted = ranked[0].candidate
                        target_rank = next(i + 1 for i, x in enumerate(ranked) if x.candidate == target)
                        history = list(episode.histories[query_key])
                        predicted_position = history.index(predicted) / max(1, len(history) - 1)
                        target_position = history.index(target) / max(1, len(history) - 1)

                        row = {
                            "model": model_name,
                            "architecture": spec.get("architecture", "unknown"),
                            "model_id": spec["model_id"],
                            "episode_id": episode_id,
                            "num_updates": n_updates,
                            "query_key": query_key,
                            "condition": condition,
                            "target": target,
                            "predicted": predicted,
                            "correct": predicted == target,
                            "target_rank": target_rank,
                            "target_position": target_position,
                            "predicted_position": predicted_position,
                            "prompt_tokens": prompt_tokens,
                            "candidate_metric": metric,
                            "scores": [
                                {
                                    "candidate": s.candidate,
                                    "sum_logprob": s.sum_logprob,
                                    "mean_logprob": s.mean_logprob,
                                    "token_count": s.token_count,
                                    "boundary_shift": s.boundary_shift,
                                    "history_position": history.index(s.candidate) / max(1, len(history) - 1),
                                }
                                for s in scores
                            ],
                            "skipped": False,
                        }
                        _append_jsonl(row, results_path)
        del loaded, model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"[done] {results_path}")
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model", action="append", default=None, help="model name from config; may be repeated"
    )
    parser.add_argument("--output-root", default=None, help="engineering-only isolated output root")
    parser.add_argument("--device", default=None, help="engineering-only device override")
    args = parser.parse_args()
    run(args.config, args.model, output_root=args.output_root, device=args.device)


if __name__ == "__main__":
    main()
