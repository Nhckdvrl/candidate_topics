from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .env import (
    SessionSpec,
    build_session_spec,
    controlled_feedback,
    surface_to_latent,
    validate_yoke,
    yoked_feedback,
)
from .model_client import ChatClient, MockClient, OpenAICompatibleClient, parse_surface_action
from .renderers import get_family

SYSTEM_PROMPT = (
    "Choose one listed action each round to maximize total net points. "
    "A successful round earns 10 points. Any active intervention costs 1 point; waiting costs 0. "
    "Return JSON only in the form {\"action\":\"ACTION_NAME\"}."
)

FORBIDDEN_CONSTRUCT_WORDS = (
    "helpless", "worldview", "controllability", "uncontrollable", "prior belief"
)


def feedback_text(success: bool, reward: float, cost: float, net: float) -> str:
    return (
        f"Outcome: {'SUCCESS' if success else 'FAILURE'}. "
        f"Outcome points: {reward:.0f}; action cost: {cost:.0f}; net this round: {net:.0f}."
    )


def record_base(condition: str, diversity: str, pair_id: int, phase: str, episode: int, trial: int, family: str) -> dict[str, Any]:
    return {
        "condition": condition,
        "diversity": diversity,
        "pair_id": pair_id,
        "phase": phase,
        "episode": episode,
        "trial": trial,
        "family": family,
    }


async def choose_action(client: ChatClient, messages: list[dict[str, str]], family: str, trial: int) -> tuple[str, str, bool]:
    renderer = get_family(family)
    user_text = renderer.render_trial(trial)
    messages.append({"role": "user", "content": user_text})
    raw = await client.complete(messages)
    surface, valid = parse_surface_action(raw, renderer.surface_actions)
    # Store only the normalized action to prevent verbose model text from changing later context.
    messages.append({"role": "assistant", "content": json.dumps({"action": surface})})
    return surface, raw, valid


async def run_session(
    client: ChatClient,
    spec: SessionSpec,
    cfg: ExperimentConfig,
    condition: str,
    *,
    master_training_success: list[bool] | None = None,
) -> list[dict[str, Any]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    rows: list[dict[str, Any]] = []
    master_index = 0

    for ep_idx, ep in enumerate(spec.episodes):
        renderer = get_family(ep.family)
        messages.append({"role": "user", "content": renderer.render_start(ep_idx + 1, ep_idx + 1)})
        for t in range(cfg.trials_per_episode):
            surface, raw, valid = await choose_action(client, messages, ep.family, t + 1)
            latent = surface_to_latent(ep.family, surface)
            if condition == "controllable":
                fb = controlled_feedback(latent, ep, t, cfg, test=False)
            elif condition == "uncontrollable":
                if master_training_success is None:
                    raise ValueError("uncontrollable sessions require master outcomes")
                fb = yoked_feedback(latent, master_training_success[master_index], cfg)
            else:
                raise ValueError(condition)
            master_index += 1
            messages.append({"role": "user", "content": feedback_text(fb.success, fb.reward, fb.cost, fb.net)})
            row = record_base(condition, spec.diversity, spec.pair_id, "train", ep_idx + 1, t + 1, ep.family)
            row.update({
                "surface_action": surface,
                "latent_action": latent,
                "active": latent != "wait",
                "success": fb.success,
                "reward": fb.reward,
                "cost": fb.cost,
                "net": fb.net,
                "valid_action": valid,
                "raw_response": raw,
                "effective_action": ep.effective_action,
            })
            rows.append(row)

    # Novel test: identical task structure for all conditions, and objectively controllable.
    test = spec.test
    renderer = get_family(test.family)
    messages.append({"role": "user", "content": renderer.render_start(cfg.episodes + 1, spec.pair_id + 1)})
    for t in range(cfg.test_trials):
        surface, raw, valid = await choose_action(client, messages, test.family, t + 1)
        latent = surface_to_latent(test.family, surface)
        fb = controlled_feedback(latent, test, t, cfg, test=True)
        messages.append({"role": "user", "content": feedback_text(fb.success, fb.reward, fb.cost, fb.net)})
        row = record_base(condition, spec.diversity, spec.pair_id, "test", 1, t + 1, test.family)
        row.update({
            "surface_action": surface,
            "latent_action": latent,
            "active": latent != "wait",
            "success": fb.success,
            "reward": fb.reward,
            "cost": fb.cost,
            "net": fb.net,
            "valid_action": valid,
            "raw_response": raw,
            "effective_action": test.effective_action,
        })
        rows.append(row)
    return rows


async def run_pair(client: ChatClient, diversity: str, pair_id: int, cfg: ExperimentConfig, seed: int) -> list[dict[str, Any]]:
    master_spec = build_session_spec(diversity, pair_id, cfg, seed)
    master_rows = await run_session(client, master_spec, cfg, "controllable")
    master_success = [bool(r["success"]) for r in master_rows if r["phase"] == "train"]

    # Same latent task schedule and randomization; only the training outcome mechanism changes.
    yoked_spec = build_session_spec(diversity, pair_id, cfg, seed)
    yoked_rows = await run_session(
        client, yoked_spec, cfg, "uncontrollable", master_training_success=master_success
    )
    yoked_success = [bool(r["success"]) for r in yoked_rows if r["phase"] == "train"]
    validate_yoke(master_success, yoked_success)
    return master_rows + yoked_rows


async def bounded_pair(sem: asyncio.Semaphore, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    async with sem:
        return await run_pair(*args, **kwargs)


async def amain(args: argparse.Namespace) -> None:
    cfg = ExperimentConfig()
    if args.preflight:
        cfg = replace(cfg, episodes=8, trials_per_episode=5, test_trials=6)
    cfg.validate()
    if args.mock:
        client: ChatClient = MockClient(seed=args.seed)
    else:
        client = OpenAICompatibleClient(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            temperature=args.temperature,
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = out.with_suffix(out.suffix + ".meta.json")
    meta.write_text(json.dumps({
        "config": asdict(cfg), "seed": args.seed, "pairs_per_diversity": args.pairs,
        "model": args.model, "base_url": args.base_url, "temperature": args.temperature,
        "mock": args.mock,
    }, indent=2), encoding="utf-8")

    sem = asyncio.Semaphore(args.concurrency)
    tasks = []
    for diversity in ("concentrated", "distributed"):
        for pair_id in range(args.pairs):
            tasks.append(asyncio.create_task(
                bounded_pair(sem, client, diversity, pair_id, cfg, args.seed)
            ))
    try:
        all_rows_nested = await asyncio.gather(*tasks)
        with out.open("w", encoding="utf-8") as f:
            for rows in all_rows_nested:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            await close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the master-yoked controllability generalization pilot")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model", default="local-model")
    p.add_argument("--pairs", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--output", default="results/pilot.jsonl")
    p.add_argument("--preflight", action="store_true", help="40 training experiences/session instead of 100")
    p.add_argument("--mock", action="store_true", help="pipeline smoke test only")
    return p


if __name__ == "__main__":
    asyncio.run(amain(build_parser().parse_args()))
