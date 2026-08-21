from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .agent import OpenAICompatibleAgent
from .environment import ControlEnvironment, EpisodePlan, make_plans, replay_effects
from .renderers import (
    SYSTEM_PROMPT,
    render_episode_end,
    render_episode_intro,
    render_feedback,
    heldout_family_key,
    training_family_keys,
)


async def run_episode(
    agent: OpenAICompatibleAgent,
    messages: List[Dict[str, str]],
    family_key: str,
    controllable: bool,
    plan: EpisodePlan,
    n_steps: int,
    budget: int,
    phase: str,
    episode_idx: int,
    yoked_effects: Optional[Sequence[int]] = None,
) -> Tuple[List[Dict[str, Any]], Tuple[int, ...]]:
    env = ControlEnvironment(
        controllable=controllable,
        plan=plan,
        n_steps=n_steps,
        intervention_budget=budget,
        yoked_effects=yoked_effects,
    )
    messages.append({"role": "user", "content": render_episode_intro(family_key, env.state, n_steps, budget)})
    records: List[Dict[str, Any]] = []

    for step_i in range(n_steps):
        reply = await agent.act(messages)
        messages.append({"role": "assistant", "content": reply.raw if reply.raw.strip() else reply.action})
        requested = reply.action
        valid_actions = env.valid_actions()
        budget_forced_wait = requested not in valid_actions and requested in {"A", "B", "C"}
        executed = "WAIT" if requested not in valid_actions else requested
        result = env.step(executed)
        records.append(
            {
                "phase": phase,
                "episode_idx": episode_idx,
                "step_idx": step_i,
                "family": family_key,
                "controllable": controllable,
                "state_before": result.state_before,
                "requested_action": requested,
                "action": result.action,
                "budget_forced_wait": budget_forced_wait,
                "raw_reply": reply.raw,
                "format_valid": reply.valid,
                "active": result.active,
                "effect": result.effect,
                "state_after": result.state_after,
                "improved": result.improved,
                "interventions_left": result.interventions_left,
                "hidden_action_effects": dict(plan.action_effects),
            }
        )
        if step_i < n_steps - 1:
            messages.append(
                {
                    "role": "user",
                    "content": render_feedback(
                        family_key,
                        result.state_before,
                        result.state_after,
                        result.interventions_left,
                        step_i + 1,
                        n_steps,
                    ),
                }
            )

    messages.append({"role": "user", "content": render_episode_end(family_key, env.state)})
    return records, replay_effects(env.results)


async def run_subject(
    agent: OpenAICompatibleAgent,
    base_seed: int,
    diversity: str,
    controllable: bool,
    train_plans: Sequence[EpisodePlan],
    test_plan: EpisodePlan,
    n_train_episodes: int,
    train_steps: int,
    test_steps: int,
    budget: int,
    yoked_training: Optional[Sequence[Sequence[int]]] = None,
) -> Dict[str, Any]:
    train_families = training_family_keys(base_seed, diversity, n_train_episodes)
    tfamily = heldout_family_key(base_seed)
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    all_steps: List[Dict[str, Any]] = []
    realized_effects: List[Tuple[int, ...]] = []

    for epi in range(n_train_episodes):
        yoked = None if yoked_training is None else yoked_training[epi]
        recs, effects = await run_episode(
            agent,
            messages,
            train_families[epi],
            controllable,
            train_plans[epi],
            train_steps,
            budget,
            "train",
            epi,
            yoked,
        )
        all_steps.extend(recs)
        realized_effects.append(effects)

    # Test is always objectively controllable and uses a held-out semantic family.
    test_recs, _ = await run_episode(
        agent,
        messages,
        tfamily,
        True,
        test_plan,
        test_steps,
        budget,
        "test",
        0,
        None,
    )
    all_steps.extend(test_recs)

    return {
        "base_seed": base_seed,
        "diversity": diversity,
        "history_controllability": "controllable" if controllable else "uncontrollable",
        "train_families": train_families,
        "test_family": tfamily,
        "n_train_episodes": n_train_episodes,
        "train_steps": train_steps,
        "test_steps": test_steps,
        "intervention_budget": budget,
        "steps": all_steps,
        "realized_training_effects": [list(x) for x in realized_effects],
    }


async def run_pair(
    semaphore: asyncio.Semaphore,
    agent: OpenAICompatibleAgent,
    base_seed: int,
    diversity: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    async with semaphore:
        # Crucial identification choice: for a given base seed, concentrated and
        # distributed histories receive identical latent episode plans. Semantic
        # family diversity is therefore the only planned difference across that axis.
        train_plans = make_plans(base_seed * 13, args.train_episodes, args.train_steps)
        test_plan = make_plans(base_seed * 97 + 100003, 1, args.test_steps)[0]

        controllable = await run_subject(
            agent,
            base_seed,
            diversity,
            True,
            train_plans,
            test_plan,
            args.train_episodes,
            args.train_steps,
            args.test_steps,
            args.budget,
        )
        uncontrollable = await run_subject(
            agent,
            base_seed,
            diversity,
            False,
            train_plans,
            test_plan,
            args.train_episodes,
            args.train_steps,
            args.test_steps,
            args.budget,
            yoked_training=controllable["realized_training_effects"],
        )
        return [controllable, uncontrollable]


async def amain(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    agent = OpenAICompatibleAgent(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = []
    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        for diversity in ("concentrated", "distributed"):
            tasks.append(asyncio.create_task(run_pair(semaphore, agent, seed, diversity, args)))

    metadata = {
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "n_seeds": args.n_seeds,
        "seed_start": args.seed_start,
        "train_episodes": args.train_episodes,
        "train_steps": args.train_steps,
        "test_steps": args.test_steps,
        "budget": args.budget,
        "design": "2x2 history controllability x semantic diversity; uncontrollable histories yoked to paired controllable raw effects",
    }
    meta_path = out.with_suffix(out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with out.open("w", encoding="utf-8") as fh:
        for fut in asyncio.as_completed(tasks):
            pair = await fut
            for record in pair:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the helplessness-generalization 2x2 pilot against an OpenAI-compatible endpoint")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--model", required=True)
    p.add_argument("--output", default="results/pilot.jsonl")
    p.add_argument("--n-seeds", type=int, default=40, help="40 seeds => 160 subject histories across 4 cells")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--train-episodes", type=int, default=10)
    p.add_argument("--train-steps", type=int, default=10)
    p.add_argument("--test-steps", type=int, default=8)
    p.add_argument("--budget", type=int, default=6)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=32, help="Concurrent paired subjects; each pair remains sequential for yoking")
    return p


if __name__ == "__main__":
    asyncio.run(amain(build_parser().parse_args()))
