from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from typing import Dict, Iterable, List, Tuple

from .environment import ACTIVE_ACTIONS, ControlEnvironment, make_episode_plan


def mutual_information(pairs: Iterable[Tuple[str, int]]) -> float:
    pairs = list(pairs)
    n = len(pairs)
    if n == 0:
        return 0.0
    joint = Counter(pairs)
    ca = Counter(a for a, _ in pairs)
    ce = Counter(e for _, e in pairs)
    mi = 0.0
    for (a, e), c in joint.items():
        p = c / n
        pa = ca[a] / n
        pe = ce[e] / n
        mi += p * math.log2(p / (pa * pe))
    return mi


def js_divergence(a: Counter, b: Counter) -> float:
    keys = sorted(set(a) | set(b))
    sa, sb = sum(a.values()), sum(b.values())
    pa = {k: a[k] / sa for k in keys}
    pb = {k: b[k] / sb for k in keys}
    m = {k: 0.5 * (pa[k] + pb[k]) for k in keys}

    def kl(p: Dict[int, float], q: Dict[int, float]) -> float:
        out = 0.0
        for k in keys:
            if p[k] > 0:
                out += p[k] * math.log2(p[k] / q[k])
        return out

    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


def audit(episodes: int = 2000, steps: int = 60, seed: int = 0) -> Dict[str, float]:
    rng = random.Random(seed)
    effect_counts = {True: Counter(), False: Counter()}
    episode_mi = {True: [], False: []}

    for controllable in (True, False):
        for epi in range(episodes):
            plan = make_episode_plan(seed * 100003 + epi * 17 + (1 if controllable else 2), steps)
            env = ControlEnvironment(controllable=controllable, plan=plan, n_steps=steps, intervention_budget=steps)
            pairs: List[Tuple[str, int]] = []
            for _ in range(steps):
                action = rng.choice(ACTIVE_ACTIONS)
                r = env.step(action)
                effect_counts[controllable][r.effect] += 1
                pairs.append((action, r.effect))
            episode_mi[controllable].append(mutual_information(pairs))

    js = js_divergence(effect_counts[True], effect_counts[False])
    c_mi = sum(episode_mi[True]) / episodes
    u_mi = sum(episode_mi[False]) / episodes
    return {
        "effect_js_divergence_bits": js,
        "mean_episode_action_effect_mi_controllable_bits": c_mi,
        "mean_episode_action_effect_mi_uncontrollable_bits": u_mi,
        "pass_effect_marginal_match": js < 0.005,
        "pass_controllability_separation": c_mi > 0.5 and u_mi < 0.08,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    result = audit(args.episodes, args.steps, args.seed)
    print(json.dumps(result, indent=2))
    if not (result["pass_effect_marginal_match"] and result["pass_controllability_separation"]):
        raise SystemExit(2)
