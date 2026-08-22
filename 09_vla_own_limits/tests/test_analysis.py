import numpy as np
import pandas as pd
import pytest
import torch
from types import SimpleNamespace

from src.feature_panel import aggregate_feature_replicates
from src.openpi_instrumented_server import ActionExpertLayerCapture, ControlledInstrumentedPolicy
from src.panel import choose_identifiable_pair, pair_stats, validate_panel
from src.relative_probe import SharedLinearProbe, paired_relative_metrics
from src.state_contract import deterministic_noise_seed, hash_sim_state


def _rows(spec, *, seeds=range(8), hash_value="abc"):
    rows = []
    for cp, counts in spec.items():
        for init_idx, n_success in enumerate(counts):
            for j, ps in enumerate(seeds):
                rows.append({
                    "suite": "libero_10",
                    "task_id": 0,
                    "init_idx": init_idx,
                    "env_seed": 7,
                    "sim_state_hash": f"{hash_value}-{init_idx}",
                    "checkpoint": cp,
                    "policy_seed": int(ps),
                    "success": int(j < n_success),
                })
    return pd.DataFrame(rows)


def test_robust_crossover_uses_rates_not_single_draws():
    df = _rows({"2k": [6, 2, 5], "9k": [2, 6, 4]})
    s = pair_stats(df, "2k", "9k", min_trials=8, rate_gap=0.5)
    assert s.n_a_wins == 1
    assert s.n_b_wins == 1
    assert s.n_ambiguous == 1


def test_common_policy_seed_sets_are_mandatory():
    df = _rows({"2k": [4], "9k": [4]})
    mask = ~((df.checkpoint == "9k") & (df.policy_seed == 7))
    with pytest.raises(ValueError, match="policy_seed sets differ|only 7 trials"):
        validate_panel(df[mask], min_trials=1)


def test_mismatched_physical_state_hash_is_rejected():
    df = _rows({"2k": [4], "9k": [4]})
    df.loc[(df.checkpoint == "9k") & (df.policy_seed == 0), "sim_state_hash"] = "wrong"
    with pytest.raises(ValueError, match="sim_state_hash mismatch"):
        validate_panel(df)


def test_pair_selection_prefers_bidirectional_robust_support():
    df = _rows({
        "2k": [7, 7, 1, 1, 7, 1],
        "9k": [1, 1, 1, 1, 7, 1],
        "3k": [1, 7, 7, 1, 1, 7],
    })
    chosen = choose_identifiable_pair(df, min_trials=8, rate_gap=0.5)
    assert chosen is not None
    assert {chosen.checkpoint_a, chosen.checkpoint_b} == {"2k", "3k"}
    assert chosen.bidirectional_support >= 2


def _paired_features(n=200, kind="specific", seed=1):
    rng = np.random.default_rng(seed)
    winner_a = np.asarray([0, 1] * (n // 2))
    generic = rng.normal(size=(n, 2))
    x, target, sid, cp, winners = [], [], [], [], []
    for i in range(n):
        sign = 1.0 if winner_a[i] else -1.0
        if kind == "specific":
            xa = np.r_[generic[i], sign + 0.1 * rng.normal()]
            xb = np.r_[generic[i], -sign + 0.1 * rng.normal()]
        elif kind == "state_only":
            xa = xb = np.r_[generic[i], 0.0]
        elif kind == "checkpoint_only":
            xa = np.r_[generic[i], 1.0]
            xb = np.r_[generic[i], -1.0]
        else:
            raise ValueError(kind)
        x += [xa, xb]
        target += [float(winner_a[i]), float(1 - winner_a[i])]
        sid += [str(i), str(i)]
        cp += ["A", "B"]
        w = "A" if winner_a[i] else "B"
        winners += [w, w]
    return np.asarray(x), np.asarray(target), np.asarray(sid), np.asarray(cp), np.asarray(winners)


def test_shared_probe_detects_state_dependent_policy_specific_signal():
    x, target, sid, cp, winners = _paired_features(kind="specific")
    p = SharedLinearProbe(alpha=1.0).fit(x, target)
    m = paired_relative_metrics(sid, cp, winners, p.score(x), "A", "B")
    assert m["relative_auc"] > 0.95


def test_state_only_signal_cancels_in_pair():
    x, target, sid, cp, winners = _paired_features(kind="state_only")
    p = SharedLinearProbe(alpha=1.0).fit(x, target)
    m = paired_relative_metrics(sid, cp, winners, p.score(x), "A", "B")
    assert m["relative_auc"] == 0.5


def test_checkpoint_identity_only_cannot_solve_bidirectional_crossover():
    x, target, sid, cp, winners = _paired_features(kind="checkpoint_only")
    p = SharedLinearProbe(alpha=1.0).fit(x, target)
    m = paired_relative_metrics(sid, cp, winners, p.score(x), "A", "B")
    assert m["relative_auc"] == 0.5


def test_state_hash_and_noise_stream_are_stable():
    x = np.array([1.234567890123, 2.0, -3.0])
    assert hash_sim_state(x) == hash_sim_state(x.copy())
    a = deterministic_noise_seed(100, suite="libero_10", task_id=1, init_idx=2, replan_idx=3)
    b = deterministic_noise_seed(100, suite="libero_10", task_id=1, init_idx=2, replan_idx=3)
    c = deterministic_noise_seed(100, suite="libero_10", task_id=1, init_idx=2, replan_idx=4)
    assert a == b and a != c


def test_feature_replicates_use_same_seed_set_and_average():
    raw = {"state_id": [], "checkpoint": [], "sim_state_hash": [], "feature_seed": [], "feature": []}
    for cp, offset in [("A", 0.0), ("B", 10.0)]:
        for fs in [1, 2, 3, 4]:
            raw["state_id"].append("s")
            raw["checkpoint"].append(cp)
            raw["sim_state_hash"].append("h")
            raw["feature_seed"].append(fs)
            raw["feature"].append([offset + fs, 2.0])
    raw = {k: np.asarray(v) for k, v in raw.items()}
    p = aggregate_feature_replicates(raw, min_seeds=4)
    assert p.feature.shape == (2, 2)
    assert np.allclose(p.feature[0], [2.5, 2.0])
    assert np.allclose(p.feature[1], [12.5, 2.0])


def test_feature_replicates_reject_different_noise_seeds():
    raw = {"state_id": [], "checkpoint": [], "sim_state_hash": [], "feature_seed": [], "feature": []}
    for cp, seeds in [("A", [1, 2, 3, 4]), ("B", [1, 2, 3, 5])]:
        for fs in seeds:
            raw["state_id"].append("s")
            raw["checkpoint"].append(cp)
            raw["sim_state_hash"].append("h")
            raw["feature_seed"].append(fs)
            raw["feature"].append([float(fs)])
    raw = {k: np.asarray(v) for k, v in raw.items()}
    with pytest.raises(ValueError, match="feature_seed sets differ"):
        aggregate_feature_replicates(raw, min_seeds=4)


def test_action_expert_capture_is_observational_and_pools_steps():
    class Layer(torch.nn.Module):
        def forward(self, x):
            return (x + 1.0,)

    layer = Layer()
    model = SimpleNamespace(
        pi05=True,
        paligemma_with_expert=SimpleNamespace(
            gemma_expert=SimpleNamespace(model=SimpleNamespace(layers=torch.nn.ModuleList([layer])))
        ),
    )
    x = torch.zeros(1, 4, 3)
    with ActionExpertLayerCapture(model, 0, expected_denoise_steps=10) as cap:
        for k in range(10):
            y = layer(x + float(k))[0]
            assert torch.allclose(y, x + float(k) + 1.0)
    assert np.allclose(cap.pooled(), np.full(3, 5.5))


def test_controlled_policy_noise_is_reproducible():
    class Base:
        metadata = {}
        _is_pytorch_model = False

        def infer(self, req, *, noise):
            return {"actions": noise.copy()}

    p = ControlledInstrumentedPolicy(Base(), action_horizon=2, action_dim=3)
    r1 = p.infer({"__topic09_noise_seed": 123})["actions"]
    r2 = p.infer({"__topic09_noise_seed": 123})["actions"]
    r3 = p.infer({"__topic09_noise_seed": 124})["actions"]
    assert np.array_equal(r1, r2)
    assert not np.array_equal(r1, r3)


def test_grouped_alpha_selection_uses_state_groups_and_regularizes_wide_features():
    """With D >> N the selected penalty must be far above the old fixed alpha=1."""
    rng = np.random.default_rng(0)
    n_states, d = 150, 1024
    sid, cp, x, y = [], [], [], []
    for i in range(n_states):
        state = rng.normal(size=d)
        for c, shift in [("A", 0.4), ("B", -0.4)]:
            sid.append(f"s{i}")
            cp.append(c)
            x.append(state + shift * rng.normal(size=d) * 0.1)
            y.append(float(np.clip(0.5 + 0.05 * state[0] + 0.02 * rng.normal(), 0, 1)))
    probe = SharedLinearProbe(alpha="cv").fit(np.asarray(x), np.asarray(y), groups=np.asarray(sid))
    assert probe.alpha_ > 1.0
    assert probe.alpha_ in {r["alpha"] for r in probe.alpha_cv_}
    assert len(probe.score(np.asarray(x))) == 2 * n_states


def test_grouped_alpha_selection_requires_groups():
    x = np.random.default_rng(0).normal(size=(20, 5))
    y = np.random.default_rng(1).random(20)
    with pytest.raises(ValueError, match="requires state_id groups"):
        SharedLinearProbe(alpha="cv").fit(x, y)

def _wide_paired_features(kind, states, d=256, seed=7):
    """Realistic G1 geometry: wide features, two checkpoints, few states."""
    rng = np.random.default_rng(seed)
    sid, cp, x, target, winners = [], [], [], [], []
    for i in states:
        k = i % 5
        if k == 0:
            p = {"A": 0.95, "B": 0.05}
        elif k == 1:
            p = {"A": 0.05, "B": 0.95}
        else:
            b = float(rng.random())
            p = {"A": b, "B": b}
        w = "A" if p["A"] - p["B"] >= 0.5 else ("B" if p["B"] - p["A"] >= 0.5 else "ambiguous")
        state = rng.normal(size=d)
        difficulty = (p["A"] + p["B"]) / 2
        for c in ["A", "B"]:
            v = state.copy()
            # dim 0 is the only informative direction; dim 1 is pure checkpoint identity
            v[0] = (p[c] - 0.5) * 4 if kind == "policy_specific" else (difficulty - 0.5) * 4
            v[1] = 3.0 if c == "A" else -3.0
            sid.append(str(i))
            cp.append(c)
            winners.append(w)
            target.append(p[c])
            x.append(v + 0.05 * rng.normal(size=d))
    return (np.asarray(sid), np.asarray(cp), np.asarray(winners),
            np.asarray(target), np.asarray(x))


def _held_out_auc(kind, d=256):
    tr = _wide_paired_features(kind, range(150), d=d, seed=7)
    te = _wide_paired_features(kind, range(150, 300), d=d, seed=8)
    probe = SharedLinearProbe(alpha="cv").fit(tr[4], tr[3], groups=tr[0])
    return paired_relative_metrics(te[0], te[1], te[2], probe.score(te[4]), "A", "B")["relative_auc"]


def test_wide_probe_separates_policy_specific_from_state_only():
    """The discrimination the whole topic rests on, at a realistic feature width.

    Both conditions carry a strong checkpoint-identity direction and the same generic
    difficulty content. Only the first also encodes *whose* success.
    """
    assert _held_out_auc("policy_specific") > 0.90
    assert 0.35 < _held_out_auc("state_only") < 0.65


def test_in_sample_scoring_would_manufacture_a_false_positive():
    """Why run_g1 must score on held-out physical states.

    Scored in-sample, the overparameterized ridge reconstructs the per-checkpoint targets
    through idiosyncratic noise directions, so `q_A - q_B` tracks the winner even when the
    only informative feature dimension is *identical* for both checkpoints. That is a
    false PASS for a purely numerical reason, and the discovery/confirmation split is what
    prevents it.
    """
    sid, cp, win, target, x = _wide_paired_features("state_only", range(150))
    probe = SharedLinearProbe(alpha="cv").fit(x, target, groups=sid)
    in_sample = paired_relative_metrics(sid, cp, win, probe.score(x), "A", "B")["relative_auc"]
    assert in_sample > 0.65, "precondition: in-sample scoring inflates a state-only signal"
    assert _held_out_auc("state_only") < in_sample


def test_collect_behavior_builds_one_environment_per_rollout():
    """Guards the fix for env-history contamination of the settled state.

    Reusing one environment across episodes stops reproducing the settled MuJoCo state
    after enough episodes, even though a freshly built environment reproduces it exactly.
    That is fatal here: the checkpoints being compared have different episode histories,
    so "the same physical state" would quietly stop being the same. The construction must
    therefore sit inside the per-rollout loop, not outside it.
    """
    import inspect

    from src import collect_behavior

    src = inspect.getsource(collect_behavior.main)
    make_env_line = next(i for i, l in enumerate(src.splitlines()) if "make_env(" in l)
    seed_loop_line = next(i for i, l in enumerate(src.splitlines()) if "for policy_seed in policy_seeds" in l)
    assert make_env_line > seed_loop_line, (
        "make_env must be inside the policy-seed loop so every rollout gets a fresh env"
    )
    assert src.count("make_env(") == 1
    assert "env.close()" in src
