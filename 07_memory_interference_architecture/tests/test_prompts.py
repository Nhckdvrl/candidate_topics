from memory_interference.data import build_episode
from memory_interference.prompts import candidate_values, render_prompt, target_value


def pool():
    return {
        "bird": ["emu", "kea", "rook"],
        "tool": ["saw", "drill", "awl"],
    }


def test_shared_stream_differs_only_in_query_target_language():
    ep = build_episode(pool(), episode_id=0, num_keys=2, num_updates=2, seed=1)
    key = ep.categories[0]
    ri = render_prompt(ep, key, "RI")
    pi = render_prompt(ep, key, "PI")
    ri_prefix = ri.split("QUERY:\n", 1)[0]
    pi_prefix = pi.split("QUERY:\n", 1)[0]
    assert ri_prefix == pi_prefix
    assert "INITIAL value" in ri
    assert "LAST (most recent) value" in pi


def test_targets_and_candidates():
    ep = build_episode(pool(), episode_id=0, num_keys=2, num_updates=2, seed=1)
    key = ep.categories[0]
    candidates = candidate_values(ep, key)
    assert target_value(ep, key, "RI") == candidates[0]
    assert target_value(ep, key, "PI") == candidates[-1]
