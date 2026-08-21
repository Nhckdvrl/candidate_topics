from memory_interference.decide import decide


def make_rows(t_i=1, g_i=-1, n_levels=4, reps=8):
    rows = []
    for n in [1, 3, 7, 15][:n_levels]:
        for j in range(reps):
            for model, d in [("transformer_1.3b", t_i), ("gated_deltanet_1.3b", g_i)]:
                if d == 1:
                    vals = {"RI": True, "PI": False}
                elif d == -1:
                    vals = {"RI": False, "PI": True}
                else:
                    vals = {"RI": True, "PI": True}
                for condition, correct in vals.items():
                    rows.append({
                        "model": model,
                        "condition": condition,
                        "num_updates": n,
                        "correct": correct,
                        "target_rank": 1 if correct else 2,
                        "skipped": False,
                        "episode_id": j,
                        "query_key": f"k{j}",
                    })
    return rows


def test_decision_strong_go_on_repeated_sign_flip():
    out = decide(make_rows(), n_boot=100)
    assert out["decision"] == "STRONG_GO"


def test_decision_paradigm_fail_without_transformer_pi_bias():
    out = decide(make_rows(t_i=0, g_i=0), n_boot=100)
    assert out["decision"] == "PARADIGM_FAIL"
