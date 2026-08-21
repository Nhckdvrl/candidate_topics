from src.environment import ControlEnvironment, EpisodePlan, make_episode_plan, replay_effects


def test_controllable_actions_have_stable_effects():
    plan = EpisodePlan(start_state=3, action_effects={"A": -1, "B": 0, "C": 1}, random_effects=(1, 1, 1, 1))
    env = ControlEnvironment(True, plan, n_steps=4, intervention_budget=4)
    assert env.step("A").effect == -1
    assert env.step("A").effect == -1
    assert env.step("C").effect == 1


def test_uncontrollable_ignores_action_identity():
    plan = EpisodePlan(start_state=3, action_effects={"A": -1, "B": 0, "C": 1}, random_effects=(1, -1, 0))
    env = ControlEnvironment(False, plan, n_steps=3, intervention_budget=3)
    assert env.step("A").effect == 1
    assert env.step("C").effect == -1
    assert env.step("B").effect == 0


def test_yoked_uncontrollable_replays_controllable_effects_exactly():
    plan = EpisodePlan(start_state=3, action_effects={"A": -1, "B": 0, "C": 1}, random_effects=(1, 1, 1))
    c = ControlEnvironment(True, plan, n_steps=3, intervention_budget=3)
    for a in ["A", "C", "B"]:
        c.step(a)
    effects = replay_effects(c.results)

    u = ControlEnvironment(False, plan, n_steps=3, intervention_budget=3, yoked_effects=effects)
    for a in ["C", "C", "A"]:
        u.step(a)
    assert replay_effects(u.results) == effects
    assert [r.state_after for r in u.results] == [r.state_after for r in c.results]


def test_budget_forces_wait_after_exhaustion():
    plan = EpisodePlan(start_state=3, action_effects={"A": -1, "B": 0, "C": 1}, random_effects=(1, 1, 1))
    env = ControlEnvironment(True, plan, n_steps=3, intervention_budget=1)
    env.step("A")
    assert env.valid_actions() == ("WAIT",)
    r = env.step("C")
    assert r.action == "WAIT"
    assert not r.active


def test_plan_is_reproducible_and_mapping_is_permutation():
    p1 = make_episode_plan(42, 10)
    p2 = make_episode_plan(42, 10)
    assert p1 == p2
    assert sorted(p1.action_effects.values()) == [-1, 0, 1]
