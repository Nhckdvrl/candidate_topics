import asyncio

from src.agent import AgentReply
from src.environment import make_plans
from src.experiment import run_subject


class CyclingAgent:
    def __init__(self):
        self.i = 0

    async def act(self, messages):
        action = ("A", "B", "C", "WAIT")[self.i % 4]
        self.i += 1
        return AgentReply(action=action, raw=action, valid=True)


def test_end_to_end_subject_and_yoke_smoke():
    plans = make_plans(123, 3, 4)
    test_plan = make_plans(999, 1, 3)[0]
    c = asyncio.run(
        run_subject(CyclingAgent(), 0, "distributed", True, plans, test_plan, 3, 4, 3, 3)
    )
    u = asyncio.run(
        run_subject(
            CyclingAgent(),
            0,
            "distributed",
            False,
            plans,
            test_plan,
            3,
            4,
            3,
            3,
            yoked_training=c["realized_training_effects"],
        )
    )
    assert len(c["steps"]) == 15
    assert len(u["steps"]) == 15
    c_train = [x for x in c["steps"] if x["phase"] == "train"]
    u_train = [x for x in u["steps"] if x["phase"] == "train"]
    assert [x["effect"] for x in c_train] == [x["effect"] for x in u_train]
    assert [x["state_after"] for x in c_train] == [x["state_after"] for x in u_train]
    assert all("requested_action" in x and "budget_forced_wait" in x for x in c["steps"])


def test_diversity_axis_reuses_identical_latent_plans():
    from types import SimpleNamespace
    from src.experiment import run_pair

    args = SimpleNamespace(train_episodes=3, train_steps=4, test_steps=3, budget=3)
    concentrated = asyncio.run(
        run_pair(asyncio.Semaphore(1), CyclingAgent(), 11, "concentrated", args)
    )[0]
    distributed = asyncio.run(
        run_pair(asyncio.Semaphore(1), CyclingAgent(), 11, "distributed", args)
    )[0]

    def episode_signature(subject, phase="train"):
        rows = [x for x in subject["steps"] if x["phase"] == phase and x["step_idx"] == 0]
        return [(x["state_before"], x["hidden_action_effects"]) for x in rows]

    assert episode_signature(concentrated) == episode_signature(distributed)
    assert episode_signature(concentrated, "test") == episode_signature(distributed, "test")
    assert concentrated["train_families"] != distributed["train_families"]
