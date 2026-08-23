from pathlib import Path
import importlib.util
import sys
import pytest


ROOT = Path(__file__).parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(f"topic18_{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generate = load("generate_g0")
score = load("score_g0")


def outputs(design, negative_correct=True):
    result = {}
    for item_id, row in design.items():
        if row["condition"] == "positive":
            choice = row["marked_action"]
        elif row["condition"] == "negative":
            if negative_correct:
                choice = row["neutral_action"]
            else:
                # Half of surface pairs fail, evenly inside every nuisance stratum.
                pair_index = generate.ACTION_PAIRS.index((row["action_a"], row["action_b"]))
                choice = row["marked_action"] if pair_index < 4 else row["neutral_action"]
        else:
            # Balanced within each symbol pair and independent of marked identity.
            choice = row["action_a"] if row["choice_order"] == "ab" else row["action_b"]
        result[item_id] = choice
    return result


def test_generator_is_complete_crossed_and_has_baseline():
    rows = generate.build_rows(64, 7)
    assert len(rows) == 192
    design = score.validate_design(rows)
    assert {r["condition"] for r in rows} == {"positive", "negative", "baseline"}
    assert all("choose the command with the better" not in r["prompt"].lower() for r in rows)
    assert len({
        (r["action_a"], r["action_b"], r["marked_action"],
         r["observation_order"], r["choice_order"])
        for r in rows if r["condition"] == "positive"
    }) == 64
    assert len(design) == 192


def test_perfect_symmetric_adaptation_kills_asymmetry_claim():
    design = score.validate_design(generate.build_rows(64, 7))
    models = []
    for i in range(3):
        result, _ = score.score_model(f"m{i}", design, outputs(design), 200, i)
        models.append(result)
    assert score.decide(models, 64, True)["verdict"] == "KILL"


def test_large_robust_gap_survives():
    design = score.validate_design(generate.build_rows(64, 7))
    models = []
    for i in range(3):
        result, _ = score.score_model(
            f"m{i}", design, outputs(design, negative_correct=False), 200, i
        )
        models.append(result)
    decision = score.decide(models, 64, True)
    assert decision["verdict"] == "SURVIVE"
    assert decision["pooled_delta"] == .5


def test_prompt_condition_confound_is_rejected():
    rows = generate.build_rows(64, 7)
    rows[0]["prompt"] += " Condition-specific hint."
    with pytest.raises(ValueError, match="prompt text differs"):
        score.validate_design(rows)


def test_three_model_ids_from_one_family_are_invalid():
    design = score.validate_design(generate.build_rows(64, 7))
    models = []
    for i in range(3):
        result, _ = score.score_model(
            f"m{i}", design, outputs(design), 200, i, model_family="same-family"
        )
        models.append(result)
    assert score.decide(models, 64, True)["verdict"] == "INVALID"
