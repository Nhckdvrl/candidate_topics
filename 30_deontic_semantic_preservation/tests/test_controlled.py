import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_g0_controlled import TRANSFORMS, _gold_span_edit


def test_perturbations_are_directionally_normative():
    names = {name for transforms in TRANSFORMS.values() for _, name in transforms.values()}
    assert "OBLIGATION_TO_PERMISSION" in names
    assert "PERMISSION_TO_OBLIGATION" in names
    assert "PROHIBITION_LOSS" in names


def test_edit_uses_gold_span_not_earlier_modal():
    text = "[tenant] Landlord may inspect and Tenant shall pay the fee ."
    edit = _gold_span_edit(text, {"obl": [[5, 6]]}, "obl")
    assert edit is not None
    original, perturbed, _, span, trigger = edit
    assert original.startswith("Landlord may")
    assert "Landlord may inspect and Tenant may pay" in perturbed
    assert span == [5, 6]
    assert trigger == "shall"


def test_entitlement_is_not_naively_perturbed():
    assert "ent" not in TRANSFORMS
