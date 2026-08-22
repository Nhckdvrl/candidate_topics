import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_ablation import complete_file, ensure_run_contract


def test_run_contract_is_idempotent_and_rejects_protocol_drift(tmp_path):
    a = {"seed": 1, "prompt": "locked"}
    first = ensure_run_contract(tmp_path, a)
    second = ensure_run_contract(tmp_path, dict(a))
    assert first == second
    with pytest.raises(RuntimeError):
        ensure_run_contract(tmp_path, {"seed": 2, "prompt": "locked"})


def test_resume_requires_matching_contract_id(tmp_path):
    path = tmp_path / "layer_00.jsonl"
    path.write_text(json.dumps({"contract_id": "A"}) + "\n", encoding="utf-8")
    assert complete_file(path, 1, "A")
    assert not complete_file(path, 1, "B")
