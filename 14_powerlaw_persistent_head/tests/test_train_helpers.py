from pathlib import Path

from train import latest_checkpoint


def test_latest_checkpoint_uses_numeric_step_order(tmp_path: Path):
    for step in [20, 100, 5]:
        (tmp_path / f"checkpoint_{step}.pt").write_bytes(b"x")
    assert latest_checkpoint(tmp_path).name == "checkpoint_100.pt"


def test_latest_checkpoint_none_when_empty(tmp_path: Path):
    assert latest_checkpoint(tmp_path) is None
