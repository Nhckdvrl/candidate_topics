import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_g0_temporal_prefix import content_recall, load_candidates


def test_candidate_withholds_final_turn(tmp_path):
    data = [{
        "abstractive": {"id": "m.s.1", "type": "decisions", "text": "Use red."},
        "extractive": [
            {"speaker": "A", "starttime": "0", "endtime": "2", "text": "Maybe we could use red."},
            {"speaker": "B", "starttime": "3", "endtime": "4", "text": "Yes, we have decided on red."},
        ],
    }]
    tmp_path.joinpath("m.json").write_text(json.dumps(data), encoding="utf-8")
    rows = load_candidates(tmp_path)
    assert len(rows) == 1
    assert rows[0]["cut_turn"] == 1
    assert "decided" not in rows[0]["prefix_text"]


def test_content_recall():
    assert content_recall("Maybe use the red plastic case", "The team will use a red case.") > 0
