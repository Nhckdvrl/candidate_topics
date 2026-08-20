import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "build_forgotten_set.py"


class ForgottenSetTests(unittest.TestCase):
    def test_incomplete_checkpoint_is_not_never_correct(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "x.jsonl"
            out = Path(td) / "out.jsonl"
            rows = []
            # Problem A has all 3 checkpoints and is genuinely never-correct.
            for order in range(3):
                for i in range(8):
                    rows.append({
                        "problem_id": "A", "checkpoint": f"s{order}",
                        "checkpoint_order": order, "response": "x",
                        "correct": "false", "prompt": "p", "gold_answer": "1"
                    })
            # Problem B is missing checkpoint 1; old implementation could admit it.
            for order in [0, 2]:
                for i in range(8):
                    rows.append({
                        "problem_id": "B", "checkpoint": f"s{order}",
                        "checkpoint_order": order, "response": "x",
                        "correct": False, "prompt": "p", "gold_answer": "1"
                    })
            inp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            subprocess.run([
                sys.executable, str(SCRIPT), "--input", str(inp), "--output", str(out),
                "--min-samples", "8"
            ], check=True, cwd=ROOT)
            result = [json.loads(x) for x in out.read_text().splitlines()]
            by = {r["problem_id"]: r for r in result}
            self.assertEqual(by["A"]["group"], "never_correct")
            self.assertNotIn("B", by)


if __name__ == "__main__":
    unittest.main()
