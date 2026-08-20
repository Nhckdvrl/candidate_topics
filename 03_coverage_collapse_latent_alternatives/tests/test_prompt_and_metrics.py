import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from prompt_utils import OPENING_SENTENCES, full_prompt, mask_query_target
from analyze_sampled_branches import pass_at_k_from_counts


def test_all_upstream_opening_prefixes_end_at_same_decision_slot():
    prompts = [full_prompt("Q?", i) for i in range(3)]
    assert len(set(prompts)) == 3
    for i, p in enumerate(prompts):
        assert OPENING_SENTENCES[i] in p
        assert p.endswith("\n1.")


def test_mask_target_only_changes_query_identity():
    q = "Consider: p = 3; c = p + 4; d = c + 5; x = p + 8; y = x + 2. If p = 3, determine the value of d."
    masked = mask_query_target(q, "d")
    assert "d = c + 5" in masked
    assert "determine the value of d" not in masked
    assert "determine the value of the requested variable" in masked


def test_pass_at_k_known_cases():
    assert pass_at_k_from_counts(16, 0, 8) == 0.0
    assert pass_at_k_from_counts(16, 16, 8) == 1.0
    assert np.isclose(pass_at_k_from_counts(16, 4, 1), 0.25)
    assert pass_at_k_from_counts(16, 1, 16) == 1.0
