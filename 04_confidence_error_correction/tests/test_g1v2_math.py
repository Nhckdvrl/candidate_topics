#!/usr/bin/env python3
"""Small deterministic checks for Topic 04 G-1v2 measurement math."""
from __future__ import annotations

import math
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from mcq_utils import (
    balanced_permutations,
    geometric_mean_distribution,
    js_divergence,
)


def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]


class G1V2MathTest(unittest.TestCase):
    def test_balanced_permutations_cover_positions(self):
        for scheme in ("cyclic", "hashed_cyclic"):
            perms = balanced_permutations(10, scheme=scheme, item_id="example")
            self.assertEqual(len(perms), 10)
            for semantic in range(10):
                positions = [perm.index(semantic) for perm in perms]
                self.assertEqual(sorted(positions), list(range(10)))

    def test_logmean_exactly_removes_additive_position_bias(self):
        alpha = [2.0, 1.0, 0.2, -0.3]
        beta = [3.0, 0.5, -1.0, -2.0]
        mapped = []
        for perm in balanced_permutations(4, "cyclic", "x"):
            local = softmax([alpha[perm[pos]] + beta[pos] for pos in range(4)])
            semantic = [0.0] * 4
            for pos, original in enumerate(perm):
                semantic[original] = local[pos]
            mapped.append(semantic)

        recovered = geometric_mean_distribution(mapped)
        target = softmax(alpha)
        self.assertLess(js_divergence(recovered, target), 1e-12)

    def test_hashed_family_is_deterministic(self):
        a = balanced_permutations(10, "hashed_cyclic", "item-17")
        b = balanced_permutations(10, "hashed_cyclic", "item-17")
        c = balanced_permutations(10, "hashed_cyclic", "item-18")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


if __name__ == "__main__":
    unittest.main()
