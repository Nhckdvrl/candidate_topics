import unittest

import numpy as np

import shared_truthful_clue_traps_preflight as shared


class TestFamilies(unittest.TestCase):
    def test_frozen_family_rules(self):
        cases = {
            "llama-2-7b_1shot": "llama",
            "meta-llama-3-70b_1shot": "llama",
            "rag-bm25_top3-flan-ul2": "rag_bm25",
            "contriever_ctx-recall@3": "retrieval_contriever",
            "T0pp-11b_1shot": "t5_t0_ul2",
            "pythia-6.9b_1shot": "pythia",
        }
        for config, expected in cases.items():
            self.assertEqual(shared.config_family(config), expected)

    def test_unknown_family_fails(self):
        with self.assertRaises(ValueError):
            shared.config_family("unknown-model")


class TestPermutation(unittest.TestCase):
    def test_wrong_answer_consensus_counts_distinct_families(self):
        encoded = {
            "boundaries": ["b0"],
            "families": ["f0", "f1", "f2"],
            "bf": np.array([0, 0, 1, 2]),
            "b": np.array([0, 0, 0, 0]),
            "f": np.array([0, 0, 1, 2]),
            "risk_bf": np.array([[2, 1, 1]]),
            "risk_b": np.array([4]),
            "risk_families": np.array([3]),
            "pair_opportunities": 3.0,
            "config_pair_opportunities": 6.0,
            "wrong_values": ["same wrong", "other wrong"],
        }
        reversal = np.array([True, True, True, True])
        wrong = np.array([0, 0, 0, 1])
        metrics = shared.compute_metrics(encoded, reversal, wrong)
        self.assertEqual(metrics["top_wrong_count"].tolist(), [3])
        self.assertEqual(metrics["top_wrong_families"].tolist(), [2])

    def test_payload_permutation_preserves_stratum_multisets(self):
        reversal = np.array([True, False, True, False, True])
        wrong = np.array([4, -1, 7, -1, 9])
        strata = [np.array([0, 1, 2]), np.array([3, 4])]
        got_r, got_w = shared.permute_payload(
            reversal, wrong, strata, np.random.default_rng(3)
        )
        for index in strata:
            self.assertEqual(sorted(reversal[index]), sorted(got_r[index]))
            self.assertEqual(sorted(wrong[index]), sorted(got_w[index]))
            self.assertEqual(
                sorted(zip(reversal[index], wrong[index])),
                sorted(zip(got_r[index], got_w[index])),
            )

    def test_plus_one_p_has_nonzero_floor(self):
        null = np.array([0.1, 0.2, 0.3])
        self.assertEqual(shared.plus_one_p(null, 1.0), 0.25)


if __name__ == "__main__":
    unittest.main()
