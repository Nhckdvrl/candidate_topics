import sys, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from schedule import build_schedules

class ScheduleTests(unittest.TestCase):
    def test_exact_multiset_fixed_unique_and_no_within_step_repeat_collision(self):
        spec,s,a=build_schedules(10000,0.1,50,7,blocks_per_optimizer_step=8)
        self.assertEqual(spec.repeat_docs,20)
        self.assertEqual(spec.max_repeat_slots_per_optimizer_step,1)
        for c in ("clustered","random","even"):
            self.assertEqual(a["conditions"][c]["max_repeat_slots_same_optimizer_step"],1)
        self.assertEqual(a["conditions"]["clustered"]["repeated_multiset_sha256"],a["conditions"]["random"]["repeated_multiset_sha256"])
        self.assertEqual(a["conditions"]["random"]["repeated_multiset_sha256"],a["conditions"]["even"]["repeated_multiset_sha256"])
        self.assertEqual(len({a["conditions"][c]["unique_slots_sha256"] for c in ("fresh","clustered","random","even")}),1)

    def test_spacing_is_large_at_optimizer_step_scale(self):
        _,_,a=build_schedules(20000,0.1,40,11,blocks_per_optimizer_step=8)
        cg=a["conditions"]["clustered"]["spacing"]["mean_gap_optimizer_steps"]
        eg=a["conditions"]["even"]["spacing"]["mean_gap_optimizer_steps"]
        self.assertLess(cg,eg)
        self.assertGreater(eg/cg,5.0)

    def test_trial_seed_changes_repeated_pool(self):
        _,_,a1=build_schedules(10000,0.1,50,7,blocks_per_optimizer_step=8)
        _,_,a2=build_schedules(10000,0.1,50,8,blocks_per_optimizer_step=8)
        self.assertNotEqual(a1["repeat_doc_ids_sha256"],a2["repeat_doc_ids_sha256"])

    def test_fails_if_repeat_fraction_requires_multiple_slots_per_step(self):
        with self.assertRaises(ValueError):
            build_schedules(10000,0.2,50,7,blocks_per_optimizer_step=8)

if __name__=="__main__": unittest.main()
