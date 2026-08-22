import sys, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from schedule import build_schedules

class ScheduleTests(unittest.TestCase):
    def test_exact_multiset_and_fixed_unique_positions(self):
        spec,s,a=build_schedules(10000,0.1,50,7)
        self.assertEqual(spec.repeat_docs,20)
        for c in ("clustered","random","even"):
            vals,counts=np.unique(s[c][np.isin(s[c],np.arange(spec.repeat_docs))],return_counts=True)
            self.assertEqual(len(vals),spec.repeat_docs); self.assertTrue(np.all(counts==spec.repeat_count))
        self.assertEqual(a["conditions"]["clustered"]["repeated_multiset_sha256"],a["conditions"]["random"]["repeated_multiset_sha256"])
        self.assertEqual(a["conditions"]["random"]["repeated_multiset_sha256"],a["conditions"]["even"]["repeated_multiset_sha256"])
        self.assertEqual(len({a["conditions"][c]["unique_slots_sha256"] for c in ("fresh","clustered","random","even")}),1)

    def test_spacing_is_real(self):
        _,_,a=build_schedules(20000,0.1,40,11)
        cg=a["conditions"]["clustered"]["spacing"]["mean_gap_blocks"]; eg=a["conditions"]["even"]["spacing"]["mean_gap_blocks"]
        self.assertLess(cg,eg); self.assertGreater(eg/cg,5.0)

if __name__=="__main__": unittest.main()
