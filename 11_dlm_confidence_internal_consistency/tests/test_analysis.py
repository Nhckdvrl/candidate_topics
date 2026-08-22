import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analyze import orientation_effect,verdict,sign_flip_pvalue,PRIMARY_METRIC

class AnalysisV3Test(unittest.TestCase):
    def test_effect_algebra(self):
        e=orientation_effect({"CC":.9,"IC":.7,"CW":.8,"IW":.6})
        self.assertAlmostEqual(e["delta_consistency"],.2); self.assertAlmostEqual(e["delta_correctness"],.1)
        self.assertAlmostEqual(e["coherent_wrong_minus_incoherent_correct"],.1)
    def _protocol(self,gap=.2,lo=.15):
        return {"arithmetic_result":{"gap_mean":gap,"gap_ci95":[lo,gap+.05]},"semantic_alias":{"gap_mean":gap,"gap_ci95":[lo,gap+.05]}}
    def _metrics(self,mean=.03,ci=(.02,.04),cross=(.01,.03),c1=.03,c0=.03):
        return {PRIMARY_METRIC:{"effects":{"delta_consistency":{"mean":mean,"ci95":list(ci)},"consistency_when_correct":{"mean":c1,"ci95":[.0,.06]},"consistency_when_wrong":{"mean":c0,"ci95":[.0,.06]},"coherent_wrong_minus_incoherent_correct":{"mean":sum(cross)/2,"ci95":list(cross)}}}}
    def test_bad_protocol_never_kills_science(self):
        v,_=verdict(self._protocol(gap=.01,lo=.005),self._metrics(),.10,.02,.01); self.assertEqual(v,"INVALID_PROTOCOL_DO_NOT_INTERPRET")
    def test_strong_go(self):
        v,_=verdict(self._protocol(),self._metrics(),.10,.02,.01); self.assertEqual(v,"GO_STRONG_COHERENCE_OVER_CORRECTNESS")
    def test_topic_stands_without_dominance(self):
        v,_=verdict(self._protocol(),self._metrics(cross=(-.01,.02)),.10,.02,.01); self.assertEqual(v,"GO_RETROACTIVE_CONSISTENCY_SIGNAL")
    def test_equivalence_style_kill_excludes_meaningful_effect(self):
        v,_=verdict(self._protocol(),self._metrics(mean=.002,ci=(-.002,.008)),.10,.02,.01); self.assertEqual(v,"KILL_NO_MEANINGFUL_RETROACTIVE_SIGNAL")
    def test_uncertain_if_meaningful_effect_not_excluded(self):
        v,_=verdict(self._protocol(),self._metrics(mean=.008,ci=(-.003,.02)),.10,.02,.01); self.assertEqual(v,"INCONCLUSIVE_FROZEN_DESIGN")
    def test_signflip(self):
        p=sign_flip_pvalue(np.linspace(.1,.3,20),np.random.default_rng(1),5000); self.assertLess(p,.01)

if __name__=="__main__":unittest.main()
