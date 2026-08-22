import sys, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analyze import trial_effects, verdict

def rows(fresh,clustered,random,even,noise=0.002):
    fp="same";n=256;base=np.linspace(2.8,3.2,n);out={}
    for c,v in {"fresh":fresh,"clustered":clustered,"random":random,"even":even}.items():
        ripple=noise*np.sin(np.arange(n)+(0 if c=="fresh" else {"random":1,"clustered":2,"even":3}[c]));x=base+(v-3.0)+ripple;out[c]={"final_eval_loss":float(x.mean()),"final_eval_se_blocks":float(x.std(ddof=1)/np.sqrt(n)),"init_fingerprint":fp,"experiment_id":"x","_eval_block_losses":x}
    return out
class AnalysisTests(unittest.TestCase):
    def test_pilot_null_does_not_kill(self):
        e=trial_effects(rows(3.0,3.061,3.06,3.0605));self.assertTrue(e["seed_reproduced"]);self.assertIn("RUN_CONFIRMATION",verdict([e],"pilot",1)[0])
    def test_setup_fail_is_not_topic_fail(self):
        e=trial_effects(rows(3.0,3.03,3.0001,3.02));self.assertFalse(e["seed_reproduced"]);self.assertIn("NOT_TOPIC_FAIL",verdict([e],"pilot",1)[0])
    def test_strong_confirmation(self):
        es=[trial_effects(rows(3.0,3.09,3.06,3.02)),trial_effects(rows(3.01,3.095,3.07,3.03)),trial_effects(rows(2.99,3.08,3.05,3.015))];self.assertEqual(verdict(es,"confirm",3)[0],"GO_STRONG_SPACING_IS_CAUSAL")
    def test_mixed_directions_not_go(self):
        es=[trial_effects(rows(3.0,3.09,3.06,3.02)),trial_effects(rows(3.01,3.01,3.07,3.05)),trial_effects(rows(2.99,3.01,3.05,3.08))];self.assertNotEqual(verdict(es,"confirm",3)[0],"GO_STRONG_SPACING_IS_CAUSAL")
    def test_confirmation_requires_all_preregistered_trials(self):
        es=[trial_effects(rows(3.0,3.09,3.06,3.02)),trial_effects(rows(3.01,3.10,3.07,3.03)),trial_effects(rows(2.99,3.08,3.05,3.01))];self.assertEqual(verdict(es,"confirm",4)[0],"CONFIRM_INCOMPLETE")
if __name__=="__main__":unittest.main()
