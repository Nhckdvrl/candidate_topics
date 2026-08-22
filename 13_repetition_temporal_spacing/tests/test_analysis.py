import sys, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from analyze import trial_effects, verdict

def rows(fresh,clustered,random,even,se=1e-5,paired=False):
    fp="same"; out={c:{"final_eval_loss":v,"final_eval_se_blocks":se,"init_fingerprint":fp} for c,v in {"fresh":fresh,"clustered":clustered,"random":random,"even":even}.items()}
    if paired:
        base=np.linspace(2.9,3.1,64)
        for c,v in {"fresh":fresh,"clustered":clustered,"random":random,"even":even}.items(): out[c]["_eval_block_losses"]=base+(v-3.0)
    return out

class AnalysisTests(unittest.TestCase):
    def test_pilot_promising(self):
        e=trial_effects(rows(3.0,3.09,3.06,3.02)); self.assertTrue(e["seed_reproduced"]); self.assertTrue(e["large_spacing_effect"]); self.assertEqual(verdict([e])[0],"PILOT_PROMISING_RUN_CONFIRMATION")
    def test_seed_gate_prevents_spacing_story(self):
        e=trial_effects(rows(3.0,3.10,3.001,2.90)); self.assertFalse(e["seed_reproduced"]); self.assertIn("SETUP_FAIL",verdict([e])[0])
    def test_confirmation(self):
        es=[trial_effects(rows(3.0,3.09,3.06,3.02)),trial_effects(rows(3.01,3.10,3.07,3.03)),trial_effects(rows(2.99,3.08,3.05,3.01))]; self.assertEqual(verdict(es)[0],"GO_SPACING_IS_CAUSAL")
    def test_paired_eval_noise_is_used(self):
        e=trial_effects(rows(3.0,3.09,3.06,3.02,paired=True)); self.assertEqual(e["seed_noise_mode"],"paired_block"); self.assertEqual(e["spacing_noise_mode"],"paired_block"); self.assertLess(e["seed_difference_se"],1e-10)

if __name__=="__main__": unittest.main()
