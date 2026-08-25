import importlib.util
import json
import pathlib
import tempfile
import unittest
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("g0", ROOT / "g0_temporal_scope.py")
g0 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = g0
spec.loader.exec_module(g0)

def turn(q, a, year, pid, subj, present):
    return {"question": q, "answer": a, "year": year, "pid": pid, "subject_label": subj,
            "present_day_answer": present, "drift_eligible": a.lower() != present.lower()}

class TestG0(unittest.TestCase):
    def synthetic(self, n=4):
        chains=[]
        for i in range(n):
            subj=f"Target{i}"
            chains.append({"chain_id":f"t{i}","family":"carryover","truth_type":"temporal","snapshot_year":2000,
                "turns":[turn(f"In 2000, who was CEO of {subj}?",f"OldCEO{i}",2000,"P169",subj,f"NewCEO{i}"),
                         turn("Who owned it?",f"OldOwner{i}",2000,"P127",subj,f"NewOwner{i}")]})
            chains.append({"chain_id":f"s{i}","family":"carryover_then","truth_type":"temporal","snapshot_year":2000,
                "turns":[turn(f"In 2000, what country was {subj} in?",f"Country{i}",2000,"P17",subj,f"Country{i}"),
                         turn("Which country was it in then?",f"Country{i}",2000,"P17",subj,f"Country{i}")]})
        for j in range(8):
            subj=f"Other{j}"
            chains.append({"chain_id":f"o{j}","family":"carryover_then","truth_type":"temporal","snapshot_year":2000,
                "turns":[turn(f"In 2000, what country was {subj} in?",f"Land{j}",2000,"P17",subj,f"Land{j}"),
                         turn("Which country was it in then?",f"Land{j}",2000,"P17",subj,f"Land{j}")]})
        return chains

    def test_panel_exact_and_deterministic(self):
        p1,r1=g0.build_panel(self.synthetic(),n=4,seed=7)
        p2,r2=g0.build_panel(self.synthetic(),n=4,seed=7)
        self.assertTrue(r1["hard_gate_pass"])
        self.assertEqual([x.item_id for x in p1],[x.item_id for x in p2])
        self.assertEqual(len(p1),4)
        for x in p1:
            self.assertNotEqual(g0.norm(x.historical_answer), g0.norm(x.present_answer))
            self.assertTrue(x.stable_fact["stable"])
            self.assertNotEqual(x.stable_fact["pid"], x.target_pid)

    def test_probe_is_byte_identical_across_conditions(self):
        p,_=g0.build_panel(self.synthetic(),n=1,seed=1)
        probes=[g0.build_messages(p[0],c)[-1]["content"] for c in g0.CONDITIONS]
        self.assertEqual(len(set(probes)),1)

    def test_present_uses_same_stable_fact_as_semantic(self):
        p,_=g0.build_panel(self.synthetic(),n=1,seed=1)
        sem=g0.build_messages(p[0],"same_entity_semantic")[-3]["content"]
        pre=g0.build_messages(p[0],"bounded_present")[-3]["content"]
        self.assertIn(p[0].stable_fact["answer"],sem)
        self.assertIn(p[0].stable_fact["answer"],pre)
        self.assertIn("2025",pre)
        self.assertNotIn("2025",sem)

    def test_summary_contrast_signs(self):
        rows=[]
        for i in range(20):
            c={}
            vals={"baseline":1,"neutral_1":1,"neutral_2":1,"neutral_4":1,
                  "same_entity_semantic":0,"bounded_present":0,"bounded_present_reinstate":1}
            for k,v in vals.items(): c[k]={"correct":bool(v),"present_drift":False}
            rows.append({"conditions":c})
        s=g0.summarize(rows)
        self.assertEqual(s["contrasts"]["same_entity_penalty"]["delta"],1.0)
        self.assertEqual(s["contrasts"]["reinstatement_gain"]["delta"],1.0)

if __name__ == "__main__":
    unittest.main()
