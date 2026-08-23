from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g0_bias_trap_screen import normalize, resolve_diagnosis
from g0_pair_locality import pair_metrics

def test_resolve_unique_diagnosis():
    labels=['Pneumonia','Pulmonary embolism','Stable angina']
    assert resolve_diagnosis('Diagnosis: Pneumonia',labels)=='Pneumonia'
    assert resolve_diagnosis('stable angina',labels)=='Stable angina'

def test_pair_metrics_detect_local_flip():
    c={'case_id':'x','narrative':'Age 50. Fever present. Cough present.','ground_truth':'Pneumonia','age':50,'sex':'F'}
    t={'case_id':'x','narrative':'Age 50. Fever absent. Cough present.','ground_truth':'Pulmonary embolism','age':50,'sex':'F'}
    m=pair_metrics(c,t)
    assert m['gt_flips'] and m['same_age'] and m['same_sex']
    assert 0 < m['changed_token_fraction'] < 0.5

def test_normalize():
    assert normalize(' Possible NSTEMI / STEMI. ')=='possible nstemi / stemi'
