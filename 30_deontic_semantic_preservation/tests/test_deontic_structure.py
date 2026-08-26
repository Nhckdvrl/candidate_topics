import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from deontic_structure import extract,compare

def test_modalities():
    assert extract('The employee must file the report.').modality=='OBLIGATION'
    assert extract('The employee may file the report.').modality=='PERMISSION'
    assert extract('The employee must not disclose the report.').modality=='PROHIBITION'

def test_condition_and_exception_loss():
    assert compare('The employee may enter only if approved.','The employee may enter.')['condition_lost']
    assert compare('Employees must report, except during leave.','Employees must report.')['exception_lost']
