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

def test_multiple_modalities_are_not_silently_collapsed():
    parsed=extract('The employee must file the report and may attach exhibits.')
    assert parsed.modality=='MULTIPLE'
    assert set(parsed.modalities)=={'OBLIGATION','PERMISSION'}
    drift=compare('The employee must file and may appeal.','The employee must file.')
    assert drift['modality_changed']
    assert drift['modalities_lost']==['PERMISSION']

def test_actor_marker_is_preserved_when_available():
    assert extract('[tenant] Tenant shall pay rent.').actor=='tenant'
