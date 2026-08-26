import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from decision_state import classify_state, transition

def test_states():
    assert classify_state('I propose that we ship in June.').state=='PROPOSED'
    assert classify_state('We decided to ship in June.').state=='DECIDED'
    assert classify_state('We agreed to ship if legal approves.').state=='CONDITIONAL'
    assert classify_state('We have not decided yet.').state=='OPEN'
    assert classify_state('We rejected the June launch.').state=='REJECTED'

def test_upgrade_and_condition_loss():
    x=transition('We could ship if legal approves.','We will ship.')
    assert x['upgrade'] and x['conditionality_lost']
