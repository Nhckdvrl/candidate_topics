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
    assert classify_state('We talked about the June launch.').state=='UNKNOWN'
    assert classify_state("We haven't decided yet.").state=='OPEN'

def test_upgrade_and_condition_loss():
    x=transition('We could ship if legal approves.','We will ship.')
    assert x['upgrade'] and x['conditionality_lost']

def test_rejection_is_not_an_ordinal_low_state():
    x=transition('We rejected the June launch.','We might revisit the June launch.')
    assert not x['upgrade']
    assert not x['downgrade']

def test_uncued_source_is_not_scorable():
    x=transition('We talked about launch timing.','We decided to launch in June.')
    assert not x['source_scorable']
    assert not x['upgrade']

def test_summary_subject_variants_and_negated_finality():
    assert classify_state('The meeting agreed to use red.', genre='summary').state == 'DECIDED'
    assert classify_state('Participants decided to use red.', genre='summary').state == 'DECIDED'
    assert classify_state('A decision was made to use red.', genre='summary').state == 'DECIDED'
    assert classify_state('They did not make a final decision.', genre='summary').state == 'OPEN'

def test_decision_to_consider_is_not_adoption():
    parsed = classify_state('The team decided to consider using red.', genre='summary')
    assert parsed.state != 'DECIDED'
    assert not transition('We could use red.', 'The team decided to consider using red.')['upgrade']
