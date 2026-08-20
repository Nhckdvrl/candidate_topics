import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
from stage2_generate import select_indices, dream_transfer_count, question_and_gold
from fate_labels import fate_from_correctness
from stage2_surface_gate import locked_support, apply_support_gate
from stage2_confirm import evaluate_locked_task, SPECS


def test_dataset_selection_and_gold():
    assert select_indices(1319, 1000, 319).tolist() == list(range(1000, 1319))
    q,g=question_and_gold('gsm8k',{'question':'q','answer':'work #### 42'})
    assert (q,g)==('q','42')
    q,g=question_and_gold('gsm1k',{'question':'q2','answer':'1,234'})
    assert (q,g)==('q2','1234')


def test_dream_transfer_schedule_matches_official_linear_maskgit_plus():
    remaining=128
    moved=[]
    timetable=np.linspace(1.0,1e-3,65)
    for step in range(64):
        k=dream_transfer_count(remaining,step,64)
        expected = remaining if step == 63 else min(
            remaining, int(remaining * (1.0 - timetable[step + 1] / timetable[step]))
        )
        assert k == expected
        moved.append(k)
        remaining -= k
    assert remaining == 0
    assert sum(moved) == 128
    assert SPECS['dream']['transient_recovery']['layer'] == 22
    assert SPECS['dream']['transient_overwrite']['layer'] == 25


def test_full_support_gate_allows_one_task_without_moving_locked_cells():
    rows=[
        {'task':'transient_recovery','step':16,'min_lead':4,'n':80,'positive':30,'negative':50},
        {'task':'transient_overwrite','step':4,'min_lead':16,'n':35,'positive':10,'negative':25},
    ]
    status, annotated=apply_support_gate(rows,25,25)
    assert status == 'GO_ONE'
    assert annotated[0]['support_ok'] is True
    assert annotated[1]['support_ok'] is False

    status, _=apply_support_gate(rows,31,51)
    assert status == 'STOP_LOW_LOCKED_SUPPORT'

    both=[dict(rows[0]), {'task':'transient_overwrite','step':4,'min_lead':16,'n':70,'positive':30,'negative':40}]
    status, annotated=apply_support_gate(both,25,25)
    assert status == 'GO_BOTH'
    assert all(r['support_ok'] for r in annotated)


def _synthetic_data():
    rng=np.random.default_rng(4)
    n_each=60
    traces=[]
    for _ in range(n_each):
        x=np.zeros(64,dtype=bool); x[20:24]=True; traces.append(x)
    for _ in range(n_each): traces.append(np.zeros(64,dtype=bool))
    for _ in range(n_each):
        x=np.ones(64,dtype=bool); x[20:24]=False; traces.append(x)
    for _ in range(n_each): traces.append(np.ones(64,dtype=bool))
    c=np.stack(traces); observed=np.ones_like(c)
    capture=np.array([0,4,16,63],dtype=np.int16)
    labels=fate_from_correctness(c,capture,observed)
    n=len(c); layers=np.array([25,28],dtype=np.int16)
    hidden=rng.normal(size=(n,4,2,1,12)).astype(np.float32)
    rec=labels['transient_recovery'][:,2].astype(float)
    ov=labels['transient_overwrite'][:,1].astype(float)
    rec_valid=rec>=0; ov_valid=ov>=0
    hidden[rec_valid,2,0,0,0]=rec[rec_valid]*7+rng.normal(scale=.15,size=rec_valid.sum())
    hidden[ov_valid,1,1,0,0]=ov[ov_valid]*7+rng.normal(scale=.15,size=ov_valid.sum())
    final=c[:,-1].astype(float)
    hidden[:,3,0,0,1]=final*5+rng.normal(scale=.2,size=n)
    hidden[:,3,1,0,1]=final*5+rng.normal(scale=.2,size=n)
    return {
      'problem_id':np.arange(n),'capture_steps':capture,'hidden_indices':layers,'hidden':hidden,
      'entropy':rng.normal(size=(n,4)).astype(np.float32),
      'selected_prob':rng.uniform(size=(n,4)).astype(np.float32),
      'clean_maxprob':rng.uniform(size=(n,4)).astype(np.float32),
      'frac_unmasked':np.tile(np.array([0,.06,.25,.98],dtype=np.float32),(n,1)),
      'prompt_tokens':rng.integers(20,60,size=n),
      'correct_strict':c,'observed_strict':observed,
    }, labels


def test_locked_surface_support_and_confirmation():
    data,labels=_synthetic_data()
    rows=locked_support(labels,data['capture_steps'])
    by={r['task']:r for r in rows}
    assert by['transient_recovery']['positive']==60
    assert by['transient_overwrite']['positive']==60
    r=evaluate_locked_task(data,labels,'transient_recovery',SPECS['llada']['transient_recovery'],4,300)
    o=evaluate_locked_task(data,labels,'transient_overwrite',SPECS['llada']['transient_overwrite'],4,300)
    assert r['auc']>.95 and r['confirmation_margin_lo_97_5']>0
    assert o['auc']>.95 and o['confirmation_margin_lo_97_5']>0
