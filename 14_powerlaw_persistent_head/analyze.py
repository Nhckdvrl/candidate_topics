#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

ARMS=['uniform','static','slow','fast']
def read_csv(p):
    with p.open() as f:return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
def auc(rows,key):
    x=np.array([r['step'] for r in rows]);y=np.array([r[key] for r in rows]);trap=getattr(np,'trapezoid',np.trapz);return float(trap(y,x)/(x[-1]-x[0]))
def one(run):
    rows=read_csv(run/'metrics.csv');cfg=json.loads((run/'config.json').read_text());done=json.loads((run/'done.json').read_text());return {'auc_exact':auc(rows,'exact_accuracy'),'auc_token':auc(rows,'token_accuracy'),'final_exact':rows[-1]['exact_accuracy'],'final_token':rows[-1]['token_accuracy'],'branch_digest':cfg['branch_digest'],'cfg':cfg,'done':done}
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--profile',required=True);a=p.parse_args();seeds={}
    for sd in sorted(a.root.glob('seed*')):
        d={}
        for arm in ARMS:
            if (sd/arm/'done.json').exists():d[arm]=one(sd/arm)
        if d:seeds[sd.name]=d
    rep={'profile':a.profile,'seeds':{},'decision':None};diffs=[];anchor=[];bad=[]
    for s,d in seeds.items():
        if all(x in d for x in ARMS):
            if len({d[x]['branch_digest'] for x in ARMS})!=1:bad.append(s)
            ps=d['slow']['cfg']['schedule'];pf=d['fast']['cfg']['schedule']
            if ps['multiset_digest']!=pf['multiset_digest'] or ps['temporal_digest']==pf['temporal_digest']:bad.append(s+':schedule')
            de=d['slow']['auc_exact']-d['fast']['auc_exact'];ag=d['static']['auc_exact']-d['uniform']['auc_exact'];diffs.append(de);anchor.append(ag)
            rep['seeds'][s]={'anchor_static_minus_uniform_auc_exact':ag,'slow_minus_fast_auc_exact':de,'metrics':{k:{q:v[q] for q in ['auc_exact','auc_token','final_exact','final_token']} for k,v in d.items()}}
    if bad:rep['decision']='TECHNICAL_INVALID_BRANCH_IDENTITY';rep['integrity_failures']=bad
    elif a.profile=='smoke':rep['decision']='SMOKE_ONLY_DO_NOT_INTERPRET'
    elif a.profile=='pilot':rep['decision']='PILOT_SIGNAL_ONLY_DO_NOT_CONCLUDE'
    elif a.profile!='full':rep['decision']='DIAGNOSTIC_ONLY'
    elif len(diffs)<5:rep['decision']='INCOMPLETE_NEED_5_SEEDS'
    else:
        dif=np.array(diffs);anc=np.array(anchor);anchor_ok=(np.median(anc)>=0.03 and np.mean(anc>0)>=0.8);pos=(np.median(dif)>=0.10 and np.mean(dif>0)>=0.8);neg=(np.median(dif)<=-0.10 and np.mean(dif<0)>=0.8);eq=(abs(np.median(dif))<=0.03 and np.mean(np.abs(dif)<=0.06)>=0.8)
        rep['summary']={'median_anchor':float(np.median(anc)),'anchor_positive_fraction':float(np.mean(anc>0)),'median_slow_minus_fast':float(np.median(dif)),'slow_positive_fraction':float(np.mean(dif>0)),'near_zero_fraction':float(np.mean(np.abs(dif)<=0.06))}
        if not anchor_ok:rep['decision']='KILL_PREREQUISITE_NOT_REPRODUCED'
        elif pos:rep['decision']='PASS_PERSISTENT_HEAD_HELPS'
        elif neg:rep['decision']='PASS_RAPID_SWITCHING_HELPS'
        elif eq:rep['decision']='KILL_NO_TEMPORAL_PERSISTENCE_EFFECT'
        else:rep['decision']='INCONCLUSIVE_FIXED_PROTOCOL_NO_TUNING'
    print(json.dumps(rep,indent=2));(a.root/'decision.json').write_text(json.dumps(rep,indent=2)+'\n')
if __name__=='__main__':main()
