#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, difflib, json, re
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

TOKEN_RE=re.compile(r"\w+|[^\w\s]",flags=re.UNICODE)
def toks(text): return TOKEN_RE.findall(text or "")

def pair_metrics(control,trap):
    a,b=toks(control['narrative']),toks(trap['narrative'])
    sm=difflib.SequenceMatcher(a=a,b=b,autojunk=False)
    changed=blocks=largest=0
    for tag,i1,i2,j1,j2 in sm.get_opcodes():
        if tag=='equal': continue
        blocks+=1; span=max(i2-i1,j2-j1); changed+=span; largest=max(largest,span)
    denom=max(1,max(len(a),len(b)))
    return {'case_id':control['case_id'],'control_gt':control['ground_truth'],'trap_gt':trap['ground_truth'],'gt_flips':control['ground_truth']!=trap['ground_truth'],'same_age':control['age']==trap['age'],'same_sex':control['sex']==trap['sex'],'control_tokens':len(a),'trap_tokens':len(b),'sequence_match_ratio':sm.ratio(),'changed_token_fraction':changed/denom,'change_blocks':blocks,'largest_change_span_fraction':largest/denom}

def q(xs,p):
    if not xs: return float('nan')
    ys=sorted(xs); return ys[min(len(ys)-1,max(0,round((len(ys)-1)*p)))]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',default='zhui711/MedEinst'); ap.add_argument('--split',default='test'); ap.add_argument('--outdir',default='artifacts/g0_pair_locality'); a=ap.parse_args()
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    ds=load_dataset(a.dataset,split=a.split); grouped=defaultdict(list)
    for row in ds: grouped[row['case_id']].append(dict(row))
    metrics=[]; malformed=[]
    for cid,rows in tqdm(grouped.items(),desc='MedEinst pair audit'):
        by=defaultdict(list)
        for r in rows: by[r['case_type']].append(r)
        if len(by['control'])!=1 or len(by['trap'])!=1:
            malformed.append({'case_id':cid,'n_control':len(by['control']),'n_trap':len(by['trap'])}); continue
        metrics.append(pair_metrics(by['control'][0],by['trap'][0]))
    if metrics:
        with (out/'pair_metrics.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(metrics[0])); w.writeheader(); w.writerows(metrics)
    with (out/'most_diffuse_200.jsonl').open('w',encoding='utf-8') as f:
        for r in sorted(metrics,key=lambda x:x['changed_token_fraction'],reverse=True)[:200]: f.write(json.dumps(r)+"\n")
    with (out/'malformed_pairs.jsonl').open('w',encoding='utf-8') as f:
        for r in malformed: f.write(json.dumps(r)+"\n")
    n=len(metrics); changed=[m['changed_token_fraction'] for m in metrics]; largest=[m['largest_change_span_fraction'] for m in metrics]
    flip=sum(m['gt_flips'] for m in metrics)/max(1,n); demo=sum(m['same_age'] and m['same_sex'] for m in metrics)/max(1,n)
    gate={'pair_count_ge_5000':n>=5000,'malformed_pairs_eq_0':not malformed,'ground_truth_flip_rate_ge_0.99':flip>=.99,'age_sex_match_rate_ge_0.99':demo>=.99,'median_changed_fraction_le_0.12':q(changed,.5)<=.12,'p90_changed_fraction_le_0.30':q(changed,.9)<=.30}
    summary={'dataset':a.dataset,'split':a.split,'raw_rows':len(ds),'valid_pairs':n,'malformed_pairs':len(malformed),'ground_truth_flip_rate':flip,'age_sex_match_rate':demo,'changed_token_fraction':{'median':q(changed,.5),'p90':q(changed,.9),'p95':q(changed,.95)},'largest_change_span_fraction':{'median':q(largest,.5),'p90':q(largest,.9)},'gate':gate,'verdict':'PAIR_STRUCTURE_OK' if all(gate.values()) else 'PAIR_STRUCTURE_NOT_CLEAN_ENOUGH'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
