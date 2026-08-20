#!/usr/bin/env python3
"""Locked re-entry analyses, reported overall + discovery + confirmation."""
from __future__ import annotations
import argparse,json,random
from collections import defaultdict
from pathlib import Path
from common import read_jsonl,as_bool

def rates(rows):
    c=defaultdict(list)
    for r in rows:
        key=(str(r['problem_id']),r.get('pair_id'),str(r.get('split','all')),str(r['group']),str(r['source']),float(r['prefix_fraction']))
        c[key].append(as_bool(r['correct']))
    return {k:sum(v)/len(v) for k,v in c.items()}

def boot(xs,n=5000,seed=0):
    if not xs:return {'n':0,'mean':None,'ci95':[None,None]}
    rng=random.Random(seed);bs=[]
    for _ in range(n):
        z=[xs[rng.randrange(len(xs))] for _ in xs];bs.append(sum(z)/len(z))
    bs.sort();return {'n':len(xs),'mean':sum(xs)/len(xs),'ci95':[bs[int(.025*n)],bs[min(n-1,int(.975*n))]]}

def subset_cell(cell,split):
    if split=='all':return cell
    return {k:v for k,v in cell.items() if k[2]==split}

def within(cell,frac,a,b):
    av={};bv={}
    for (pid,pair,sp,g,src,f),rate in cell.items():
        if g!='forgotten' or abs(f-frac)>1e-9:continue
        if src==a:av[pid]=rate
        elif src==b:bv[pid]=rate
    return [av[p]-bv[p] for p in sorted(set(av)&set(bv))]

def base_rescue(cell,frac,src):
    base={};t={}
    for (pid,pair,sp,g,s,f),rate in cell.items():
        if g!='forgotten':continue
        if s=='baseline' and abs(f)<1e-9:base[pid]=rate
        elif s==src and abs(f-frac)<1e-9:t[pid]=rate
    return [t[p]-base[p] for p in sorted(set(base)&set(t))]

def fn(cell,frac):
    F={};N={}
    for (pid,pair,sp,g,s,f),rate in cell.items():
        if pair is None or abs(f-frac)>1e-9:continue
        if g=='forgotten' and s=='oldself':F[str(pair)]=rate
        elif g=='never_correct' and s=='verified_correct':N[str(pair)]=rate
    return [F[p]-N[p] for p in sorted(set(F)&set(N))]

def summarize(cell,nboot):
    raw=defaultdict(list)
    for (pid,pair,sp,g,s,f),v in cell.items():raw[(g,s,f)].append(v)
    out={'raw_cells':{f'{g}|{s}|{f:.2f}':{'n':len(xs),'mean':sum(xs)/len(xs)} for (g,s,f),xs in sorted(raw.items())},'primary':{}}
    fracs=sorted({f for (_,_,_,_,_,f) in cell if f>0})
    for f in fracs:
        out['primary'][f'{f:.2f}']={
            'oldself_minus_other_correct':boot(within(cell,f,'oldself','other_correct'),nboot),
            'oldself_minus_final_wrong':boot(within(cell,f,'oldself','final_wrong'),nboot),
            'oldself_rescue_over_baseline':boot(base_rescue(cell,f,'oldself'),nboot),
            'other_correct_rescue_over_baseline':boot(base_rescue(cell,f,'other_correct'),nboot),
            'forgotten_oldself_minus_matched_never_correct':boot(fn(cell,f),nboot),
        }
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output-json',default=None);ap.add_argument('--bootstrap',type=int,default=5000);a=ap.parse_args()
    cell=rates(read_jsonl(a.input));summary={}
    available={k[2] for k in cell}
    for sp in ['all','discovery','confirmation']:
        if sp=='all' or sp in available:summary[sp]=summarize(subset_cell(cell,sp),a.bootstrap)
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    if a.output_json:
        p=Path(a.output_json);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
if __name__=='__main__':main()
