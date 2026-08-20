#!/usr/bin/env python3
"""Summarize old/verified route suffix-NLL curves by group and matched pair."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from common import read_jsonl


def boot(xs: list[float], n: int = 5000, seed: int = 0) -> dict:
    if not xs:
        return {"n": 0, "mean": None, "ci95": [None, None]}
    rng = random.Random(seed)
    b=[]
    for _ in range(n):
        z=[xs[rng.randrange(len(xs))] for _ in xs]
        b.append(sum(z)/len(z))
    b.sort()
    return {"n":len(xs),"mean":sum(xs)/len(xs),"ci95":[b[int(.025*n)],b[min(n-1,int(.975*n))]]}


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True)
    ap.add_argument('--pairs',default=None)
    ap.add_argument('--output-json',default=None)
    ap.add_argument('--bootstrap',type=int,default=5000)
    args=ap.parse_args()
    rows=read_jsonl(args.input)
    pair_of={}
    if args.pairs:
        for p in read_jsonl(args.pairs):
            pair_of[('forgotten',str(p['forgotten_problem_id']))]=p['pair_id']
            pair_of[('never_correct',str(p['never_problem_id']))]=p['pair_id']

    cells=defaultdict(list)
    paired=defaultdict(dict)
    for r in rows:
        g=str(r.get('group'))
        pid=str(r['problem_id'])
        f=float(r['prefix_fraction'])
        v=float(r['suffix_nll'])
        cells[(g,f)].append(v)
        pair=pair_of.get((g,pid))
        if pair:
            paired[(pair,f)][g]=v
    summary={'groups':{},'paired_F_minus_N':{}}
    for (g,f),xs in sorted(cells.items()):
        summary['groups'][f'{g}|{f:.2f}']=boot(xs,args.bootstrap)
    fracs=sorted({f for (_,f) in cells})
    for f in fracs:
        diffs=[]
        for (pair,pf),vals in paired.items():
            if abs(pf-f)<1e-9 and 'forgotten' in vals and 'never_correct' in vals:
                diffs.append(vals['forgotten']-vals['never_correct'])
        summary['paired_F_minus_N'][f'{f:.2f}']=boot(diffs,args.bootstrap)
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    if args.output_json:
        p=Path(args.output_json);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

if __name__=='__main__':main()
