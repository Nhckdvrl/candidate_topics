#!/usr/bin/env python3
"""Analyze re-entry results with problem-cluster bootstrap."""
import argparse,json,random
from collections import defaultdict
def read(path):
    with open(path,encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
def problem_rates(rows):
    c=defaultdict(list)
    for r in rows:c[(str(r["problem_id"]),str(r["group"]),str(r["source"]),float(r["prefix_fraction"]))].append(bool(r["correct"]))
    return {k:sum(v)/len(v) for k,v in c.items()}
def contrast(pr,frac,a,b):
    x={};y={}
    for (pid,g,s,f),rate in pr.items():
        if g!="forgotten" or abs(f-frac)>1e-9:continue
        if s==a:x[pid]=rate
        elif s==b:y[pid]=rate
    return [x[p]-y[p] for p in sorted(set(x)&set(y))]
def boot(x,n=5000,seed=0):
    if not x:return float("nan"),float("nan"),float("nan")
    r=random.Random(seed);bs=[]
    for _ in range(n):
        z=[x[r.randrange(len(x))] for _ in x];bs.append(sum(z)/len(z))
    bs.sort();return sum(x)/len(x),bs[int(.025*n)],bs[int(.975*n)]
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",required=True);ap.add_argument("--bootstrap",type=int,default=5000);a=ap.parse_args();rows=read(a.input);pr=problem_rates(rows);cells=defaultdict(list)
    for (pid,g,s,f),rate in pr.items():cells[(g,s,f)].append(rate)
    for k in sorted(cells,key=lambda x:(x[0],x[1],x[2])):print(f"{k}: n_problem={len(cells[k])} mean={sum(cells[k])/len(cells[k]):.4f}")
    print("\nPrimary forgotten-item contrasts:")
    for frac in sorted({float(r["prefix_fraction"]) for r in rows}):
        for x,y in [("oldself","other_correct"),("oldself","final_wrong")]:
            d=contrast(pr,frac,x,y);m,lo,hi=boot(d,a.bootstrap);print(f"frac={frac:.2f} {x}-{y}: n={len(d)} delta={m:.4f} 95%CI=[{lo:.4f},{hi:.4f}]")
if __name__=="__main__":main()
