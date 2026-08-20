#!/usr/bin/env python3
"""Fixed correction-curve summaries with matched-pair bootstrap."""
import argparse,json,random
from collections import defaultdict
def read(path):
    with open(path,encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
def summaries(rows):
    b=defaultdict(list)
    for r in rows:b[(r["seed"],r["pair_id"],r["id"],r["group"])].append(r)
    out={}
    for k,rs in b.items():
        rs=sorted(rs,key=lambda r:float(r["exposure"]));ps=[float(r["p_correct"]) for r in rs];old=[float(r["p_old_wrong"]) for r in rs];t=None
        for i,p in enumerate(ps):
            if p>=.5 and all(x>=.5 for x in ps[i:min(i+2,len(ps))]):t=float(rs[i]["exposure"]);break
        out[k]={"auc_correct":sum(ps)/len(ps),"suppression":old[0]-old[-1],"t50":t}
    return out
def diffs(s,metric):
    c=defaultdict(dict)
    for (seed,pair,i,g),v in s.items():c[(seed,pair)][g]=v
    out=[]
    for g in c.values():
        if "high" in g and "low" in g and not(metric=="t50" and (g["high"][metric] is None or g["low"][metric] is None)):out.append(float(g["high"][metric])-float(g["low"][metric]))
    return out
def boot(x,n=5000,seed=0):
    if not x:return float("nan"),float("nan"),float("nan")
    r=random.Random(seed);bs=[]
    for _ in range(n):
        z=[x[r.randrange(len(x))] for _ in x];bs.append(sum(z)/len(z))
    bs.sort();return sum(x)/len(x),bs[int(.025*n)],bs[int(.975*n)]
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",required=True);ap.add_argument("--bootstrap",type=int,default=5000);a=ap.parse_args();s=summaries(read(a.input))
    for m in ["auc_correct","suppression","t50"]:
        d=diffs(s,m);mean,lo,hi=boot(d,a.bootstrap);print(f"{m}: n_pairs={len(d)} high-low={mean:.5f} 95%CI=[{lo:.5f},{hi:.5f}]")
if __name__=="__main__":main()
