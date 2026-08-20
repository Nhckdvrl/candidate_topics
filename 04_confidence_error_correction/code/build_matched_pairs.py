#!/usr/bin/env python3
"""Construct accessibility-matched high/low wrong-conviction pairs using base scores only."""
import argparse,json
from pathlib import Path

def read(path):
    with open(path,encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
def write(path,rows):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")
def qtile(v,q):
    x=sorted(v); p=q*(len(x)-1); lo=int(p); hi=min(lo+1,len(x)-1); w=p-lo; return x[lo]*(1-w)+x[hi]*w

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",required=True);ap.add_argument("--output",required=True);ap.add_argument("--p-caliper",type=float,default=.03);ap.add_argument("--length-ratio",type=float,default=1.25);ap.add_argument("--min-stability",type=float,default=.75);a=ap.parse_args()
    rows=[r for r in read(a.input) if "scoring_error" not in r and r.get("top_wrong_stability",0)>=a.min_stability]
    rows=[r for r in rows if max(range(len(r["semantic_probs"])),key=lambda i:r["semantic_probs"][i])!=int(r["answer"])]
    vals=[float(r["wrong_concentration"]) for r in rows]; lo,hi=qtile(vals,1/3),qtile(vals,2/3)
    low=[r for r in rows if float(r["wrong_concentration"])<=lo]; high=[r for r in rows if float(r["wrong_concentration"])>=hi]
    used=set();pairs=[]
    for h in sorted(high,key=lambda r:float(r["p_correct"])):
        hl=max(len(str(h["question"]).split()),1); cand=[]
        for j,l in enumerate(low):
            if j in used:continue
            if h.get("dataset") and l.get("dataset") and h["dataset"]!=l["dataset"]:continue
            if abs(float(h["p_correct"])-float(l["p_correct"]))>a.p_caliper:continue
            ll=max(len(str(l["question"]).split()),1)
            if max(hl,ll)/min(hl,ll)>a.length_ratio:continue
            cand.append((abs(float(h["p_correct"])-float(l["p_correct"])),j,l))
        if cand:
            _,j,l=min(cand,key=lambda x:x[0]);used.add(j);pairs.append({"pair_id":f"pair_{len(pairs):05d}","high":h,"low":l,"p_correct_abs_diff":abs(float(h["p_correct"])-float(l["p_correct"]))})
    write(a.output,pairs);md=sum(p["p_correct_abs_diff"] for p in pairs)/len(pairs) if pairs else float("nan")
    print(f"eligible={len(rows)} low={len(low)} high={len(high)} pairs={len(pairs)} mean_abs_p_correct_diff={md:.4f}")
if __name__=="__main__":main()
