#!/usr/bin/env python3
"""Build robust forgotten / never-correct / stable-correct groups."""
import argparse,json
from collections import defaultdict
from pathlib import Path
def read(path):
    with open(path,encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
def write(path,rows):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",required=True);ap.add_argument("--output",required=True);ap.add_argument("--correct-threshold",type=float,default=.75);ap.add_argument("--wrong-threshold",type=float,default=.125);ap.add_argument("--min-samples",type=int,default=8);a=ap.parse_args();rows=read(a.input)
    if not rows:raise SystemExit("empty input")
    final_order=max(int(r["checkpoint_order"]) for r in rows);b=defaultdict(list);meta={}
    for r in rows:
        pid=str(r["problem_id"]);b[(pid,int(r["checkpoint_order"]),str(r["checkpoint"]))].append(bool(r["correct"]));meta.setdefault(pid,{k:r[k] for k in ("problem_id","prompt","gold_answer") if k in r})
    rates=defaultdict(list)
    for (pid,o,c),v in b.items():
        if len(v)>=a.min_samples:rates[pid].append({"checkpoint":c,"checkpoint_order":o,"n":len(v),"pass_rate":sum(v)/len(v)})
    out=[]
    for pid,ck in rates.items():
        ck=sorted(ck,key=lambda x:x["checkpoint_order"])
        if not any(x["checkpoint_order"]==final_order for x in ck):continue
        final=next(x for x in ck if x["checkpoint_order"]==final_order);earlier=[x for x in ck if x["checkpoint_order"]<final_order];good=[x for x in earlier if x["pass_rate"]>=a.correct_threshold];group=None;old=None
        if final["pass_rate"]<=a.wrong_threshold and good:group="forgotten";old=max(good,key=lambda x:x["checkpoint_order"])
        elif final["pass_rate"]<=a.wrong_threshold and all(x["pass_rate"]<=a.wrong_threshold for x in earlier):group="never_correct"
        elif final["pass_rate"]>=a.correct_threshold and good:group="stable_correct";old=max(good,key=lambda x:x["checkpoint_order"])
        if group:out.append({**meta.get(pid,{"problem_id":pid}),"group":group,"final":final,"old_checkpoint":old,"trajectory_rates":ck})
    write(a.output,out);counts=defaultdict(int)
    for r in out:counts[r["group"]]+=1
    print(f"final_checkpoint_order={final_order} groups={dict(counts)}")
if __name__=="__main__":main()
