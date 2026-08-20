#!/usr/bin/env python3
"""Expand matched items into balanced corrective-SFT exposures."""
import argparse,json,random
from pathlib import Path
LABELS="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def read(path):
    with open(path,encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
def perms(k):
    b=list(range(k));return [b[s:]+b[:s] for s in range(k)]
def make(item,pair,group,perm,v):
    ch=item["choices"];ans=int(item["answer"]);pc=[ch[i] for i in perm];la=perm.index(ans)
    prompt="\n".join([item["question"].strip(),"","Options:",*[f"{LABELS[i]}. {c}" for i,c in enumerate(pc)],"","Give the correct answer. Respond with the answer letter and answer text."])
    return {"id":item["id"],"pair_id":pair,"group":group,"exposure_variant":v,"prompt":prompt,"response":f"{LABELS[la]}. {ch[ans]}","base_p_correct":item["p_correct"],"wrong_concentration":item["wrong_concentration"],"old_wrong_index":item["top_wrong"]}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",required=True);ap.add_argument("--output",required=True);ap.add_argument("--shuffle-seed",type=int,default=0);a=ap.parse_args();out=[]
    for p in read(a.input):
        for g in ("high","low"):
            for v,perm in enumerate(perms(len(p[g]["choices"]))):out.append(make(p[g],p["pair_id"],g,perm,v))
    random.Random(a.shuffle_seed).shuffle(out);Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,"w",encoding="utf-8") as f:
        for r in out:f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(f"wrote {len(out)} corrective examples")
if __name__=="__main__":main()
