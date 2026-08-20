#!/usr/bin/env python3
"""Create answer-leakage-checked re-entry prompts using transparent step boundaries."""
import argparse,json,math,re
from pathlib import Path
PATS=[re.compile(r"\\boxed\s*\{([^}]*)\}",re.I),re.compile(r"(?:final answer|answer is)\s*[:=]?\s*(.+)",re.I)]
def read(path):
    with open(path,encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
def norm(x):return re.sub(r"\s+","",str(x)).lower().strip(".$")
def split(trace):
    x=[z.strip() for z in re.split(r"\n+",trace) if z.strip()];return x if len(x)>=2 else [z.strip() for z in re.split(r"(?<=[.!?])\s+",trace) if z.strip()]
def leaks(prefix,gold):
    if any(p.search(prefix) for p in PATS):return True
    if gold:
        g=norm(gold);return len(g)>=2 and g in norm(prefix)
    return False
def pref(trace,frac):
    if frac<=0:return ""
    s=split(trace)
    if not s:return ""
    n=max(1,math.ceil(frac*len(s)));n=min(n,max(1,len(s)-1));return "\n".join(s[:n])
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--groups",required=True);ap.add_argument("--output",required=True);ap.add_argument("--fractions",default="0,0.10,0.25,0.50");a=ap.parse_args();fr=[float(x) for x in a.fractions.split(",")];out=[]
    for r in read(a.groups):
        if not r.get("prompt"):continue
        sources=[];g=r["group"]
        if g in {"forgotten","stable_correct"} and r.get("old_correct_trace"):sources.append(("oldself",r["old_correct_trace"]))
        if r.get("other_correct_trace"):sources.append(("other_correct",r["other_correct_trace"]))
        if r.get("final_wrong_trace"):sources.append(("final_wrong",r["final_wrong_trace"]))
        if g=="never_correct" and r.get("verified_correct_trace"):sources.append(("verified_correct",r["verified_correct_trace"]))
        for source,trace in sources:
            for f in fr:
                p=pref(trace,f)
                if p and leaks(p,r.get("gold_answer")):continue
                prompt=r["prompt"].rstrip()
                if p:prompt+="\n\nA partial reasoning attempt is provided below. Continue the reasoning and solve the problem.\n"+p+"\n"
                out.append({"problem_id":r["problem_id"],"group":g,"source":source,"prefix_fraction":f,"prefix":p,"prompt":prompt,"gold_answer":r.get("gold_answer")})
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,"w",encoding="utf-8") as f:
        for r in out:f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(f"wrote {len(out)} prompts")
if __name__=="__main__":main()
