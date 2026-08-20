#!/usr/bin/env python3
"""Teacher-forced per-token suffix NLL of frozen old/verified solution traces."""
import argparse,json,math
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
def read(path):
    with open(path,encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
def split(trace):
    x=[z.strip() for z in trace.splitlines() if z.strip()];return x if x else [trace.strip()]
def cut(trace,frac):
    s=split(trace)
    if frac<=0:return "","\n".join(s)
    n=max(1,min(math.ceil(frac*len(s)),len(s)-1));return "\n".join(s[:n]),"\n".join(s[n:])
@torch.inference_mode()
def nll(model,tok,prompt,prefix,suffix):
    context=prompt.rstrip()+"\n"+prefix.rstrip()+("\n" if prefix else "");ci=tok(context,return_tensors="pt",add_special_tokens=False).input_ids;full=tok(context+suffix,return_tensors="pt",add_special_tokens=False).input_ids.to(model.device);cl=ci.shape[1]
    lp=model(full).logits[:,:-1].log_softmax(-1);target=full[:,1:];vals=lp.gather(-1,target.unsqueeze(-1)).squeeze(-1)[:,max(cl-1,0):]
    return float(-vals.mean().item()),vals.numel()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",required=True);ap.add_argument("--input",required=True);ap.add_argument("--output",required=True);a=ap.parse_args();tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True);model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map="auto",trust_remote_code=True).eval();out=[]
    for r in read(a.input):
        p,s=cut(r["trace"],float(r["prefix_fraction"]));
        if not s:continue
        loss,n=nll(model,tok,r["prompt"],p,s);out.append({**r,"prefix":p,"suffix_nll":loss,"suffix_tokens":n})
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,"w",encoding="utf-8") as f:
        for r in out:f.write(json.dumps(r,ensure_ascii=False)+"\n")
if __name__=="__main__":main()
