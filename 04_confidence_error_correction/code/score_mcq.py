#!/usr/bin/env python3
"""Permutation-robust MCQ scoring for Topic 04.

Input JSONL fields: id, question, choices (list[str]), answer (int), optional dataset.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Iterable
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]

def write_jsonl(path, rows: Iterable[dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")

def cyclic_permutations(k):
    base=list(range(k)); return [base[s:]+base[:s] for s in range(k)]

def format_prompt(question, choices):
    return "\n".join([question.strip(),"","Options:",*[f"{LABELS[i]}. {c}" for i,c in enumerate(choices)],"","Answer with only the letter of the best option.","Answer:"])

@torch.inference_mode()
def continuation_logprob(model, tokenizer, prompt, continuation):
    p=tokenizer(prompt,return_tensors="pt",add_special_tokens=False).input_ids
    full=tokenizer(prompt+continuation,return_tensors="pt",add_special_tokens=False).input_ids.to(model.device)
    plen=p.shape[1]
    logits=model(full).logits[:,:-1].log_softmax(-1); target=full[:,1:]
    token_lp=logits.gather(-1,target.unsqueeze(-1)).squeeze(-1)
    return float(token_lp[:,max(plen-1,0):].sum().item())

@torch.inference_mode()
def option_probs(model, tokenizer, question, choices):
    prompt=format_prompt(question,choices)
    scores=[continuation_logprob(model,tokenizer,prompt," "+LABELS[i]) for i in range(len(choices))]
    return torch.softmax(torch.tensor(scores,dtype=torch.float64),0).tolist()

def semantic_metrics(probs, answer):
    pc=float(probs[answer]); wrong=[float(p) for i,p in enumerate(probs) if i!=answer]
    mass=max(1-pc,1e-12); q=[p/mass for p in wrong]
    ent=-sum(v*math.log(max(v,1e-12)) for v in q); entn=ent/math.log(len(q)) if len(q)>1 else 0
    top=max((i for i in range(len(probs)) if i!=answer),key=lambda i:probs[i])
    return {"p_correct":pc,"wrong_concentration":max(q),"wrong_entropy_norm":entn,"wrong_concentration_entropy":1-entn,"top_wrong":int(top)}

def score_item(model,tokenizer,item):
    choices=list(item["choices"]); answer=int(item["answer"]); k=len(choices)
    if k<4: raise ValueError("requires >=4 choices")
    mapped=[]; tops=[]
    for perm in cyclic_permutations(k):
        local=option_probs(model,tokenizer,item["question"],[choices[i] for i in perm]); semantic=[0.0]*k
        for li,oi in enumerate(perm): semantic[oi]=local[li]
        mapped.append(semantic); tops.append(max((i for i in range(k) if i!=answer),key=lambda i:semantic[i]))
    avg=[sum(r[i] for r in mapped)/len(mapped) for i in range(k)]; m=semantic_metrics(avg,answer)
    modal=max(set(tops),key=tops.count)
    return {**item,"semantic_probs":avg,**m,"top_wrong_stability":tops.count(modal)/len(tops),"top_wrong_by_perm":tops,"permutation_probs":mapped}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--input",required=True); ap.add_argument("--output",required=True); ap.add_argument("--max-items",type=int)
    a=ap.parse_args(); tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map="auto",trust_remote_code=True).eval()
    items=read_jsonl(a.input); items=items[:a.max_items] if a.max_items else items; out=[]
    for n,it in enumerate(items,1):
        try: out.append(score_item(model,tok,it))
        except Exception as e: out.append({**it,"scoring_error":repr(e)})
        if n%50==0: print(f"scored {n}/{len(items)}")
    write_jsonl(a.output,out)
if __name__=="__main__": main()
