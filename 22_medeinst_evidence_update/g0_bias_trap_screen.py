#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, re
from collections import defaultdict, deque
from pathlib import Path
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def normalize(text):
    text=(text or '').lower().replace('–','-').replace('—','-')
    text=re.sub(r'[^a-z0-9+\-/ ]+',' ',text)
    return re.sub(r'\s+',' ',text).strip()

def resolve_diagnosis(text,labels):
    ans=normalize(text)
    if not ans: return None
    exact=[x for x in labels if normalize(x)==ans]
    if len(exact)==1: return exact[0]
    padded=f' {ans} '; hits=[x for x in labels if normalize(x) and f' {normalize(x)} ' in padded]
    if hits:
        m=max(len(normalize(x)) for x in hits); longest=[x for x in hits if len(normalize(x))==m]
        if len(longest)==1: return longest[0]
    return None

def make_pairs(ds):
    grouped=defaultdict(list)
    for row in ds: grouped[row['case_id']].append(dict(row))
    pairs=[]
    for rows in grouped.values():
        by=defaultdict(list)
        for r in rows: by[r['case_type']].append(r)
        if len(by['control'])==1 and len(by['trap'])==1: pairs.append((by['control'][0],by['trap'][0]))
    return pairs

def stratified_sample(pairs,n,seed):
    rng=random.Random(seed); buckets=defaultdict(list)
    for pair in pairs: buckets[pair[0]['ground_truth']].append(pair)
    for b in buckets.values(): rng.shuffle(b)
    keys=sorted(buckets); rng.shuffle(keys); qs={k:deque(buckets[k]) for k in keys}; out=[]
    while len(out)<min(n,len(pairs)):
        progress=False
        for k in keys:
            if qs[k] and len(out)<n: out.append(qs[k].popleft()); progress=True
        if not progress: break
    return out

def prompt(tok,row):
    user=("Read the clinical case and give the single most likely diagnosis. Return ONLY the diagnosis name, with no reasoning or explanation.\n\n"+f"Age: {row.get('age')}\nSex: {row.get('sex')}\nClinical narrative:\n{row.get('narrative','')}")
    msgs=[{'role':'system','content':'Controlled diagnostic benchmark. Use only the case. Output exactly one diagnosis name.'},{'role':'user','content':user}]
    if getattr(tok,'chat_template',None):
        try:return tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        except TypeError:return tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    return msgs[0]['content']+'\n\n'+user+'\nDiagnosis:'

@torch.inference_mode()
def generate(model,tok,text):
    e=tok(text,return_tensors='pt',add_special_tokens=False); ids=e['input_ids'].to(model.device); mask=e['attention_mask'].to(model.device)
    out=model.generate(input_ids=ids,attention_mask=mask,do_sample=False,max_new_tokens=32,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
    return tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True).strip()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',default='zhui711/MedEinst'); ap.add_argument('--split',default='test'); ap.add_argument('--model',default='Qwen/Qwen3-8B'); ap.add_argument('--n-pairs',type=int,default=512); ap.add_argument('--seed',type=int,default=20260823); ap.add_argument('--outdir',default='artifacts/g0_behavior'); a=ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True); rp=out/'records.jsonl'; rp.unlink(missing_ok=True)
    ds=load_dataset(a.dataset,split=a.split); pairs=stratified_sample(make_pairs(ds),a.n_pairs,a.seed); labels=sorted({r['ground_truth'] for r in ds})
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True); tok.pad_token_id=tok.pad_token_id or tok.eos_token_id
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map='auto',trust_remote_code=True).eval(); recs=[]
    for control,trap in tqdm(pairs,desc='MedEinst G0b'):
        ct=generate(model,tok,prompt(tok,control)); tt=generate(model,tok,prompt(tok,trap)); cp=resolve_diagnosis(ct,labels); tp=resolve_diagnosis(tt,labels); cgt=control['ground_truth']; tgt=trap['ground_truth']
        row={'case_id':control['case_id'],'control_gt':cgt,'trap_gt':tgt,'control_output':ct,'trap_output':tt,'control_pred':cp,'trap_pred':tp,'control_correct':cp==cgt,'trap_correct':tp==tgt,'bias_trap':bool(cp==cgt and tgt!=cgt and tp==cgt),'invalid':cp is None or tp is None}
        recs.append(row)
        with rp.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    n=len(recs); cc=[r for r in recs if r['control_correct']]; bt=[r for r in recs if r['bias_trap']]; ca=len(cc)/max(1,n); ta=sum(r['trap_correct'] for r in recs)/max(1,n); inv=sum(r['invalid'] for r in recs)/max(1,n); btr=len(bt)/max(1,len(cc))
    gate={'control_accuracy_ge_0.35':ca>=.35,'control_correct_count_ge_50':len(cc)>=50,'bias_trap_count_ge_20':len(bt)>=20,'bias_trap_rate_ge_0.20':btr>=.20,'invalid_rate_le_0.10':inv<=.10}
    summary={'model':a.model,'evaluated_pairs':n,'control_accuracy':ca,'trap_accuracy':ta,'control_correct_count':len(cc),'bias_trap_count':len(bt),'bias_trap_rate_among_control_correct':btr,'invalid_rate':inv,'gate':gate,'verdict':'GO_MECHANISM' if all(gate.values()) else 'STOP_OR_REPRODUCE_SEED_MORE_FAITHFULLY'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
