#!/usr/bin/env python3
"""Frozen G1 causal-access test for Topic 20.

Implements G1-P0..P3 from G1_CAUSAL_ACCESS.md:
- fit the seed-0 rank probe at fixed L_sat=20 (block 19 zero-based);
- evaluate that frozen probe on fresh int-sci seed 20260824;
- deduplicate exact displayed pairs for inferential counts;
- reflect the residual across the rank-probe hyperplane;
- compare against eight equal-norm random directions orthogonal to rank.

No layer/token/strength/model/prompt search is allowed here.
"""
from __future__ import annotations

import argparse, hashlib, json, math, random, re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET = "int_sci_compare"
L_SAT_BLOCK_ZERO_BASED = 19
L_SAT_ONE_BASED = 20
EXPECTED_SEED0_VAL_ACC = 0.990625
EXPECTED_SEED0_TRAIN_SHA256 = "8a995020ecd21dc23f3a3ac97880652c78c85573fa95b53305a1f89004092914"
EXPECTED_SEED0_VAL_SHA256 = "73f0a6703283d186243b4f4db4238712e0e6b523757693553e3b33b202d33d2e"
RANDOM_SEEDS = list(range(20260831, 20260839))
NUMBER_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?(?:\s*[×x*]\s*10\^?-?\d+)?")
SCI_RE = re.compile(r"[×x*]\s*10\^")
FEW_SHOT = [
    ("9.9 × 10^2", "100", 0),
    ("161230", "7.182 × 10^5", 1),
    ("713", "4.78 × 10^2", 0),
    ("1.354 × 10^6", "4906723", 1),
    ("20834", "6.5 × 10^3", 0),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed0-data-root", type=Path, required=True)
    p.add_argument("--fresh-data-root", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--out-dir", type=Path, default=Path("20_numeracy_representation_access/artifacts/g1"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260824)
    p.add_argument("--train-limit", type=int, default=None, help="Smoke only")
    p.add_argument("--val-limit", type=int, default=None, help="Smoke only")
    p.add_argument("--test-limit", type=int, default=None, help="Smoke only")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def parse_value(s):
    return float(eval(str(s).replace("×", "*").replace("x", "*").replace("^", "**").replace(",", "")))


def is_sci(s): return SCI_RE.search(str(s)) is not None


def make_prompt(sample):
    demos = [f"Q: Which is larger, {a} or {b}? A: {(a,b)[ans]}" for a,b,ans in FEW_SHOT]
    return "\n".join(demos) + f"\nQ: Which is larger, {sample['a']} or {sample['b']}? A:"


def label(sample): return int(parse_value(sample["a"]) > parse_value(sample["b"]))
def gold_side(sample): return "a" if label(sample) == 1 else "b"


def is_tie(sample):
    return math.isclose(parse_value(sample["a"]), parse_value(sample["b"]), rel_tol=0.0, abs_tol=1e-12)


def is_hard(sample):
    a,b = parse_value(sample["a"]), parse_value(sample["b"])
    return (not math.isclose(a,b,rel_tol=0.0,abs_tol=1e-12)) and abs(math.log2(a/b)) < 0.1


def load_jsonl(path, limit=None):
    rows=[]
    with Path(path).open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit: break
    return rows


def parse_generated_number(text):
    m=NUMBER_RE.search(str(text))
    if not m: return None
    try: return parse_value(m.group(0))
    except Exception: return None


def same_num(x,y):
    if x is None or y is None: return False
    return math.isclose(float(x),float(y),rel_tol=1e-10,abs_tol=max(1e-6,1e-10*abs(float(y))))


def classify_completion(sample, completion):
    pred=parse_generated_number(completion); a,b=parse_value(sample["a"]),parse_value(sample["b"])
    if pred is None:
        return {"parseable":False,"choice":"invalid","correct":False,"scientific_choice":None,"pred_value":None}
    am,bm=same_num(pred,a),same_num(pred,b)
    side="a" if am and not bm else "b" if bm and not am else "neither_or_ambiguous"
    return {"parseable":True,"choice":side,"correct":side==gold_side(sample),
            "scientific_choice":is_sci(sample[side]) if side in {"a","b"} else None,
            "pred_value":float(pred)}


def input_device(model):
    try: return model.get_input_embeddings().weight.device
    except Exception: return next(model.parameters()).device


def decoder_layers(model):
    candidates=[getattr(getattr(model,"model",None),"layers",None),
                getattr(getattr(getattr(model,"model",None),"model",None),"layers",None),
                getattr(getattr(model,"transformer",None),"h",None)]
    for layers in candidates:
        if layers is not None: return layers
    raise RuntimeError("Could not locate decoder layers")


def extract_layer_hidden(model, tokenizer, prompts, block_idx, batch_size):
    device=input_device(model); chunks=[]; hidden_index=block_idx+1
    for start in tqdm(range(0,len(prompts),batch_size),desc=f"hidden L{block_idx+1}"):
        ps=prompts[start:start+batch_size]
        enc=tokenizer(ps,return_tensors="pt",padding=True,add_special_tokens=True)
        ids=enc["input_ids"].to(device); mask=enc["attention_mask"].to(device)
        with torch.inference_mode():
            out=model(input_ids=ids,attention_mask=mask,output_hidden_states=True,use_cache=False,return_dict=True)
        chunks.append(out.hidden_states[hidden_index][:,-1,:].detach().to("cpu",dtype=torch.float32).numpy())
        del out,ids,mask
    return np.concatenate(chunks,axis=0)


def restore_output(output:Any,new_hidden):
    if torch.is_tensor(output): return new_hidden
    if isinstance(output,tuple): return (new_hidden,)+output[1:]
    if isinstance(output,list): return [new_hidden]+list(output[1:])
    raise TypeError(type(output))


class PrefillIntervention:
    def __init__(self,mode,w,b,random_direction=None):
        self.mode=mode; self.w=np.asarray(w,dtype=np.float32); self.b=float(b)
        self.w_norm2=float(np.dot(self.w,self.w))
        if self.w_norm2<=0: raise ValueError("degenerate w")
        self.random_direction=None if random_direction is None else np.asarray(random_direction,dtype=np.float32)
        self.n_prefill_calls=0; self.n_modified_rows=0

    def hook(self,module,inputs,output):
        hidden=output if torch.is_tensor(output) else output[0]
        if hidden.ndim!=3 or hidden.shape[1]<=1: return output
        h=hidden[:,-1,:].float(); w=torch.as_tensor(self.w,device=h.device,dtype=h.dtype)
        margin=h@w+self.b
        rank_delta=(-2.0*margin/self.w_norm2)[:,None]*w[None,:]
        if self.mode=="rank_reflection": delta=rank_delta
        elif self.mode=="random_equal_norm":
            if self.random_direction is None: raise RuntimeError("missing random direction")
            r=torch.as_tensor(self.random_direction,device=h.device,dtype=h.dtype)
            delta=rank_delta.norm(dim=-1)[:,None]*r[None,:]
        else: raise ValueError(self.mode)
        changed=hidden.clone(); changed[:,-1,:]=(h+delta).to(hidden.dtype)
        self.n_prefill_calls+=1; self.n_modified_rows+=int(hidden.shape[0])
        return restore_output(output,changed)


def run_generation(model,tokenizer,prompts,samples,batch_size,max_new_tokens,intervention=None):
    device=input_device(model); rows=[]; handle=None
    if intervention is not None:
        handle=decoder_layers(model)[L_SAT_BLOCK_ZERO_BASED].register_forward_hook(intervention.hook)
    try:
        for start in tqdm(range(0,len(prompts),batch_size),desc=intervention.mode if intervention else "baseline"):
            ps=prompts[start:start+batch_size]; ss=samples[start:start+batch_size]
            enc=tokenizer(ps,return_tensors="pt",padding=True,add_special_tokens=True)
            ids=enc["input_ids"].to(device); mask=enc["attention_mask"].to(device)
            with torch.inference_mode():
                seq=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=max_new_tokens,
                                   do_sample=False,use_cache=True,pad_token_id=tokenizer.pad_token_id,
                                   eos_token_id=tokenizer.eos_token_id)
            comps=tokenizer.batch_decode(seq[:,ids.shape[1]:],skip_special_tokens=True)
            rows.extend([{**classify_completion(s,c),"completion":c} for s,c in zip(ss,comps)])
            del seq,ids,mask
    finally:
        if handle is not None: handle.remove()
    if intervention is not None:
        expected=math.ceil(len(prompts)/batch_size)
        if intervention.n_prefill_calls!=expected or intervention.n_modified_rows!=len(prompts):
            raise RuntimeError(f"Prefill hook contract failed: calls={intervention.n_prefill_calls}/{expected}, rows={intervention.n_modified_rows}/{len(prompts)}")
    return rows


def orthogonal_random_direction(dim,w,seed):
    rng=np.random.default_rng(seed); r=rng.standard_normal(dim).astype(np.float64); w64=np.asarray(w,dtype=np.float64)
    r-=np.dot(r,w64)/np.dot(w64,w64)*w64; n=np.linalg.norm(r)
    if n<1e-12: raise RuntimeError("random direction collapse")
    r=(r/n).astype(np.float32)
    if abs(float(np.dot(r.astype(np.float64),w64)))>1e-5*np.linalg.norm(w64):
        raise RuntimeError("random direction not sufficiently orthogonal")
    return r


def unique_test_rows(rows):
    kept=[]; seen=set(); ties=dups=0
    for i,row in enumerate(rows):
        if is_tie(row): ties+=1; continue
        key=(str(row["a"]),str(row["b"]))
        if key in seen: dups+=1; continue
        seen.add(key); x=dict(row); x["raw_index"]=i; kept.append(x)
    return kept,{"raw_n":len(rows),"unique_n":len(kept),"excluded_ties":ties,"excluded_exact_displayed_duplicates":dups}


def baseline_summary(samples,probe_pred,generated):
    hard=np.asarray([is_hard(x) for x in samples],dtype=bool); y=np.asarray([label(x) for x in samples])
    probe_ok=probe_pred==y; gen_ok=np.asarray([x["correct"] for x in generated],dtype=bool)
    invalid=np.asarray([not x["parseable"] for x in generated],dtype=bool); critical=probe_ok & ~gen_ok
    hard_errors=[generated[i] for i in range(len(samples)) if hard[i] and not gen_ok[i]]
    operand=[x for x in hard_errors if x["choice"] in {"a","b"}]; sci=sum(bool(x["scientific_choice"]) for x in operand)
    out={"n":len(samples),"n_hard":int(hard.sum()),"probe_accuracy_full":float(probe_ok.mean()),
         "probe_accuracy_hard":float(probe_ok[hard].mean()) if hard.any() else None,
         "generation_accuracy_full":float(gen_ok.mean()),"generation_accuracy_hard":float(gen_ok[hard].mean()) if hard.any() else None,
         "n_hard_critical":int((hard&critical).sum()),"hard_critical_rate":float(critical[hard].mean()) if hard.any() else None,
         "hard_invalid_rate":float(invalid[hard].mean()) if hard.any() else None,"n_hard_generation_errors":len(hard_errors),
         "n_hard_errors_exact_operand":len(operand),"n_hard_errors_choose_scientific_operand":sci,
         "hard_error_scientific_choice_rate":sci/len(operand) if operand else None}
    cond={"n_hard_ge_100":out["n_hard"]>=100,"hard_probe_ge_0p90":(out["probe_accuracy_hard"] or 0)>=.90,
          "n_hard_critical_ge_25":out["n_hard_critical"]>=25,"hard_critical_rate_ge_0p20":(out["hard_critical_rate"] or 0)>=.20,
          "hard_invalid_lt_0p05":(out["hard_invalid_rate"] if out["hard_invalid_rate"] is not None else 1)<.05}
    out["fresh_object_gate"]={"pass":all(cond.values()),"conditions":cond}
    out["notation_followup_eligible"]=out["hard_error_scientific_choice_rate"] is not None and out["hard_error_scientific_choice_rate"]>=.80
    return out,hard,probe_ok,gen_ok


def bootstrap_delta(rank_flip,null_flip_matrix,n_boot,seed):
    diff=rank_flip.astype(float)-null_flip_matrix.mean(axis=1); rng=np.random.default_rng(seed); n=len(diff)
    means=np.empty(n_boot)
    for i in range(n_boot): means[i]=diff[rng.integers(0,n,size=n)].mean()
    return float(diff.mean()),[float(np.quantile(means,.025)),float(np.quantile(means,.975))]


def main():
    args=parse_args(); set_seed(args.seed); args.out_dir.mkdir(parents=True,exist_ok=True)
    smoke=any(x is not None for x in (args.train_limit,args.val_limit,args.test_limit))
    train_path=args.seed0_data_root/DATASET/"train.jsonl"; val_path=args.seed0_data_root/DATASET/"val.jsonl"; fresh_path=args.fresh_data_root/DATASET/"test.jsonl"
    integrity={"seed0_train_sha256":sha256_file(train_path),"seed0_val_sha256":sha256_file(val_path),"fresh_test_sha256":sha256_file(fresh_path)}
    if not smoke:
        if integrity["seed0_train_sha256"]!=EXPECTED_SEED0_TRAIN_SHA256 or integrity["seed0_val_sha256"]!=EXPECTED_SEED0_VAL_SHA256:
            raise RuntimeError(f"Seed-0 data checksum mismatch: {integrity}")

    tok=AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    tok.padding_side="left"
    dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model=AutoModelForCausalLM.from_pretrained(args.model,device_map="auto",torch_dtype=dtype); model.eval()

    train=load_jsonl(train_path,args.train_limit); val=load_jsonl(val_path,args.val_limit)
    xtr=extract_layer_hidden(model,tok,[make_prompt(x) for x in train],L_SAT_BLOCK_ZERO_BASED,args.batch_size)
    xva=extract_layer_hidden(model,tok,[make_prompt(x) for x in val],L_SAT_BLOCK_ZERO_BASED,args.batch_size)
    ytr=np.asarray([label(x) for x in train]); yva=np.asarray([label(x) for x in val])
    probe=LogisticRegression(max_iter=10000,random_state=0).fit(xtr,ytr); val_acc=float(accuracy_score(yva,probe.predict(xva)))
    w=probe.coef_[0].astype(np.float32); b=float(probe.intercept_[0])
    np.savez(args.out_dir/"rank_probe_lsat.npz",w=w,b=b,block_zero_based=L_SAT_BLOCK_ZERO_BASED,layer_one_based=L_SAT_ONE_BASED,seed0_val_accuracy=val_acc)

    fresh_raw=load_jsonl(fresh_path,args.test_limit)
    if not smoke and len(fresh_raw)!=1600: raise RuntimeError(f"Fresh raw test size must be 1600, got {len(fresh_raw)}")
    fresh,audit=unique_test_rows(fresh_raw); prompts=[make_prompt(x) for x in fresh]
    xfr=extract_layer_hidden(model,tok,prompts,L_SAT_BLOCK_ZERO_BASED,args.batch_size); pred=probe.predict(xfr)
    base=run_generation(model,tok,prompts,fresh,args.batch_size,args.max_new_tokens)
    summary,hard,probe_ok,gen_ok=baseline_summary(fresh,pred,base)

    with (args.out_dir/"fresh_baseline_records.jsonl").open("w",encoding="utf-8") as f:
        for i,s in enumerate(fresh):
            f.write(json.dumps({"unique_index":i,"raw_index":s["raw_index"],"a":s["a"],"b":s["b"],"digit":s.get("digit"),"hard":bool(hard[i]),"gold_side":gold_side(s),"probe_side":"a" if int(pred[i])==1 else "b","probe_correct":bool(probe_ok[i]),**base[i],"critical":bool(probe_ok[i] and not gen_ok[i])},ensure_ascii=False)+"\n")
    (args.out_dir/"fresh_data_audit.json").write_text(json.dumps({**integrity,**audit},indent=2)+"\n")
    (args.out_dir/"fresh_baseline_summary.json").write_text(json.dumps({"seed0_probe_val_accuracy":val_acc,**summary},indent=2)+"\n")

    if smoke:
        payload={"verdict":"SMOKE_ONLY_NO_G1_DECISION","integrity":integrity,"seed0_probe_val_accuracy":val_acc,"fresh_data_audit":audit,"fresh_baseline":summary}
        (args.out_dir/"rank_causal_summary.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2)); return
    if abs(val_acc-EXPECTED_SEED0_VAL_ACC)>1e-12: raise RuntimeError(f"L_sat val accuracy drift: {val_acc} vs {EXPECTED_SEED0_VAL_ACC}")
    if not summary["fresh_object_gate"]["pass"]:
        payload={"verdict":"STOP_G1_NONREPLICATION","integrity":integrity,"seed0_probe_val_accuracy":val_acc,"fresh_data_audit":audit,"fresh_baseline":summary}
        (args.out_dir/"rank_causal_summary.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2)); return

    idx=np.flatnonzero(hard & probe_ok & gen_ok); samples=[fresh[i] for i in idx]; ps=[prompts[i] for i in idx]
    if not samples: raise RuntimeError("No G1 intervention population")
    rank_int=PrefillIntervention("rank_reflection",w,b); rank_after=run_generation(model,tok,ps,samples,args.batch_size,args.max_new_tokens,rank_int)
    rank_flip=np.zeros(len(samples),dtype=bool); changed=np.zeros(len(samples),dtype=bool); changed_exact=np.zeros(len(samples),dtype=bool)
    for i,(s,r) in enumerate(zip(samples,rank_after)):
        g=gold_side(s); opp="b" if g=="a" else "a"; rank_flip[i]=r["choice"]==opp; changed[i]=r["choice"]!=g; changed_exact[i]=changed[i] and r["choice"] in {"a","b"}

    null=np.zeros((len(samples),len(RANDOM_SEEDS)),dtype=bool); null_rows=[]
    for j,seed in enumerate(RANDOM_SEEDS):
        rdir=orthogonal_random_direction(len(w),w,seed); intr=PrefillIntervention("random_equal_norm",w,b,rdir)
        after=run_generation(model,tok,ps,samples,args.batch_size,args.max_new_tokens,intr)
        for i,(s,row) in enumerate(zip(samples,after)):
            g=gold_side(s); opp="b" if g=="a" else "a"; fl=row["choice"]==opp; null[i,j]=fl
            null_rows.append({"population_index":i,"source_unique_index":int(idx[i]),"random_seed":seed,"gold_side":g,"opposite_flip":bool(fl),**row})

    delta,ci=bootstrap_delta(rank_flip,null,args.bootstrap,20260840); f_rank=float(rank_flip.mean()); null_rates=[float(null[:,j].mean()) for j in range(null.shape[1])]; f_null=float(null.mean())
    changed_n=int(changed.sum()); exact_changed=float(changed_exact.sum()/changed_n) if changed_n else None
    margins=xfr[idx]@w+b
    probe_flip_fraction=float((np.sign(margins)!=np.sign(-margins)).mean())

    if probe_flip_fraction>=.99 and delta>=.20 and ci[0]>0 and exact_changed is not None and exact_changed>=.80: verdict="RANK_DIRECTION_CAUSAL"
    elif delta<=.05 and ci[1]<=.10: verdict="READABLE_BUT_NOT_CAUSALLY_USED_AT_LSAT"
    else: verdict="INCONCLUSIVE_DO_NOT_TUNE"

    with (args.out_dir/"rank_reflection_records.jsonl").open("w",encoding="utf-8") as f:
        for i,(s,row) in enumerate(zip(samples,rank_after)):
            f.write(json.dumps({"population_index":i,"source_unique_index":int(idx[i]),"raw_index":s["raw_index"],"a":s["a"],"b":s["b"],"digit":s.get("digit"),"gold_side":gold_side(s),"baseline_completion":base[idx[i]]["completion"],"rank_margin_before":float(margins[i]),"rank_margin_after_analytic":float(-margins[i]),"opposite_flip":bool(rank_flip[i]),**row},ensure_ascii=False)+"\n")
    with (args.out_dir/"random_null_records.jsonl").open("w",encoding="utf-8") as f:
        for row in null_rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")

    payload={"verdict":verdict,"fresh_seed":20260824,"model":args.model,"integrity":integrity,
             "L_sat":{"block_zero_based":L_SAT_BLOCK_ZERO_BASED,"layer_one_based":L_SAT_ONE_BASED,"seed0_validation_probe_accuracy":val_acc},
             "fresh_data_audit":audit,"fresh_baseline":summary,"population_n":len(samples),"probe_flip_fraction_analytic":probe_flip_fraction,
             "F_rank":f_rank,"F_null_by_seed":dict(zip(map(str,RANDOM_SEEDS),null_rates)),"F_null_mean":f_null,"DeltaF":delta,"DeltaF_bootstrap_95ci":ci,
             "rank_changed_n":changed_n,"exact_operand_fraction_among_changed":exact_changed,"notation_followup_eligible":summary["notation_followup_eligible"]}
    (args.out_dir/"rank_causal_summary.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2))


if __name__=="__main__": main()
