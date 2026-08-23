#!/usr/bin/env python3
from __future__ import annotations

import argparse, ast, json, random, re
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TargetProgram:
    sample_id: int
    code: str
    input_x: int
    output: list[int]
    lexical_line: str
    trace: list[dict]


def build_target_program(rng: random.Random, sample_id: int, n_steps: int) -> TargetProgram:
    arr = [0] * n_steps
    x = rng.randint(10, 99)
    order = list(range(n_steps)); rng.shuffle(order)
    lines = [f"def target_{sample_id}(x):", f"    arr = {arr!r}"]
    prev, prev_ref, trace, assigns = x, "x", [], []
    for step, idx in enumerate(order):
        delta = rng.choice([d for d in range(-99, 100) if d])
        value = prev + delta
        op = "+" if delta > 0 else "-"
        line = f"    arr[{idx}] = {prev_ref} {op} {abs(delta)}"
        lines.append(line); assigns.append(line); arr[idx] = value
        trace.append({"step": step, "target_index": idx, "source": prev_ref, "delta": delta, "value": value})
        prev, prev_ref = value, f"arr[{idx}]"
    lines.append("    return arr")
    return TargetProgram(sample_id, "\n".join(lines), x, arr, assigns[len(assigns)//2], trace)


def build_distractor(rng: random.Random, idx: int, n_lines: int = 12) -> str:
    lines = [f"def helper_{idx}(x):", f"    v = x + {rng.randint(-20,20)}"]
    for j in range(n_lines):
        k = rng.randint(1,97)
        if j % 3 == 0: lines.append(f"    v = (v * {rng.randint(2,9)} + {k}) % 100003")
        elif j % 3 == 1: lines.append(f"    v = v - {k}")
        else: lines.append(f"    v = v + {k}")
    lines.append("    return v")
    return "\n".join(lines)


def token_len(tok, text): return len(tok(text, add_special_tokens=False).input_ids)


def make_context(tok, target: str, rng: random.Random, target_tokens: int, position: str) -> str:
    header = "# Synthetic code repository\n\n"
    budget = max(256, target_tokens - token_len(tok,target) - token_len(tok,header) - 128)
    blocks, used, i = [], 0, 0
    while used < budget:
        b = build_distractor(rng, i); n = token_len(tok, b + "\n\n")
        if used + n > budget and blocks: break
        blocks.append(b); used += n; i += 1
    if position == "start": seq = [target] + blocks
    elif position == "middle":
        c = len(blocks)//2; seq = blocks[:c] + [target] + blocks[c:]
    else: raise ValueError(position)
    return header + "\n\n".join(seq)


def chat(tok, user):
    messages=[{"role":"system","content":"Controlled code experiment. Follow output format exactly; no explanation."},{"role":"user","content":user}]
    if getattr(tok,"chat_template",None):
        try: return tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        except TypeError: return tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    return messages[0]["content"]+"\n\n"+user+"\nAnswer:"


@torch.inference_mode()
def generate(model,tok,prompt,max_new_tokens):
    e=tok(prompt,return_tensors="pt",add_special_tokens=False)
    ids=e["input_ids"].to(model.device); mask=e["attention_mask"].to(model.device)
    out=model.generate(input_ids=ids,attention_mask=mask,do_sample=False,max_new_tokens=max_new_tokens,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
    return tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True).strip()


def normalize_line(text):
    lines=[x.strip(" `\t") for x in text.splitlines() if x.strip()]
    cand=next((x for x in lines if re.search(r"arr\[\d+\]\s*=",x)),text.strip())
    return re.sub(r"\s+"," ",cand.strip().rstrip(";"))


def parse_list(text):
    for m in re.findall(r"\[[^\[\]\n]*\]",text):
        try: v=ast.literal_eval(m)
        except Exception: continue
        if isinstance(v,list) and all(isinstance(x,int) for x in v): return v
    return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="Qwen/Qwen2.5-Coder-7B-Instruct"); p.add_argument("--n",type=int,default=64)
    p.add_argument("--seed",type=int,default=20260823); p.add_argument("--context-tokens",type=int,default=8192)
    p.add_argument("--steps",type=int,default=8); p.add_argument("--outdir",default="artifacts/g0")
    a=p.parse_args(); outdir=Path(a.outdir); outdir.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True); tok.pad_token_id=tok.pad_token_id or tok.eos_token_id
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map="auto",trust_remote_code=True).eval()
    recs=[]; rp=outdir/"records.jsonl"; rp.unlink(missing_ok=True)
    for sid in tqdm(range(a.n),desc="SemTrace G0"):
        prog=build_target_program(random.Random(a.seed+sid),sid,a.steps)
        contexts={p:make_context(tok,prog.code,random.Random(a.seed*1000003+sid),a.context_tokens,p) for p in ("start","middle")}
        row={"sample_id":sid,"input_x":prog.input_x,"expected_output":prog.output,"lexical_line":prog.lexical_line,"trace":prog.trace,"conditions":{}}
        idx=re.search(r"arr\[(\d+)\]",prog.lexical_line).group(1)
        for pos,ctx in contexts.items():
            sq=chat(tok,f"{ctx}\n\nEvaluate target_{sid}({prog.input_x}) exactly. Return ONLY the resulting Python list of integers.")
            lq=chat(tok,f"{ctx}\n\nInside target_{sid}, copy exactly the assignment line whose left-hand side is arr[{idx}]. Return ONLY that one assignment line.")
            st=generate(model,tok,sq,64); lt=generate(model,tok,lq,48)
            row["conditions"][pos]={"context_tokens_actual":token_len(tok,ctx),"semantic_output":st,"semantic_correct":parse_list(st)==prog.output,"lexical_output":lt,"lexical_correct":normalize_line(lt)==normalize_line(prog.lexical_line)}
        c=row["conditions"]; row["eligible"]=c["start"]["semantic_correct"] and c["middle"]["lexical_correct"]
        row["critical_cell"]=row["eligible"] and not c["middle"]["semantic_correct"]
        recs.append(row)
        with rp.open("a") as f: f.write(json.dumps(row)+"\n")
    n=len(recs); eligible=[r for r in recs if r["eligible"]]; critical=[r for r in recs if r["critical_cell"]]
    start=sum(r["conditions"]["start"]["semantic_correct"] for r in recs)/n; midlex=sum(r["conditions"]["middle"]["lexical_correct"] for r in recs)/n; midsem=sum(r["conditions"]["middle"]["semantic_correct"] for r in recs)/n
    rate=len(critical)/max(1,len(eligible)); gate={"start_semantic_acc_ge_0.50":start>=.5,"middle_lexical_acc_ge_0.80":midlex>=.8,"critical_count_ge_16":len(critical)>=16,"critical_rate_ge_0.20":rate>=.2}
    summary={"model":a.model,"n":n,"start_semantic_accuracy":start,"middle_semantic_accuracy":midsem,"middle_lexical_accuracy":midlex,"eligible_count":len(eligible),"critical_count":len(critical),"critical_rate_among_eligible":rate,"gate":gate,"verdict":"GO_MECHANISM" if all(gate.values()) else "STOP_OR_REDESIGN_G0"}
    (outdir/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__ == "__main__": main()
