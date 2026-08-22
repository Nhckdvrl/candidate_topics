#!/usr/bin/env python3
"""Prepare the fixed-length exact-document corpus used by Topic-13 G-0.

This causal corpus intentionally uses long single documents so every schedule slot contains exactly 2048 real tokens (2047 document tokens + EOS). It is narrower than the seed paper's full document-length distribution; failure to reproduce seed damage is therefore a setup failure, never evidence against spacing.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

def stable_doc_id(row: dict, text: str) -> str:
    for key in ("id","url","source_id","document_id"):
        v=row.get(key)
        if v is not None and str(v): return f"{key}:{v}"
    return "textsha256:"+hashlib.sha256(text.encode("utf-8",errors="ignore")).hexdigest()
def split_bucket(doc_id: str, seed: int=0, modulo: int=1000) -> int:
    b=hashlib.sha256(f"{seed}:{doc_id}".encode()).digest()[:8]; return int.from_bytes(b,"big")%modulo
def id64(doc_id: str) -> np.uint64:return np.uint64(int.from_bytes(hashlib.sha256(doc_id.encode()).digest()[:8],"big"))
def main():
    p=argparse.ArgumentParser();p.add_argument("--out-dir",type=Path,required=True);p.add_argument("--train-blocks",type=int,required=True);p.add_argument("--eval-blocks",type=int,default=2048);p.add_argument("--seq-len",type=int,default=2048);p.add_argument("--tokenizer",default="Qwen/Qwen3-0.6B-Base");p.add_argument("--dataset",default="HuggingFaceTB/smollm-corpus");p.add_argument("--subset",default="fineweb-edu-dedup");p.add_argument("--split-seed",type=int,default=0);p.add_argument("--eval-permille",type=int,default=25);p.add_argument("--shuffle-buffer",type=int,default=10000);p.add_argument("--stream-seed",type=int,default=0);p.add_argument("--tokenize-batch",type=int,default=64);p.add_argument("--schema-version",type=int,default=2);args=p.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    meta_path=args.out_dir/"corpus_meta.json";train_path=args.out_dir/"train_blocks.npy";eval_path=args.out_dir/"eval_blocks.npy";ids_path=args.out_dir/"train_doc_ids.npy";eval_ids_path=args.out_dir/"eval_doc_ids.npy"
    if all(x.exists() for x in (meta_path,train_path,eval_path,ids_path,eval_ids_path)):
        old=json.loads(meta_path.read_text());keys_ok=(old.get("schema_version")==args.schema_version and old.get("dataset")==args.dataset and old.get("subset")==args.subset and old.get("tokenizer")==args.tokenizer and old.get("seq_len")==args.seq_len and old.get("split_seed")==args.split_seed and old.get("stream_seed")==args.stream_seed)
        if keys_ok and old.get("train_blocks",0)>=args.train_blocks and old.get("eval_blocks",0)>=args.eval_blocks: print("reusing prepared corpus:",meta_path);return
        raise RuntimeError("Existing corpus cache is incompatible with this experiment schema. Use a new work-dir or delete the stale corpus directory.")
    tok=AutoTokenizer.from_pretrained(args.tokenizer,use_fast=True)
    if tok.eos_token_id is None:raise RuntimeError("Qwen tokenizer has no EOS token")
    if len(tok)!=151670:raise RuntimeError(f"Expected Qwen3 vocab 151670, got {len(tok)}")
    train_mm=np.lib.format.open_memmap(train_path,mode="w+",dtype=np.uint32,shape=(args.train_blocks,args.seq_len));eval_mm=np.lib.format.open_memmap(eval_path,mode="w+",dtype=np.uint32,shape=(args.eval_blocks,args.seq_len));train_ids=np.lib.format.open_memmap(ids_path,mode="w+",dtype=np.uint64,shape=(args.train_blocks,));eval_ids=np.lib.format.open_memmap(eval_ids_path,mode="w+",dtype=np.uint64,shape=(args.eval_blocks,))
    ds=load_dataset(args.dataset,args.subset,split="train",streaming=True).shuffle(seed=args.stream_seed,buffer_size=args.shuffle_buffer);n_train=n_eval=scanned=long_enough=0;batch=[]
    def consume(rows):
        nonlocal n_train,n_eval,long_enough
        if not rows:return
        enc=tok([r[1] for r in rows],add_special_tokens=False,truncation=True,max_length=args.seq_len-1)["input_ids"]
        for (doc_id,_),ids in zip(rows,enc):
            if len(ids)<args.seq_len-1:continue
            long_enough+=1;block=np.asarray(ids+[tok.eos_token_id],dtype=np.uint32);is_eval=split_bucket(doc_id,args.split_seed)<args.eval_permille
            if is_eval and n_eval<args.eval_blocks:eval_mm[n_eval]=block;eval_ids[n_eval]=id64(doc_id);n_eval+=1
            elif (not is_eval) and n_train<args.train_blocks:train_mm[n_train]=block;train_ids[n_train]=id64(doc_id);n_train+=1
    for row in ds:
        if n_train>=args.train_blocks and n_eval>=args.eval_blocks:break
        scanned+=1;text=row.get("text","");batch.append((stable_doc_id(row,text),text))
        if len(batch)>=args.tokenize_batch:consume(batch);batch=[]
        if scanned%10000==0:print(f"scanned={scanned:,} long={long_enough:,} train={n_train:,}/{args.train_blocks:,} eval={n_eval:,}/{args.eval_blocks:,}",flush=True)
    if (n_train<args.train_blocks or n_eval<args.eval_blocks) and batch:consume(batch)
    train_mm.flush();eval_mm.flush();train_ids.flush();eval_ids.flush()
    if n_train!=args.train_blocks or n_eval!=args.eval_blocks:raise RuntimeError(f"dataset exhausted early: train={n_train}, eval={n_eval}")
    if len(np.unique(np.asarray(train_ids)))!=args.train_blocks:raise RuntimeError("duplicate document IDs entered train corpus")
    if len(np.unique(np.asarray(eval_ids)))!=args.eval_blocks:raise RuntimeError("duplicate document IDs entered eval corpus")
    if np.intersect1d(np.asarray(train_ids),np.asarray(eval_ids)).size:raise RuntimeError("train/eval document collision")
    meta={"schema_version":args.schema_version,"dataset":args.dataset,"subset":args.subset,"tokenizer":args.tokenizer,"vocab_size":len(tok),"seq_len":args.seq_len,"train_blocks":args.train_blocks,"eval_blocks":args.eval_blocks,"train_tokens":args.train_blocks*args.seq_len,"eval_tokens":args.eval_blocks*args.seq_len,"split_seed":args.split_seed,"eval_permille":args.eval_permille,"stream_seed":args.stream_seed,"shuffle_buffer":args.shuffle_buffer,"scanned_documents":scanned,"long_enough_documents":long_enough,"causal_corpus_note":"single long documents only; first seq_len-1 tokens + EOS; chosen to make every temporal slot exactly equal length"};meta_path.write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n");print(json.dumps(meta,indent=2,sort_keys=True))
if __name__=="__main__":main()
