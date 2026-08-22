#!/usr/bin/env python3
"""Fail-fast environment/model audit before downloading data or launching training."""
from __future__ import annotations
import argparse, json, math
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

EXPECTED=34_061_856

def make_config(attn):
    c=Qwen3Config(vocab_size=151670,hidden_size=96,intermediate_size=256,num_hidden_layers=3,num_attention_heads=32,num_key_value_heads=32,head_dim=128,max_position_embeddings=32768,rms_norm_eps=1e-6,rope_theta=1_000_000.0,attention_bias=False,tie_word_embeddings=False,use_cache=False)
    c._attn_implementation=attn
    return c

def main():
    p=argparse.ArgumentParser(); p.add_argument('--attn-implementation',default='flash_attention_2'); p.add_argument('--micro-batch',type=int,default=4); p.add_argument('--grad-accum',type=int,default=16); p.add_argument('--seq-len',type=int,default=2048); p.add_argument('--base-lr',type=float,default=1e-6); args=p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('FAIL: CUDA is required for Topic 13 G-0 training')
    model=Qwen3ForCausalLM(make_config(args.attn_implementation)); n=sum(x.numel() for x in model.parameters())
    if n!=EXPECTED: raise SystemExit(f'FAIL parameter count: expected {EXPECTED}, got {n}')
    tps=args.micro_batch*args.grad_accum*args.seq_len
    info={'status':'PASS','model_params':n,'expected_model_params':EXPECTED,'cuda_available':torch.cuda.is_available(),'cuda_device_count':torch.cuda.device_count(),'torch':torch.__version__,'tokens_per_optimizer_step':tps,'frozen_peak_lr':args.base_lr*math.sqrt(tps),'attention_implementation':args.attn_implementation}
    info['gpus']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(json.dumps(info,indent=2,sort_keys=True))
if __name__=='__main__': main()
