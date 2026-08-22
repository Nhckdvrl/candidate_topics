#!/usr/bin/env python3
"""Train the paper-matched 34M Qwen3-style model on one frozen schedule."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import Qwen3Config, Qwen3ForCausalLM

EXPECTED_PARAMS_34M = 34_061_856


class ScheduledBlocks(Dataset):
    def __init__(self, blocks_path: Path, schedule_path: Path):
        self.blocks = np.load(blocks_path, mmap_mode="r")
        self.schedule = np.load(schedule_path, mmap_mode="r")
        if self.schedule.ndim != 1:
            raise ValueError("schedule must be 1-D")
        if int(self.schedule.max()) >= len(self.blocks):
            raise ValueError("schedule references corpus block out of range")
    def __len__(self): return len(self.schedule)
    def __getitem__(self, i): return torch.from_numpy(np.array(self.blocks[int(self.schedule[i])], dtype=np.int64, copy=True))


class EvalBlocks(Dataset):
    def __init__(self, path: Path): self.blocks = np.load(path, mmap_mode="r")
    def __len__(self): return len(self.blocks)
    def __getitem__(self, i): return torch.from_numpy(np.array(self.blocks[i], dtype=np.int64, copy=True))


def qwen34m_config(attn_impl: str) -> Qwen3Config:
    cfg = Qwen3Config(vocab_size=151670,hidden_size=96,intermediate_size=256,num_hidden_layers=3,num_attention_heads=32,num_key_value_heads=32,head_dim=128,max_position_embeddings=32768,rms_norm_eps=1e-6,rope_theta=1_000_000.0,attention_bias=False,tie_word_embeddings=False,use_cache=False)
    cfg._attn_implementation = attn_impl
    return cfg


def fingerprint_model(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    with torch.no_grad():
        for name, p in model.named_parameters():
            h.update(name.encode())
            flat = p.detach().view(-1)
            h.update(flat[: min(256, flat.numel())].float().cpu().numpy().tobytes())
    return h.hexdigest()


def lr_at(step: int, total_steps: int, peak_lr: float, warmup_ratio: float) -> float:
    warmup = max(1, int(total_steps * warmup_ratio))
    if step < warmup: return peak_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, max_batches: int | None = None) -> tuple[float, float, int, np.ndarray]:
    model.eval(); block_losses = []
    with torch.inference_mode():
        for bi, ids in enumerate(loader):
            if max_batches is not None and bi >= max_batches: break
            ids = ids.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = model(input_ids=ids, use_cache=False).logits
                loss_tok = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="none").view(ids.shape[0], -1)
                block_losses.extend(loss_tok.mean(dim=1).cpu().tolist())
            del logits
    x = np.asarray(block_losses, dtype=np.float64)
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x))), int(len(x)), x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--blocks", type=Path, required=True); p.add_argument("--eval-blocks", type=Path, required=True); p.add_argument("--schedule", type=Path, required=True); p.add_argument("--out-dir", type=Path, required=True); p.add_argument("--condition", required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--micro-batch", type=int, default=4); p.add_argument("--grad-accum", type=int, default=16); p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--base-lr", type=float, default=1e-6); p.add_argument("--peak-lr", type=float, default=None); p.add_argument("--warmup-ratio", type=float, default=0.2); p.add_argument("--weight-decay", type=float, default=0.01); p.add_argument("--beta1", type=float, default=0.9); p.add_argument("--beta2", type=float, default=0.95); p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--attn-implementation", choices=["flash_attention_2", "sdpa", "eager"], default="flash_attention_2"); p.add_argument("--compile", action="store_true"); p.add_argument("--monitor-every", type=int, default=100); p.add_argument("--monitor-eval-blocks", type=int, default=128)
    args = p.parse_args()

    if not torch.cuda.is_available(): raise RuntimeError("G-0 training requires CUDA")
    device = torch.device("cuda"); set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    train_ds = ScheduledBlocks(args.blocks, args.schedule); eval_ds = EvalBlocks(args.eval_blocks)
    seq_len = int(train_ds.blocks.shape[1])
    if seq_len != 2048: raise RuntimeError(f"expected seq_len=2048, got {seq_len}")
    loader = DataLoader(train_ds,batch_size=args.micro_batch,shuffle=False,num_workers=args.num_workers,pin_memory=True,drop_last=False)
    eval_loader = DataLoader(eval_ds,batch_size=max(1,args.micro_batch),shuffle=False,num_workers=args.num_workers,pin_memory=True)

    model = Qwen3ForCausalLM(qwen34m_config(args.attn_implementation))
    n_params = sum(p.numel() for p in model.parameters())
    if n_params != EXPECTED_PARAMS_34M: raise RuntimeError(f"Qwen3 config drift: expected {EXPECTED_PARAMS_34M:,} params, got {n_params:,}")
    init_fp = fingerprint_model(model)
    model = model.to(device=device, dtype=torch.bfloat16)
    if args.compile: model = torch.compile(model)

    tokens_per_step = args.micro_batch * seq_len * args.grad_accum
    peak_lr = args.peak_lr if args.peak_lr is not None else args.base_lr * math.sqrt(tokens_per_step)
    optimizer = torch.optim.AdamW(model.parameters(),lr=peak_lr,betas=(args.beta1,args.beta2),weight_decay=args.weight_decay,fused=True)
    total_micro = len(loader); total_steps = math.ceil(total_micro / args.grad_accum)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True); t0=time.time(); micro_in_step=0; opt_step=0; seen_blocks=0

    with (args.out_dir / "train_log.jsonl").open("w") as logf:
        for micro_idx, ids in enumerate(loader):
            ids=ids.to(device,non_blocking=True)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                out=model(input_ids=ids,labels=ids,use_cache=False); loss=out.loss/args.grad_accum
            loss.backward(); seen_blocks += ids.shape[0]; micro_in_step += 1
            is_last = micro_idx + 1 == total_micro
            if micro_in_step == args.grad_accum or is_last:
                if micro_in_step != args.grad_accum:
                    scale=args.grad_accum/micro_in_step
                    for param in model.parameters():
                        if param.grad is not None: param.grad.mul_(scale)
                grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad))
                lr=lr_at(opt_step,total_steps,peak_lr,args.warmup_ratio)
                for group in optimizer.param_groups: group["lr"]=lr
                optimizer.step(); optimizer.zero_grad(set_to_none=True); opt_step += 1; micro_in_step=0
                record={"optimizer_step":opt_step,"blocks_seen":seen_blocks,"tokens_seen":seen_blocks*seq_len,"train_loss_last_micro":float(loss.detach().item()*args.grad_accum),"lr":lr,"grad_norm":grad_norm,"elapsed_sec":time.time()-t0}
                if args.monitor_every>0 and (opt_step%args.monitor_every==0 or opt_step==total_steps):
                    monitor_batches=math.ceil(args.monitor_eval_blocks/max(1,args.micro_batch)); mloss,mse,mn,_=evaluate(model,eval_loader,device,monitor_batches); record.update({"monitor_eval_loss":mloss,"monitor_eval_se":mse,"monitor_eval_blocks":mn}); model.train()
                logf.write(json.dumps(record,sort_keys=True)+"\n"); logf.flush()
            del out,loss

    final_loss,final_se,final_n,final_block_losses=evaluate(model,eval_loader,device,None)
    np.save(args.out_dir/"eval_block_losses.npy",final_block_losses.astype(np.float64))
    metrics={"condition":args.condition,"seed":args.seed,"final_eval_loss":final_loss,"final_eval_se_blocks":final_se,"eval_blocks":final_n,"eval_block_losses_sha256":hashlib.sha256(final_block_losses.astype(np.float64).tobytes()).hexdigest(),"train_blocks":len(train_ds),"train_tokens":len(train_ds)*seq_len,"optimizer_steps":opt_step,"micro_batch":args.micro_batch,"grad_accum":args.grad_accum,"tokens_per_optimizer_step_nominal":tokens_per_step,"base_lr":args.base_lr,"peak_lr":peak_lr,"warmup_ratio":args.warmup_ratio,"model_params":n_params,"init_fingerprint":init_fp,"schedule_sha256":hashlib.sha256(np.load(args.schedule).tobytes()).hexdigest(),"elapsed_sec":time.time()-t0,"torch_version":torch.__version__}
    (args.out_dir/"metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True)+"\n")
    print(json.dumps(metrics,indent=2,sort_keys=True))


if __name__ == "__main__": main()
