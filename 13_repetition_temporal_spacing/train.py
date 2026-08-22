#!/usr/bin/env python3
"""Train the 34M Qwen3-style model on one frozen Topic-13 schedule."""
from __future__ import annotations
import argparse, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import Qwen3Config, Qwen3ForCausalLM
EXPECTED_PARAMS_34M=34_061_856
class ScheduledBlocks(Dataset):
    def __init__(self,blocks_path,schedule_path):
        self.blocks=np.load(blocks_path,mmap_mode="r");self.schedule=np.load(schedule_path,mmap_mode="r")
        if self.schedule.ndim!=1:raise ValueError("schedule must be 1-D")
        if int(self.schedule.max())>=len(self.blocks):raise ValueError("schedule references corpus block out of range")
    def __len__(self):return len(self.schedule)
    def __getitem__(self,i):return torch.from_numpy(np.array(self.blocks[int(self.schedule[i])],dtype=np.int64,copy=True))
class EvalBlocks(Dataset):
    def __init__(self,path):self.blocks=np.load(path,mmap_mode="r")
    def __len__(self):return len(self.blocks)
    def __getitem__(self,i):return torch.from_numpy(np.array(self.blocks[i],dtype=np.int64,copy=True))
def qwen34m_config(attn):
    c=Qwen3Config(vocab_size=151670,hidden_size=96,intermediate_size=256,num_hidden_layers=3,num_attention_heads=32,num_key_value_heads=32,head_dim=128,max_position_embeddings=32768,rms_norm_eps=1e-6,rope_theta=1_000_000.0,attention_bias=False,tie_word_embeddings=False,use_cache=False);c._attn_implementation=attn;return c
def fingerprint_model(model):
    h=hashlib.sha256()
    with torch.no_grad():
        for name,p in model.named_parameters():h.update(name.encode());flat=p.detach().view(-1);h.update(flat[:min(256,flat.numel())].float().cpu().numpy().tobytes())
    return h.hexdigest()
def lr_at(step,total,peak,warm):
    w=max(1,int(total*warm))
    if step<w:return peak*(step+1)/w
    prog=(step-w)/max(1,total-w);return peak*0.5*(1+math.cos(math.pi*min(1.0,prog)))
def set_seed(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def evaluate(model,loader,device,max_batches=None):
    model.eval();losses=[]
    with torch.inference_mode():
        for bi,ids in enumerate(loader):
            if max_batches is not None and bi>=max_batches:break
            ids=ids.to(device,non_blocking=True)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=(device.type=="cuda")):
                logits=model(input_ids=ids,use_cache=False).logits;lt=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction="none").view(ids.shape[0],-1);losses.extend(lt.mean(1).cpu().tolist())
            del logits
    x=np.asarray(losses,dtype=np.float64);return float(x.mean()),float(x.std(ddof=1)/math.sqrt(len(x))),len(x),x
def main():
    p=argparse.ArgumentParser();p.add_argument("--blocks",type=Path,required=True);p.add_argument("--eval-blocks",type=Path,required=True);p.add_argument("--schedule",type=Path,required=True);p.add_argument("--out-dir",type=Path,required=True);p.add_argument("--condition",required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--experiment-id",required=True);p.add_argument("--hardware-label",default="unknown");p.add_argument("--micro-batch",type=int,default=8);p.add_argument("--grad-accum",type=int,default=1);p.add_argument("--num-workers",type=int,default=2);p.add_argument("--base-lr",type=float,default=1e-6);p.add_argument("--peak-lr",type=float,default=None);p.add_argument("--warmup-ratio",type=float,default=0.2);p.add_argument("--weight-decay",type=float,default=0.01);p.add_argument("--beta1",type=float,default=0.9);p.add_argument("--beta2",type=float,default=0.95);p.add_argument("--clip-grad",type=float,default=1.0);p.add_argument("--attn-implementation",choices=["flash_attention_2","sdpa","eager"],default="flash_attention_2");p.add_argument("--compile",action="store_true");p.add_argument("--monitor-every",type=int,default=500);p.add_argument("--monitor-eval-blocks",type=int,default=128);args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("training requires CUDA")
    device=torch.device("cuda");set_seed(args.seed);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    train_ds=ScheduledBlocks(args.blocks,args.schedule);eval_ds=EvalBlocks(args.eval_blocks);seq_len=int(train_ds.blocks.shape[1])
    if seq_len!=2048:raise RuntimeError(f"expected seq_len=2048, got {seq_len}")
    loader=DataLoader(train_ds,batch_size=args.micro_batch,shuffle=False,num_workers=args.num_workers,pin_memory=True,drop_last=False);eval_loader=DataLoader(eval_ds,batch_size=args.micro_batch,shuffle=False,num_workers=args.num_workers,pin_memory=True)
    model=Qwen3ForCausalLM(qwen34m_config(args.attn_implementation));n_params=sum(p.numel() for p in model.parameters())
    if n_params!=EXPECTED_PARAMS_34M:raise RuntimeError(f"config drift: expected {EXPECTED_PARAMS_34M}, got {n_params}")
    init_fp=fingerprint_model(model);model=model.to(device=device,dtype=torch.bfloat16)
    if args.compile:model=torch.compile(model)
    tokens_per_step=args.micro_batch*seq_len*args.grad_accum;peak=args.peak_lr if args.peak_lr is not None else args.base_lr*math.sqrt(tokens_per_step);opt=torch.optim.AdamW(model.parameters(),lr=peak,betas=(args.beta1,args.beta2),weight_decay=args.weight_decay,fused=True);total_micro=len(loader);total_steps=math.ceil(total_micro/args.grad_accum);args.out_dir.mkdir(parents=True,exist_ok=True);opt.zero_grad(set_to_none=True);t0=time.time();micro_in=opt_step=seen=0
    with (args.out_dir/"train_log.jsonl").open("w") as logf:
        for mi,ids in enumerate(loader):
            ids=ids.to(device,non_blocking=True)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):out=model(input_ids=ids,labels=ids,use_cache=False);loss=out.loss/args.grad_accum
            loss.backward();seen+=ids.shape[0];micro_in+=1;is_last=mi+1==total_micro
            if micro_in==args.grad_accum or is_last:
                if micro_in!=args.grad_accum:
                    scale=args.grad_accum/micro_in
                    for param in model.parameters():
                        if param.grad is not None:param.grad.mul_(scale)
                gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad));lr=lr_at(opt_step,total_steps,peak,args.warmup_ratio)
                for g in opt.param_groups:g["lr"]=lr
                opt.step();opt.zero_grad(set_to_none=True);opt_step+=1;micro_in=0;rec={"optimizer_step":opt_step,"blocks_seen":seen,"tokens_seen":seen*seq_len,"train_loss_last_micro":float(loss.detach().item()*args.grad_accum),"lr":lr,"grad_norm":gn,"elapsed_sec":time.time()-t0}
                if args.monitor_every>0 and (opt_step%args.monitor_every==0 or opt_step==total_steps):
                    mb=math.ceil(args.monitor_eval_blocks/args.micro_batch);ml,se,mn,_=evaluate(model,eval_loader,device,mb);rec.update({"monitor_eval_loss":ml,"monitor_eval_se":se,"monitor_eval_blocks":mn});model.train()
                logf.write(json.dumps(rec,sort_keys=True)+"\n");logf.flush()
            del out,loss
    fl,fse,fn,fb=evaluate(model,eval_loader,device,None);np.save(args.out_dir/"eval_block_losses.npy",fb)
    metrics={"experiment_id":args.experiment_id,"condition":args.condition,"seed":args.seed,"final_eval_loss":fl,"final_eval_se_blocks":fse,"eval_blocks":fn,"train_blocks":len(train_ds),"train_tokens":len(train_ds)*seq_len,"optimizer_steps":opt_step,"micro_batch":args.micro_batch,"grad_accum":args.grad_accum,"blocks_per_optimizer_step":args.micro_batch*args.grad_accum,"tokens_per_optimizer_step_nominal":tokens_per_step,"base_lr":args.base_lr,"peak_lr":peak,"warmup_ratio":args.warmup_ratio,"model_params":n_params,"init_fingerprint":init_fp,"schedule_sha256":hashlib.sha256(np.load(args.schedule).tobytes()).hexdigest(),"elapsed_sec":time.time()-t0,"torch_version":torch.__version__,"hardware_label":args.hardware_label,"cuda_device_name":torch.cuda.get_device_name(0)};(args.out_dir/"metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True)+"\n");print(json.dumps(metrics,indent=2,sort_keys=True))
if __name__=="__main__":main()
