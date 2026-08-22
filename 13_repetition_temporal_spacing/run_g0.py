#!/usr/bin/env python3
"""End-to-end orchestration: corpus -> frozen schedules -> parallel conditions -> verdict."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

CONDS=("fresh","clustered","random","even")


def run(cmd,env=None):
    print("+"," ".join(map(str,cmd)),flush=True); subprocess.run(list(map(str,cmd)),check=True,env=env)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--mode",choices=["pilot","confirm"],default="pilot"); p.add_argument("--config",type=Path,default=Path(__file__).parent/"configs/g0.json"); p.add_argument("--work-dir",type=Path,default=Path(__file__).parent/"runs/g0"); p.add_argument("--num-gpus",type=int,default=None); p.add_argument("--no-compile",action="store_true"); args=p.parse_args()
    here=Path(__file__).parent; cfg=json.loads(args.config.read_text())
    run([sys.executable,here/"preflight.py","--attn-implementation",cfg["attn_implementation"],"--micro-batch",cfg["micro_batch"],"--grad-accum",cfg["grad_accum"],"--seq-len",cfg["seq_len"],"--base-lr",cfg["base_lr"]])

    parent_visible=os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent_visible: visible_ids=[x.strip() for x in parent_visible.split(",") if x.strip()]
    else:
        import torch
        visible_ids=[str(i) for i in range(torch.cuda.device_count())]
    if args.num_gpus is None: args.num_gpus=int(os.environ.get("NUM_GPUS",min(4,len(visible_ids))))
    if args.num_gpus<1 or args.num_gpus>len(visible_ids): raise SystemExit(f"num_gpus must be 1..{len(visible_ids)}")
    gpu_ids=visible_ids[:args.num_gpus]

    total_tokens=int(cfg["tokens_per_parameter_at_ot1"]*cfg["overtrain_multiplier"]*cfg["model_total_params"]); total_blocks=total_tokens//cfg["seq_len"]
    from schedule import build_schedules
    seeds=cfg["pilot_seeds"] if args.mode=="pilot" else cfg["confirmation_seeds"]
    specs=[build_schedules(total_blocks,cfg["repeat_fraction"],cfg["repeat_count"],int(seed))[0] for seed in seeds]
    train_blocks=max(s.required_corpus_blocks for s in specs)
    corpus_dir=args.work_dir/"corpus"; corpus_dir.mkdir(parents=True,exist_ok=True)
    run([sys.executable,here/"prepare_corpus.py","--out-dir",corpus_dir,"--train-blocks",train_blocks,"--eval-blocks",cfg["eval_blocks"],"--seq-len",cfg["seq_len"],"--tokenizer",cfg["tokenizer"],"--dataset",cfg["dataset"],"--subset",cfg["subset"]])

    for seed in seeds:
        sched_dir=args.work_dir/f"schedules_seed_{seed}"
        run([sys.executable,here/"schedule.py","--out-dir",sched_dir,"--total-blocks",total_blocks,"--repeat-fraction",cfg["repeat_fraction"],"--repeat-count",cfg["repeat_count"],"--seed",seed])
        pending=[]
        for c in CONDS:
            out=args.work_dir/f"seed_{seed}"/c; out.mkdir(parents=True,exist_ok=True)
            if (out/"metrics.json").exists() and (out/"eval_block_losses.npy").exists(): print("reuse completed",out); continue
            cmd=[sys.executable,here/"train.py","--blocks",corpus_dir/"train_blocks.npy","--eval-blocks",corpus_dir/"eval_blocks.npy","--schedule",sched_dir/f"{c}.npy","--out-dir",out,"--condition",c,"--seed",seed,"--micro-batch",cfg["micro_batch"],"--grad-accum",cfg["grad_accum"],"--base-lr",cfg["base_lr"],"--warmup-ratio",cfg["warmup_ratio"],"--weight-decay",cfg["weight_decay"],"--beta1",cfg["beta1"],"--beta2",cfg["beta2"],"--clip-grad",cfg["clip_grad"],"--attn-implementation",cfg["attn_implementation"]]
            if not args.no_compile: cmd.append("--compile")
            pending.append((c,cmd))

        for start in range(0,len(pending),args.num_gpus):
            procs=[]
            for local,(c,cmd) in enumerate(pending[start:start+args.num_gpus]):
                env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=gpu_ids[local]
                print("+ GPU",gpu_ids[local]," ".join(map(str,cmd)),flush=True); procs.append((c,subprocess.Popen(list(map(str,cmd)),env=env)))
            failures=[]
            for c,proc in procs:
                rc=proc.wait()
                if rc!=0: failures.append((c,rc))
            if failures: raise SystemExit(f"training failures: {failures}")

    run([sys.executable,here/"analyze.py","--run-dir",args.work_dir,"--out-json",args.work_dir/f"summary_{args.mode}.json","--out-md",args.work_dir/f"summary_{args.mode}.md","--config",args.config])
    print((args.work_dir/f"summary_{args.mode}.md").read_text())


if __name__=="__main__": main()
