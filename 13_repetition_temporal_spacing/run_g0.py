#!/usr/bin/env python3
"""End-to-end Topic-13 orchestration with stale-run protection and GPU rotation."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
CONDS=("fresh","clustered","random","even")

def run(cmd,env=None): print("+"," ".join(map(str,cmd)),flush=True);subprocess.run(list(map(str,cmd)),check=True,env=env)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def make_experiment_id(config_path:Path,here:Path):
    payload={"config":json.loads(config_path.read_text()),"code":{f:sha(here/f) for f in ("schedule.py","train.py","prepare_corpus.py","analyze.py","run_g0.py")}}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()[:20],payload

def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["pilot","confirm"],default="pilot");p.add_argument("--config",type=Path,default=Path(__file__).parent/"configs/g0.json");p.add_argument("--work-dir",type=Path,default=Path(__file__).parent/"runs/g0");p.add_argument("--num-gpus",type=int,default=None);p.add_argument("--no-compile",action="store_true");args=p.parse_args()
    here=Path(__file__).parent;cfg=json.loads(args.config.read_text());experiment_id,payload=make_experiment_id(args.config,here);args.work_dir.mkdir(parents=True,exist_ok=True)
    manifest=args.work_dir/"experiment_manifest.json"
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old.get("experiment_id")!=experiment_id: raise RuntimeError(f"work-dir contains stale experiment {old.get('experiment_id')}; current is {experiment_id}. Use a new work-dir or archive/delete the old runs.")
    else: manifest.write_text(json.dumps({"experiment_id":experiment_id,"payload":payload},indent=2,sort_keys=True)+"\n")
    run([sys.executable,here/"preflight.py","--attn-implementation",cfg["attn_implementation"],"--micro-batch",cfg["micro_batch"],"--grad-accum",cfg["grad_accum"],"--seq-len",cfg["seq_len"],"--base-lr",cfg["base_lr"]])
    parent=os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent: visible=[x.strip() for x in parent.split(",") if x.strip()]
    else:
        import torch
        visible=[str(i) for i in range(torch.cuda.device_count())]
    if args.num_gpus is None: args.num_gpus=int(os.environ.get("NUM_GPUS",min(4,len(visible))))
    if args.num_gpus<1 or args.num_gpus>len(visible): raise SystemExit(f"num_gpus must be 1..{len(visible)}")
    gpu_ids=visible[:args.num_gpus]
    total_tokens=int(cfg["tokens_per_parameter_at_ot1"]*cfg["overtrain_multiplier"]*cfg["model_total_params"]);total_blocks=total_tokens//cfg["seq_len"]
    blocks_per_step=int(cfg["micro_batch"])*int(cfg["grad_accum"])
    from schedule import build_schedules
    seeds=[int(x) for x in (cfg["pilot_seeds"] if args.mode=="pilot" else cfg["confirmation_seeds"])]
    specs=[build_schedules(total_blocks,cfg["repeat_fraction"],cfg["repeat_count"],seed,blocks_per_step)[0] for seed in seeds]
    train_blocks=max(s.required_corpus_blocks for s in specs);corpus_dir=args.work_dir/"corpus";corpus_dir.mkdir(parents=True,exist_ok=True)
    run([sys.executable,here/"prepare_corpus.py","--out-dir",corpus_dir,"--train-blocks",train_blocks,"--eval-blocks",cfg["eval_blocks"],"--seq-len",cfg["seq_len"],"--tokenizer",cfg["tokenizer"],"--dataset",cfg["dataset"],"--subset",cfg["subset"],"--schema-version",cfg["corpus_schema_version"]])
    for seed_i,seed in enumerate(seeds):
        sched_dir=args.work_dir/f"schedules_seed_{seed}"
        run([sys.executable,here/"schedule.py","--out-dir",sched_dir,"--total-blocks",total_blocks,"--repeat-fraction",cfg["repeat_fraction"],"--repeat-count",cfg["repeat_count"],"--seed",seed,"--blocks-per-optimizer-step",blocks_per_step])
        audit=json.loads((sched_dir/"audit.json").read_text())
        for c in ("clustered","random","even"):
            if audit["conditions"][c]["max_repeat_slots_same_optimizer_step"]!=1: raise RuntimeError("schedule violated one-repeat-slot-per-optimizer-step invariant")
        pending=[]
        ordered=list(CONDS[seed_i%len(CONDS):]+CONDS[:seed_i%len(CONDS)])
        gpu_for={c:gpu_ids[j % len(gpu_ids)] for j,c in enumerate(ordered)}
        for c in ordered:
            out=args.work_dir/f"seed_{seed}"/c;out.mkdir(parents=True,exist_ok=True);mp=out/"metrics.json";lp=out/"eval_block_losses.npy"
            if mp.exists() and lp.exists():
                old=json.loads(mp.read_text())
                if old.get("experiment_id")!=experiment_id: raise RuntimeError(f"stale result in {out}; expected experiment {experiment_id}")
                print("reuse completed",out);continue
            cmd=[sys.executable,here/"train.py","--blocks",corpus_dir/"train_blocks.npy","--eval-blocks",corpus_dir/"eval_blocks.npy","--schedule",sched_dir/f"{c}.npy","--out-dir",out,"--condition",c,"--seed",seed,"--experiment-id",experiment_id,"--hardware-label",gpu_for[c],"--micro-batch",cfg["micro_batch"],"--grad-accum",cfg["grad_accum"],"--base-lr",cfg["base_lr"],"--warmup-ratio",cfg["warmup_ratio"],"--weight-decay",cfg["weight_decay"],"--beta1",cfg["beta1"],"--beta2",cfg["beta2"],"--clip-grad",cfg["clip_grad"],"--attn-implementation",cfg["attn_implementation"]]
            if not args.no_compile:cmd.append("--compile")
            pending.append((c,cmd,gpu_for[c]))
        while pending:
            used=set();group=[];rest=[]
            for item in pending:
                c,cmd,gpu=item
                if gpu not in used and len(group)<args.num_gpus: group.append(item);used.add(gpu)
                else: rest.append(item)
            pending=rest;procs=[]
            for c,cmd,gpu in group:
                env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=gpu;print("+ GPU",gpu,c," ".join(map(str,cmd)),flush=True);procs.append((c,subprocess.Popen(list(map(str,cmd)),env=env)))
            failures=[]
            for c,proc in procs:
                rc=proc.wait()
                if rc!=0:failures.append((c,rc))
            if failures:raise SystemExit(f"training failures: {failures}")
    outj=args.work_dir/f"summary_{args.mode}.json";outm=args.work_dir/f"summary_{args.mode}.md"
    run([sys.executable,here/"analyze.py","--run-dir",args.work_dir,"--out-json",outj,"--out-md",outm,"--config",args.config,"--mode",args.mode,"--experiment-id",experiment_id]);print(outm.read_text())
if __name__=="__main__":main()
