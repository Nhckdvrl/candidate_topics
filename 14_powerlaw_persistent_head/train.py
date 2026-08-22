#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import torch
import torch.nn.functional as F
from core import PROFILES,StateTrackingTransformer,all_permutations,branch_digest,fixed_eval,key_schedule,make_power_batch,make_uniform_batch,model_digest,profile_dict,seed_all

EVAL_SEED=424242

def parse_args():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['warmup','arm'],required=True);p.add_argument('--condition',choices=['uniform','static','slow','fast','persistence']);p.add_argument('--profile',choices=sorted(PROFILES),default='pilot');p.add_argument('--seed',type=int,default=0);p.add_argument('--output',type=Path,default=Path('outputs'));p.add_argument('--batch-size',type=int,default=256);p.add_argument('--eval-batch-size',type=int,default=2048);p.add_argument('--alpha',type=float,default=1.5);p.add_argument('--mapping-seed',type=int,default=1729);p.add_argument('--stream-seed',type=int,default=31415);p.add_argument('--d-model',type=int,default=256);p.add_argument('--layers',type=int,default=4);p.add_argument('--heads',type=int,default=8);p.add_argument('--ff-mult',type=int,default=4);p.add_argument('--peak-lr',type=float,default=2e-4);p.add_argument('--weight-decay',type=float,default=1e-6);p.add_argument('--precision',choices=['fp32','bf16','fp16'],default='bf16');p.add_argument('--device',default='cuda');p.add_argument('--persistence-h',type=int);p.add_argument('--resume',action='store_true');p.add_argument('--save-every',type=int,default=20000);return p.parse_args()

def device_for(x):
    if x.startswith('cuda') and not torch.cuda.is_available(): return torch.device('cpu')
    return torch.device(x)
def model_for(a,d): return StateTrackingTransformer(a.d_model,a.layers,a.heads,a.ff_mult).to(d)
def opt_for(m,a): return torch.optim.AdamW(m.parameters(),lr=a.peak_lr,betas=(0.9,0.999),eps=1e-8,weight_decay=a.weight_decay)
def ac(d,p):
    enabled=d.type=='cuda' and p in {'bf16','fp16'}; dtype=torch.bfloat16 if p=='bf16' else torch.float16
    return torch.autocast(device_type=d.type,dtype=dtype,enabled=enabled)
@torch.inference_mode()
def evaluate(m,x0,y0,d,bs,prec):
    m.eval();ex=tok=ne=nt=0;ls=0.0
    for i in range(0,len(x0),bs):
        x=x0[i:i+bs].to(d);y=y0[i:i+bs].to(d)
        with ac(d,prec): z=m(x);loss=F.cross_entropy(z.reshape(-1,5),y.reshape(-1),reduction='sum')
        pr=z.argmax(-1);ls+=float(loss);ex+=int((pr==y).all(1).sum());tok+=int((pr==y).sum());ne+=y.shape[0];nt+=y.numel()
    m.train();return {'exact_accuracy':ex/ne,'token_accuracy':tok/nt,'eval_loss':ls/nt}
def save(path,m,o,step,extra): path.parent.mkdir(parents=True,exist_ok=True);torch.save({'model':m.state_dict(),'optimizer':o.state_dict(),'step':step,'extra':extra},path)
def load(path,m,o,d):
    ck=torch.load(path,map_location=d,weights_only=False);m.load_state_dict(ck['model']);o.load_state_dict(ck['optimizer']);return ck

def warmup(a):
    pr=PROFILES[a.profile];root=a.output/a.profile/f'seed{a.seed}';root.mkdir(parents=True,exist_ok=True);out=root/'branch.pt'
    if out.exists() and a.resume: print(f'warmup exists: {out}');return
    d=device_for(a.device);seed_all(a.seed);m=model_for(a,d);o=opt_for(m,a);perm=all_permutations()
    for s in range(pr.warmup_steps):
        lr=a.peak_lr*(s+1)/max(1,pr.warmup_steps)
        for g in o.param_groups:g['lr']=lr
        xn,yn=make_uniform_batch(a.seed,s,a.batch_size,a.stream_seed+99_000_000,perm);x=torch.from_numpy(xn).long().to(d);y=torch.from_numpy(yn).long().to(d);o.zero_grad(set_to_none=True)
        with ac(d,a.precision): z=m(x);loss=F.cross_entropy(z.reshape(-1,5),y.reshape(-1))
        loss.backward();o.step()
    for g in o.param_groups:g['lr']=a.peak_lr
    extra={'profile':a.profile,'seed':a.seed,'model_digest':model_digest(m),'branch_digest':branch_digest(m,o)};save(out,m,o,pr.warmup_steps,extra);(root/'branch.json').write_text(json.dumps(extra,indent=2)+'\n')

def arm(a):
    if a.condition is None: raise ValueError('--condition required')
    pr=PROFILES[a.profile];root=a.output/a.profile/f'seed{a.seed}';branch=root/'branch.pt'
    if not branch.exists(): raise SystemExit(f'missing common branch checkpoint: {branch}')
    name=a.condition if a.condition!='persistence' else f'persistence_h{a.persistence_h}';rd=root/name;rd.mkdir(parents=True,exist_ok=True);done=rd/'done.json'
    if done.exists() and a.resume: print(f'done: {rd}');return
    d=device_for(a.device);seed_all(a.seed);m=model_for(a,d);o=opt_for(m,a);ck=load(branch,m,o,d);start=branch_digest(m,o)
    if start!=ck['extra']['branch_digest']: raise SystemExit('branch checkpoint digest mismatch after load')
    for g in o.param_groups:g['lr']=a.peak_lr
    perm=all_permutations();xe,ye=fixed_eval(pr.eval_examples,EVAL_SEED,perm)
    keys=key_schedule(a.condition,pr.core_steps,pr.phase_steps,a.persistence_h) if a.condition in {'slow','fast','persistence'} else None
    sched=None
    if keys is not None:
        from core import schedule_digests
        sched=schedule_digests(keys)|{'condition':a.condition,'persistence_h':a.persistence_h,'n_steps':len(keys)};(rd/'schedule.json').write_text(json.dumps(sched,indent=2)+'\n')
    met=rd/'metrics.csv';fields=['step','exact_accuracy','token_accuracy','eval_loss'];rows=[{'step':0,**evaluate(m,xe,ye,d,a.eval_batch_size,a.precision)}]
    with met.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(rows[-1]);f.flush()
        for s in range(pr.core_steps):
            if a.condition=='uniform': xn,yn=make_uniform_batch(a.seed,s,a.batch_size,a.stream_seed,perm)
            elif a.condition=='static': xn,yn=make_power_batch(a.seed,'A',s,a.batch_size,a.alpha,a.mapping_seed,a.stream_seed,perm)
            else:
                mid,occ=keys[s];xn,yn=make_power_batch(a.seed,mid,occ,a.batch_size,a.alpha,a.mapping_seed,a.stream_seed,perm)
            x=torch.from_numpy(xn).long().to(d);y=torch.from_numpy(yn).long().to(d);o.zero_grad(set_to_none=True)
            with ac(d,a.precision): z=m(x);loss=F.cross_entropy(z.reshape(-1,5),y.reshape(-1))
            loss.backward();o.step();step=s+1
            if step%pr.eval_every==0 or step==pr.core_steps:
                row={'step':step,**evaluate(m,xe,ye,d,a.eval_batch_size,a.precision)};rows.append(row);w.writerow(row);f.flush()
            if a.save_every>0 and step%a.save_every==0 and step<pr.core_steps: save(rd/f'checkpoint_{step}.pt',m,o,step,{'branch_digest':start})
    cfg={'profile':a.profile,'profile_resolved':profile_dict(a.profile),'seed':a.seed,'condition':a.condition,'persistence_h':a.persistence_h,'batch_size':a.batch_size,'alpha':a.alpha,'mapping_seed':a.mapping_seed,'stream_seed':a.stream_seed,'peak_lr':a.peak_lr,'post_warmup_lr_schedule':'constant','eval_seed':EVAL_SEED,'branch_digest':start,'schedule':sched};(rd/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');final={'branch_digest':start,'final_model_digest':model_digest(m),'last':rows[-1]};done.write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final,indent=2))

def main():
    a=parse_args();warmup(a) if a.mode=='warmup' else arm(a)
if __name__=='__main__':main()
