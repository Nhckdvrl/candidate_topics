#!/usr/bin/env python3
import argparse,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--profile',default='pilot');p.add_argument('--seeds',default='0');p.add_argument('--output',default='outputs');p.add_argument('--resume',action='store_true');a=p.parse_args();seeds=[int(x) for x in a.seeds.split(',') if x.strip()];ngpu=max(1,int(os.environ.get('N_GPUS',os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3').count(',')+1)))
for seed in seeds:
    warm=[sys.executable,'train.py','--mode','warmup','--profile',a.profile,'--seed',str(seed),'--output',a.output]
    if a.resume:warm.append('--resume')
    subprocess.run(warm,check=True);procs=[]
    for i,arm in enumerate(['uniform','static','slow','fast']):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(i%ngpu);cmd=[sys.executable,'train.py','--mode','arm','--condition',arm,'--profile',a.profile,'--seed',str(seed),'--output',a.output]
        if a.resume:cmd.append('--resume')
        procs.append(subprocess.Popen(cmd,env=env))
    codes=[p.wait() for p in procs]
    if any(codes):raise SystemExit(f'arm failure codes={codes}')
subprocess.run([sys.executable,'analyze.py','--root',str(Path(a.output)/a.profile),'--profile',a.profile],check=True)
