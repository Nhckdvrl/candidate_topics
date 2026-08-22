#!/usr/bin/env python3
import argparse,json
from core import PROFILES,head_overlap,key_schedule,map_orders,max_map_run,schedule_digests
p=argparse.ArgumentParser();p.add_argument('--profile',choices=sorted(PROFILES),default='pilot');p.add_argument('--mapping-seed',type=int,default=1729);a=p.parse_args();pr=PROFILES[a.profile]
slow=key_schedule('slow',pr.core_steps,pr.phase_steps);fast=key_schedule('fast',pr.core_steps,pr.phase_steps);ma,mb=map_orders(a.mapping_seed);ds=schedule_digests(slow);df=schedule_digests(fast)
report={'profile':a.profile,'core_steps':pr.core_steps,'phase_steps':pr.phase_steps,'head_overlap_top20pct':head_overlap(ma,mb),'slow_max_map_run':max_map_run(slow),'fast_max_map_run':max_map_run(fast),'slow':ds,'fast':df,'same_multiset':ds['multiset_digest']==df['multiset_digest'],'different_order':ds['temporal_digest']!=df['temporal_digest']}
if not report['same_multiset'] or not report['different_order'] or report['fast_max_map_run']!=1 or report['slow_max_map_run']!=pr.phase_steps:raise SystemExit('schedule audit failed: '+json.dumps(report))
print(json.dumps(report,indent=2))
