#!/usr/bin/env python3
"""Analyze Topic-13 matched condition runs with frozen, interpretable gates."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

CONDS = ("fresh", "clustered", "random", "even")


def load_trials(run_dir: Path):
    trials=[]
    for seed_dir in sorted(run_dir.glob("seed_*")):
        rows={}
        for c in CONDS:
            p=seed_dir/c/"metrics.json"
            if p.exists():
                rows[c]=json.loads(p.read_text())
                lp=seed_dir/c/"eval_block_losses.npy"
                if lp.exists(): rows[c]["_eval_block_losses"]=np.load(lp)
        if set(rows)!=set(CONDS): continue
        fps={rows[c]["init_fingerprint"] for c in CONDS}
        if len(fps)!=1: raise RuntimeError(f"initialization mismatch in {seed_dir.name}: {fps}")
        trials.append((seed_dir.name,rows))
    return trials


def trial_effects(rows,min_relative_damage=0.005,spacing_fraction=0.25):
    L={c:float(rows[c]["final_eval_loss"]) for c in CONDS}; SE={c:float(rows[c]["final_eval_se_blocks"]) for c in CONDS}
    damage=L["random"]-L["fresh"]; gap_ce=L["clustered"]-L["even"]; gap_re=L["random"]-L["even"]; gap_cr=L["clustered"]-L["random"]

    def diff_se(a,b):
        xa=rows[a].get("_eval_block_losses"); xb=rows[b].get("_eval_block_losses")
        if xa is not None and xb is not None:
            xa=np.asarray(xa,dtype=float); xb=np.asarray(xb,dtype=float)
            if xa.shape!=xb.shape: raise RuntimeError(f"paired eval arrays disagree: {a} {xa.shape} vs {b} {xb.shape}")
            d=xa-xb
            return float(d.std(ddof=1)/math.sqrt(len(d))),"paired_block"
        return math.sqrt(SE[a]**2+SE[b]**2),"independent_fallback"

    se_damage,noise_mode_damage=diff_se("random","fresh"); se_ce,noise_mode_ce=diff_se("clustered","even")
    seed_gate=max(5.0*se_damage,min_relative_damage*L["fresh"])
    spacing_gate=max(5.0*se_ce,spacing_fraction*max(damage,0.0))
    return {"loss":L,"se":SE,"random_repetition_damage":damage,"clustered_minus_even":gap_ce,"random_minus_even":gap_re,"clustered_minus_random":gap_cr,"seed_reproduction_gate":seed_gate,"spacing_practical_gate":spacing_gate,"seed_reproduced":damage>seed_gate,"large_spacing_effect":abs(gap_ce)>spacing_gate if damage>0 else False,"spacing_direction":"even_better" if gap_ce>0 else "clustered_better" if gap_ce<0 else "tie","spacing_fraction_of_random_damage":gap_ce/damage if damage!=0 else None,"seed_difference_se":se_damage,"spacing_difference_se":se_ce,"seed_noise_mode":noise_mode_damage,"spacing_noise_mode":noise_mode_ce}


def bootstrap_mean(x,seed=13,n=10000):
    x=np.asarray(x,dtype=float)
    if len(x)<2: return [None,None]
    rng=np.random.default_rng(seed); out=[]
    for _ in range(n): out.append(float(rng.choice(x,size=len(x),replace=True).mean()))
    return [float(np.percentile(out,2.5)),float(np.percentile(out,97.5))]


def verdict(effects):
    n=len(effects); reproduced=sum(e["seed_reproduced"] for e in effects)
    if n==1:
        e=effects[0]
        if not e["seed_reproduced"]: return "PILOT_SETUP_FAIL_SEED_DAMAGE_NOT_REPRODUCED","The random-repetition seed phenomenon did not clear the frozen reproduction gate; do not interpret spacing."
        if e["large_spacing_effect"]: return "PILOT_PROMISING_RUN_CONFIRMATION",f"Spacing changed final loss by a practically large amount ({e['spacing_direction']}); run the frozen 3-seed confirmation."
        return "PILOT_WEAK_DO_NOT_TUNE","Seed repetition damage reproduced, but clustered-vs-even spacing did not clear the practical-effect gate. Do not sweep schedules to rescue it."

    dirs=[np.sign(e["clustered_minus_even"]) for e in effects if e["seed_reproduced"]]; large=[e for e in effects if e["seed_reproduced"] and e["large_spacing_effect"]]
    if reproduced<math.ceil(2*n/3): return "CONFIRM_FAIL_SEED_UNSTABLE","Random repetition damage itself was not reproduced in at least two-thirds of matched trials."
    if len(large)<math.ceil(2*n/3): return "CONFIRM_NO_LARGE_SPACING_EFFECT","A large clustered-vs-even spacing effect failed to appear in at least two-thirds of matched trials."
    nonzero=[d for d in dirs if d!=0]; dominant=max(nonzero.count(1),nonzero.count(-1)) if nonzero else 0
    if dominant<math.ceil(2*len(nonzero)/3): return "CONFIRM_SPACING_EFFECT_DIRECTION_UNSTABLE","Spacing mattered in magnitude but its direction was not stable across matched trials."
    direction="even_better" if nonzero.count(1)>nonzero.count(-1) else "clustered_better"
    return "GO_SPACING_IS_CAUSAL",f"Seed damage reproduced and a practically large spacing effect was stable across matched trials; dominant direction: {direction}."


def render_md(summary):
    lines=["# Topic 13 G-0 result","",f"**Verdict:** `{summary['verdict']}`","",summary["rationale"],"","## Matched trials","","| trial | fresh | clustered | random | even | random-fresh | clustered-even | seed gate | spacing gate |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in summary["trials"]:
        e=t["effects"]; L=e["loss"]
        lines.append(f"| {t['name']} | {L['fresh']:.6f} | {L['clustered']:.6f} | {L['random']:.6f} | {L['even']:.6f} | {e['random_repetition_damage']:.6f} | {e['clustered_minus_even']:.6f} | {e['seed_reproduction_gate']:.6f} | {e['spacing_practical_gate']:.6f} |")
    if summary.get("aggregate"):
        a=summary["aggregate"]
        lines += ["","## Aggregate","",f"- mean random repetition damage: `{a['mean_random_repetition_damage']:.6f}`; bootstrap CI {a['ci_random_repetition_damage']}",f"- mean clustered - even: `{a['mean_clustered_minus_even']:.6f}`; bootstrap CI {a['ci_clustered_minus_even']}",f"- seed reproduced: `{a['seed_reproduced']}/{a['n_trials']}`",f"- large spacing effect: `{a['large_spacing_effect']}/{a['n_trials']}`"]
    lines += ["","The spacing claim is only interpretable after the random-repetition condition reproduces the seed-paper damage relative to the matched fresh control. Gate standard errors use paired per-block held-out loss differences when available. The main causal comparison is clustered vs even: these conditions have the exact same document multiset and exact multiplicities."]
    return "\n".join(lines)+"\n"


def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-dir",type=Path,required=True); p.add_argument("--out-json",type=Path,required=True); p.add_argument("--out-md",type=Path,required=True); p.add_argument("--config",type=Path,default=Path(__file__).parent/"configs/g0.json"); args=p.parse_args()
    cfg=json.loads(args.config.read_text()); raw=load_trials(args.run_dir)
    if not raw: raise SystemExit("No complete matched trials found")
    trials=[{"name":name,"effects":trial_effects(rows,cfg["seed_reproduction_min_relative_loss"],cfg["spacing_min_fraction_of_random_damage"])} for name,rows in raw]
    v,r=verdict([t["effects"] for t in trials]); summary={"verdict":v,"rationale":r,"trials":trials}
    if len(trials)>=2:
        damages=[t["effects"]["random_repetition_damage"] for t in trials]; gaps=[t["effects"]["clustered_minus_even"] for t in trials]
        summary["aggregate"]={"n_trials":len(trials),"mean_random_repetition_damage":float(np.mean(damages)),"ci_random_repetition_damage":bootstrap_mean(damages),"mean_clustered_minus_even":float(np.mean(gaps)),"ci_clustered_minus_even":bootstrap_mean(gaps),"seed_reproduced":sum(t["effects"]["seed_reproduced"] for t in trials),"large_spacing_effect":sum(t["effects"]["large_spacing_effect"] for t in trials)}
    args.out_json.parent.mkdir(parents=True,exist_ok=True); args.out_md.parent.mkdir(parents=True,exist_ok=True)
    args.out_json.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n"); args.out_md.write_text(render_md(summary)); print(json.dumps({"verdict":v,"trials":len(trials)},indent=2))


if __name__=="__main__": main()
