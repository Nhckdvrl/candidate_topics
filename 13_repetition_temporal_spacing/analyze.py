#!/usr/bin/env python3
"""Analyze Topic-13 matched runs without arbitrary practical-effect thresholds."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any
import numpy as np
CONDS=("fresh","clustered","random","even")

def load_trials(run_dir: Path, expected_seeds: list[int], experiment_id: str | None = None):
    trials=[]
    for seed in expected_seeds:
        seed_dir=run_dir/f"seed_{seed}"; rows={}
        for c in CONDS:
            p=seed_dir/c/"metrics.json"; lp=seed_dir/c/"eval_block_losses.npy"
            if not (p.exists() and lp.exists()): continue
            row=json.loads(p.read_text()); row["_eval_block_losses"]=np.load(lp); rows[c]=row
        if set(rows)!=set(CONDS): continue
        fps={rows[c]["init_fingerprint"] for c in CONDS}
        if len(fps)!=1: raise RuntimeError(f"initialization mismatch in seed {seed}: {fps}")
        ids={rows[c].get("experiment_id") for c in CONDS}
        if len(ids)!=1: raise RuntimeError(f"experiment-id mismatch in seed {seed}: {ids}")
        if experiment_id is not None and next(iter(ids))!=experiment_id: raise RuntimeError(f"seed {seed} belongs to stale experiment {ids}, expected {experiment_id}")
        trials.append((f"seed_{seed}",rows))
    return trials

def paired_diff(rows: dict[str, dict[str, Any]], a: str, b: str, z: float=1.96):
    xa=np.asarray(rows[a]["_eval_block_losses"],dtype=float); xb=np.asarray(rows[b]["_eval_block_losses"],dtype=float)
    if xa.shape!=xb.shape: raise RuntimeError(f"paired eval arrays disagree: {a} {xa.shape} vs {b} {xb.shape}")
    d=xa-xb; mean=float(d.mean()); se=float(d.std(ddof=1)/math.sqrt(len(d))); return mean,se,[mean-z*se,mean+z*se]

def trial_effects(rows,z=1.96):
    L={c:float(rows[c]["final_eval_loss"]) for c in CONDS}
    damage,se_damage,ci_damage=paired_diff(rows,"random","fresh",z); gap_ce,se_ce,ci_ce=paired_diff(rows,"clustered","even",z); gap_re,se_re,ci_re=paired_diff(rows,"random","even",z); gap_cr,se_cr,ci_cr=paired_diff(rows,"clustered","random",z)
    seed_reproduced=ci_damage[0]>0; spacing_positive=ci_ce[0]>0; spacing_negative=ci_ce[1]<0; spacing_detected=spacing_positive or spacing_negative
    return {"loss":L,"random_repetition_damage":damage,"random_repetition_damage_se":se_damage,"random_repetition_damage_ci95":ci_damage,"seed_reproduced":seed_reproduced,"clustered_minus_even":gap_ce,"clustered_minus_even_se":se_ce,"clustered_minus_even_ci95":ci_ce,"spacing_detected_within_trial":spacing_detected,"spacing_direction":"even_better" if gap_ce>0 else "clustered_better" if gap_ce<0 else "tie","random_minus_even":gap_re,"random_minus_even_ci95":ci_re,"clustered_minus_random":gap_cr,"clustered_minus_random_ci95":ci_cr,"spacing_fraction_of_random_damage":gap_ce/damage if damage!=0 else None}

def verdict(effects, mode: str, expected_n: int | None = None):
    if mode=="pilot":
        if len(effects)!=1: return "PILOT_INCOMPLETE", "Pilot analysis expects exactly one complete matched trial."
        e=effects[0]
        if not e["seed_reproduced"]: return "PILOT_SETUP_FAIL_REPRODUCTION_NOT_TOPIC_FAIL", "The random-repetition seed phenomenon was not clearly reproduced. Do not judge the spacing hypothesis from this run; first repair reproduction fidelity or rerun the locked setup if there was an engineering fault."
        if e["spacing_detected_within_trial"]: return "PILOT_SIGNAL_RUN_CONFIRMATION", f"The seed phenomenon reproduced and one matched trial already shows a spacing signal ({e['spacing_direction']}). Run the frozen confirmation; do not tune the schedule."
        return "PILOT_NULL_BUT_RUN_CONFIRMATION", "The seed phenomenon reproduced but one matched trial did not resolve a spacing effect. A single training seed is not allowed to kill the topic; run the frozen multi-pool confirmation once."
    n=len(effects)
    if expected_n is not None and n<expected_n: return "CONFIRM_INCOMPLETE", f"Need all {expected_n} preregistered confirmation trials before a scientific verdict; only {n} are complete."
    if n<3: return "CONFIRM_INCOMPLETE", "Need the preregistered confirmation trials before a scientific verdict."
    reproduced=sum(e["seed_reproduced"] for e in effects)
    if reproduced<math.ceil(2*n/3): return "CONFIRM_SETUP_UNSTABLE_REPRODUCTION", "The prerequisite repetition damage is not stable across repeated-pool/model seeds. This invalidates the current setup as a test of spacing; it is not evidence that spacing itself is false."
    gaps=np.asarray([e["clustered_minus_even"] for e in effects if e["seed_reproduced"]],dtype=float); sig_dirs=[]
    for e in effects:
        if not e["seed_reproduced"] or not e["spacing_detected_within_trial"]: continue
        sig_dirs.append(1 if e["clustered_minus_even"]>0 else -1)
    all_dirs=[1 if g>0 else -1 if g<0 else 0 for g in gaps]; pos=sum(d>0 for d in all_dirs); neg=sum(d<0 for d in all_dirs); dominant=max(pos,neg); detected_same_direction=max(sig_dirs.count(1),sig_dirs.count(-1)) if sig_dirs else 0; need=max(1,min(3,len(gaps)))
    if dominant==len(gaps) and detected_same_direction>=need:
        direction="even_better" if pos>neg else "clustered_better"; return "GO_STRONG_SPACING_IS_CAUSAL", f"Across independent repeated pools/model seeds, the clustered-vs-even effect has one direction in every valid trial and clears the paired held-out interval in at least {need} trials; dominant direction: {direction}."
    if dominant==len(gaps):
        direction="even_better" if pos>neg else "clustered_better"; return "PROMISING_SPACING_EFFECT_NEEDS_MORE_REPLICATION", f"All valid trials agree in direction ({direction}) but the effect is not individually resolved in enough trials. This is a real lead, not a kill; scale replication before method work."
    if dominant>=math.ceil(2*len(gaps)/3) and abs(float(gaps.mean()))>0: return "INCONCLUSIVE_DIRECTION_MOSTLY_STABLE", "Most valid trials agree in direction, but the multi-trial evidence is not clean enough to claim a stable spacing law. Do not tune schedules; one larger confirmation is the only justified next step."
    return "NO_EVIDENCE_SPACING_IN_LOCKED_TEST", "The prerequisite repetition damage reproduced, but the clean optimizer-step spacing manipulation did not show a stable direction across independent repeated pools. This is a substantive negative result for the current hypothesis."

def aggregate_summary(trials):
    damages=np.asarray([t["effects"]["random_repetition_damage"] for t in trials],dtype=float); gaps=np.asarray([t["effects"]["clustered_minus_even"] for t in trials],dtype=float); ratios=[abs(t["effects"]["spacing_fraction_of_random_damage"]) for t in trials if t["effects"]["spacing_fraction_of_random_damage"] is not None]
    return {"n_trials":len(trials),"mean_random_repetition_damage":float(damages.mean()),"mean_clustered_minus_even":float(gaps.mean()),"sd_clustered_minus_even_across_trials":float(gaps.std(ddof=1)) if len(gaps)>1 else None,"mean_abs_spacing_fraction_of_damage":float(np.mean(ratios)) if ratios else None,"seed_reproduced":sum(t["effects"]["seed_reproduced"] for t in trials),"within_trial_spacing_detected":sum(t["effects"]["spacing_detected_within_trial"] for t in trials)}

def render_md(summary):
    lines=["# Topic 13 result","",f"**Verdict:** `{summary['verdict']}`","",summary["rationale"],"","## Matched trials","","| trial | fresh | random | clustered | even | random-fresh (95% CI) | clustered-even (95% CI) |","|---|---:|---:|---:|---:|---:|---:|"]
    for t in summary["trials"]:
        e=t["effects"]; L=e["loss"]; cd=e["random_repetition_damage_ci95"]; cs=e["clustered_minus_even_ci95"]; lines.append(f"| {t['name']} | {L['fresh']:.6f} | {L['random']:.6f} | {L['clustered']:.6f} | {L['even']:.6f} | {e['random_repetition_damage']:.6f} [{cd[0]:.6f},{cd[1]:.6f}] | {e['clustered_minus_even']:.6f} [{cs[0]:.6f},{cs[1]:.6f}] |")
    if summary.get("aggregate"):
        a=summary["aggregate"]; lines += ["","## Across-trial summary","",f"- mean random repetition damage: `{a['mean_random_repetition_damage']:.6f}`",f"- mean clustered - even: `{a['mean_clustered_minus_even']:.6f}`",f"- across-trial SD of clustered - even: `{a['sd_clustered_minus_even_across_trials']}`",f"- mean |spacing effect / random damage|: `{a['mean_abs_spacing_fraction_of_damage']:.3f}` (descriptive only; no hard practical threshold)",f"- seed phenomenon reproduced: `{a['seed_reproduced']}/{a['n_trials']}`",f"- within-trial paired interval excludes 0: `{a['within_trial_spacing_detected']}/{a['n_trials']}`"]
    lines += ["","Interpretation order is strict but not kill-oriented: first establish that the chosen training setup actually exhibits repetition damage; then ask whether the optimizer-step-matched spacing manipulation has a stable effect. Pilot nulls never kill the topic. Practical effect size is reported continuously rather than thresholded at an arbitrary fraction of seed damage."]; return "\n".join(lines)+"\n"

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-dir",type=Path,required=True); p.add_argument("--out-json",type=Path,required=True); p.add_argument("--out-md",type=Path,required=True); p.add_argument("--config",type=Path,default=Path(__file__).parent/"configs/g0.json"); p.add_argument("--mode",choices=["pilot","confirm"],required=True); p.add_argument("--experiment-id",default=None); args=p.parse_args()
    cfg=json.loads(args.config.read_text()); expected=cfg["pilot_seeds"] if args.mode=="pilot" else cfg["confirmation_seeds"]; raw=load_trials(args.run_dir,[int(x) for x in expected],args.experiment_id)
    if not raw: raise SystemExit("No complete matched trials found")
    trials=[{"name":name,"effects":trial_effects(rows,float(cfg.get("paired_ci_z",1.96)))} for name,rows in raw]; v,r=verdict([t["effects"] for t in trials],args.mode,expected_n=len(expected)); summary={"verdict":v,"rationale":r,"mode":args.mode,"experiment_id":args.experiment_id,"trials":trials}
    if len(trials)>=2: summary["aggregate"]=aggregate_summary(trials)
    args.out_json.parent.mkdir(parents=True,exist_ok=True); args.out_md.parent.mkdir(parents=True,exist_ok=True); args.out_json.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n"); args.out_md.write_text(render_md(summary)); print(json.dumps({"verdict":v,"trials":len(trials)},indent=2))
if __name__=="__main__": main()
