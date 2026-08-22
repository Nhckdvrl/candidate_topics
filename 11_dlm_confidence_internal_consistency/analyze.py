#!/usr/bin/env python3
"""Analyze Topic-11 v3 retroactive factorial and emit a locked verdict."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

CELLS = ("CC","IC","CW","IW")
PRIMARY_METRIC = "confidence_result_middle"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def orientation_effect(c: dict[str,float]) -> dict[str,float]:
    cons_correct = c["CC"] - c["IC"]; cons_wrong = c["CW"] - c["IW"]
    corr_consistent = c["CC"] - c["CW"]; corr_inconsistent = c["IC"] - c["IW"]
    return {"consistency_when_correct": cons_correct, "consistency_when_wrong": cons_wrong,
        "delta_consistency": 0.5*(cons_correct+cons_wrong), "correctness_when_consistent": corr_consistent,
        "correctness_when_inconsistent": corr_inconsistent, "delta_correctness": 0.5*(corr_consistent+corr_inconsistent),
        "coherent_wrong_minus_incoherent_correct": c["CW"] - c["IC"],
        "prompt_check_match_interaction": cons_correct - cons_wrong}


def build_pair_effects(rows, metric):
    nested = defaultdict(lambda: defaultdict(dict)); seen = set()
    for r in rows:
        key=(int(r["pair_id"]), int(r["orientation"]), r["cell"])
        if key in seen: raise RuntimeError(f"duplicate score row: {key}")
        seen.add(key); nested[key[0]][key[1]][key[2]] = float(r[metric])
    pair_ids=[]; by_key=defaultdict(list)
    for pid in sorted(nested):
        if set(nested[pid]) != {0,1} or any(set(nested[pid][o]) != set(CELLS) for o in (0,1)): continue
        e0=orientation_effect(nested[pid][0]); e1=orientation_effect(nested[pid][1]); pair_ids.append(pid)
        for k in e0: by_key[k].append(0.5*(e0[k]+e1[k]))
        by_key["mirror_consistency_product"].append(e0["delta_consistency"]*e1["delta_consistency"])
    return pair_ids, {k:np.asarray(v,dtype=np.float64) for k,v in by_key.items()}


def bootstrap_ci(x, rng, n_boot):
    if len(x)<2: return (float("nan"),float("nan"))
    means=np.empty(n_boot,dtype=np.float64); n=len(x); pos=0
    while pos<n_boot:
        k=min(1000,n_boot-pos); idx=rng.integers(0,n,size=(k,n)); means[pos:pos+k]=x[idx].mean(axis=1); pos+=k
    return tuple(map(float,np.percentile(means,[2.5,97.5])))


def sign_flip_pvalue(x, rng, n_perm):
    if len(x)==0: return float("nan")
    observed=float(x.mean()); n=len(x); exceed=0; done=0
    while done<n_perm:
        k=min(1000,n_perm-done); signs=rng.choice(np.array([-1.,1.]),size=(k,n))
        exceed += int(np.sum((signs*x[None,:]).mean(axis=1)>=observed)); done += k
    return float((exceed+1)/(n_perm+1))


def summarize_effects(effects, seed, n_boot, n_perm):
    out={}
    for i,(k,x) in enumerate(sorted(effects.items())):
        if k=="mirror_consistency_product":
            out[k]={"fraction_same_sign":float(np.mean(x>0)),"fraction_opposite_sign":float(np.mean(x<0))}; continue
        ci=bootstrap_ci(x,np.random.default_rng(seed+i*7919),n_boot)
        out[k]={"mean":float(x.mean()),"median":float(np.median(x)),"ci95":list(ci),"fraction_positive":float(np.mean(x>0)),
                "p_signflip_one_sided":sign_flip_pvalue(x,np.random.default_rng(seed+1000003+i*7919),n_perm),"n_pairs":int(len(x))}
    return out


def cell_means(rows, metric):
    d=defaultdict(list)
    for r in rows: d[r["cell"]].append(float(r[metric]))
    return {k:float(np.mean(v)) for k,v in d.items()}


def summarize_probe(rows, probe_type, seed, n_boot, n_perm):
    subset=[r for r in rows if r.get("probe_type")==probe_type]
    if not subset: raise RuntimeError(f"missing protocol probe type: {probe_type}")
    gaps=np.asarray([float(r["gap"]) for r in subset]); c=np.asarray([float(r["confidence_correct"]) for r in subset]); w=np.asarray([float(r["confidence_wrong"]) for r in subset])
    offset = 17011 if probe_type == "arithmetic_result" else 29021
    ci=bootstrap_ci(gaps,np.random.default_rng(seed+offset),n_boot)
    return {"n_pairs":len(subset),"correct_mean":float(c.mean()),"wrong_mean":float(w.mean()),"gap_mean":float(gaps.mean()),"gap_ci95":list(ci),
            "fraction_positive":float(np.mean(gaps>0)),"p_signflip_one_sided":sign_flip_pvalue(gaps,np.random.default_rng(seed+333333+offset),n_perm)}


def verdict(protocol, metrics, min_arithmetic_probe_gap, min_alias_probe_gap, min_primary_effect):
    reasons=[]
    gates = (("arithmetic_result", "seed-paper arithmetic", min_arithmetic_probe_gap),("semantic_alias", "semantic-alias comprehension", min_alias_probe_gap))
    for name,label,floor in gates:
        q=protocol[name]
        if q["gap_ci95"][0] <= 0 or q["gap_mean"] < floor:
            return "INVALID_PROTOCOL_DO_NOT_INTERPRET", [f"{label} prerequisite failed the locked magnitude/stability gate (mean={q['gap_mean']:.4f}, CI={q['gap_ci95']}, floor={floor:.4f})."]
    reasons.append("Both scoring prerequisites passed: arithmetic discrimination and semantic-alias comprehension.")
    e=metrics[PRIMARY_METRIC]["effects"]; cons=e["delta_consistency"]; lo,hi=cons["ci95"]
    if hi < min_primary_effect:
        reasons.append(f"The 95% CI excludes the preregistered meaningful retroactive effect floor ({min_primary_effect:.3f}).")
        return "KILL_NO_MEANINGFUL_RETROACTIVE_SIGNAL", reasons
    if lo <= 0:
        reasons.append("The CI still includes zero while allowing a scientifically meaningful effect; do not tune the design. A larger frozen-design confirmation may resolve power.")
        return "INCONCLUSIVE_FROZEN_DESIGN", reasons
    if cons["mean"] < min_primary_effect:
        reasons.append("A positive retroactive signal exists, but its mean is below the meaningful-effect floor.")
        return "WEAK_REAL_BUT_TOO_SMALL", reasons
    c1=e["consistency_when_correct"]; c0=e["consistency_when_wrong"]
    if c1["mean"] <= 0 or c0["mean"] <= 0:
        reasons.append("The positive main effect reverses sign in one external-correctness stratum.")
        return "MIXED_INTERACTION_DEPENDENT", reasons
    reasons.append("A meaningful positive consistency effect appears on unchanged Step-2/3 result tokens that occur before the future consistency check, in both correctness strata.")
    cross=e["coherent_wrong_minus_incoherent_correct"]
    if cross["ci95"][0] > 0:
        reasons.append("Coherent-but-wrong also stably outranks incoherent-but-correct on the retroactive metric.")
        return "GO_STRONG_COHERENCE_OVER_CORRECTNESS", reasons
    reasons.append("Internal consistency is independently identified, but it does not stably dominate external correctness; the topic stands on the retroactive consistency signal itself.")
    return "GO_RETROACTIVE_CONSISTENCY_SIGNAL", reasons


def render_markdown(s):
    lines=["# Topic 11 G-0 Result","",f"**Verdict:** `{s['verdict']}`","",f"Eligible mirrored anchor pairs: **{s['n_pairs']}**","","## Protocol prerequisites",""]
    for k,label in (("arithmetic_result","Arithmetic result substitution"),("semantic_alias","Semantic-alias comprehension")):
        p=s["protocol_probe"][k]; lines += [f"- **{label}:** gap={p['gap_mean']:.6f}, 95% CI [{p['gap_ci95'][0]:.6f}, {p['gap_ci95'][1]:.6f}], positive pairs={p['fraction_positive']:.3f}"]
    lines += ["","If either prerequisite fails, the factorial is invalid rather than scientifically negative.","","## Locked factorial results","","| Metric | Effect | Mean | 95% CI | Positive pairs | sign-flip p |","|---|---|---:|---:|---:|---:|"]
    metrics=("confidence_result_middle","confidence_result_first","confidence_result_final","confidence_result_all","confidence_trajectory","confidence_full")
    effects=("delta_consistency","delta_correctness","consistency_when_correct","consistency_when_wrong","coherent_wrong_minus_incoherent_correct","prompt_check_match_interaction")
    for m in metrics:
        es=s[m]["effects"]
        for k in effects:
            x=es[k]; lines.append(f"| {m} | {k} | {x['mean']:.6f} | [{x['ci95'][0]:.6f}, {x['ci95'][1]:.6f}] | {x['fraction_positive']:.3f} | {x['p_signflip_one_sided']:.4g} |")
    lines += ["","## Decision rationale",""] + [f"- {r}" for r in s["reasons"]]
    lines += ["","The primary metric is `confidence_result_middle` (Step 2 and Step 3 result tokens). These tokens are byte/token-identical across the 2x2 and occur *before* the internal-consistency suffix intervention. Therefore its consistency contrast is a retroactive final-forward effect, not confidence on the changed suffix token itself.","",f"Locked arithmetic-probe floor: {s['design']['min_arithmetic_probe_gap']:.3f}; semantic-alias floor: {s['design']['min_alias_probe_gap']:.3f}; meaningful primary-effect floor: {s['design']['min_primary_effect']:.3f}."]
    return "\n".join(lines)+"\n"


def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--protocol-probe",type=Path,required=True); p.add_argument("--out-json",type=Path,required=True); p.add_argument("--out-md",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=10000); p.add_argument("--permutations",type=int,default=20000); p.add_argument("--seed",type=int,default=20260822); p.add_argument("--min-arithmetic-probe-gap",type=float,default=0.10); p.add_argument("--min-alias-probe-gap",type=float,default=0.02); p.add_argument("--min-primary-effect",type=float,default=0.01); a=p.parse_args()
    rows=read_jsonl(a.input); probes=read_jsonl(a.protocol_probe)
    if not rows or not probes: raise SystemExit("missing scores/probes")
    metric_names=("confidence_result_middle","confidence_result_first","confidence_result_final","confidence_result_all","confidence_trajectory","confidence_full","confidence_check")
    summary={}; ref=None
    for m in metric_names:
        pids,effects=build_pair_effects(rows,m)
        if ref is None: ref=pids
        elif pids != ref: raise RuntimeError("metric pair sets disagree")
        summary[m]={"cell_means":cell_means(rows,m),"effects":summarize_effects(effects,a.seed,a.bootstrap,a.permutations)}
    summary["n_pairs"]=len(ref or []); summary["protocol_probe"]={"arithmetic_result":summarize_probe(probes,"arithmetic_result",a.seed,a.bootstrap,a.permutations),"semantic_alias":summarize_probe(probes,"semantic_alias",a.seed+1,a.bootstrap,a.permutations)}
    summary["design"]={"primary_identification_metric":PRIMARY_METRIC,"paper_compatible_metric":"confidence_full","unit_of_resampling":"mirrored anchor pair","bootstrap":a.bootstrap,"sign_flip_permutations":a.permutations,"seed":a.seed,"min_arithmetic_probe_gap":a.min_arithmetic_probe_gap,"min_alias_probe_gap":a.min_alias_probe_gap,"min_primary_effect":a.min_primary_effect}
    summary["verdict"],summary["reasons"]=verdict(summary["protocol_probe"],summary,a.min_arithmetic_probe_gap,a.min_alias_probe_gap,a.min_primary_effect)
    a.out_json.parent.mkdir(parents=True,exist_ok=True); a.out_md.parent.mkdir(parents=True,exist_ok=True); a.out_json.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8"); a.out_md.write_text(render_markdown(summary),encoding="utf-8"); print(json.dumps({"verdict":summary["verdict"],"n_pairs":summary["n_pairs"]},indent=2))


if __name__=="__main__":
    main()
