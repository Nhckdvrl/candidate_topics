from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from collections import Counter
from deontic_structure import extract,compare

def rows(path:Path):
    if path.suffix.lower()=='.jsonl':
        for ln in path.read_text(encoding='utf-8').splitlines():
            if ln.strip(): yield json.loads(ln)
    else:
        with path.open(encoding='utf-8',newline='') as f: yield from csv.DictReader(f)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--original-col',default='original'); ap.add_argument('--simplified-col',default='simplified'); ap.add_argument('--out',type=Path,default=Path('simplification_audit.json')); a=ap.parse_args()
    n=eligible=0; flags=Counter(); mods=Counter(); examples=[]
    for r in rows(a.input):
        n+=1; o=str(r[a.original_col]); s=str(r[a.simplified_col]); eo=extract(o)
        if eo.modality=='NONE' and not eo.conditional and not eo.exception: continue
        eligible+=1; mods[eo.modality]+=1; c=compare(o,s)
        for k in ('modality_changed','condition_lost','exception_lost','negation_changed'): flags[k]+=int(c[k])
        if len(examples)<20 and any(c[k] for k in ('modality_changed','condition_lost','exception_lost','negation_changed')): examples.append({'original':o,'simplified':s,**c})
    out={'n_pairs':n,'n_deontic_eligible':eligible,'eligible_rate':eligible/max(n,1),'original_modality_hist':dict(mods),'structural_flag_counts':dict(flags),'gates':{'G_eligible_300':eligible>=300,'G_eligible_rate_5pct':eligible/max(n,1)>=.05,'G_multiple_modalities':sum(v>0 for k,v in mods.items() if k!='NONE')>=2},'examples':examples}
    a.out.write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8'); print(json.dumps(out['gates'],indent=2)); print('eligible',eligible,'/',n)
if __name__=='__main__': main()
