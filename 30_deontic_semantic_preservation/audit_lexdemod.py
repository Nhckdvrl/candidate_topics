from __future__ import annotations
import argparse,csv,ast,json
from pathlib import Path
from collections import Counter
LABELS=['obl','ent','pro','per','pow','dis','none']

def audit_csv(path:Path)->dict:
    counts=Counter(); span_counts=Counter(); n=0; unique_cid=set(); malformed=0
    with path.open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            n+=1; unique_cid.add(r.get('cid',''))
            try:
                vec=ast.literal_eval(r['label']); spans=ast.literal_eval(r['span'])
                if len(vec)!=7: raise ValueError
                lab=LABELS[max(range(7),key=lambda i:vec[i])]; counts[lab]+=1
                for k,v in spans.items(): span_counts[k]+=len(v)
            except Exception: malformed+=1
    non_none=sum(v for k,v in counts.items() if k!='none')
    return {'file':str(path),'n_rows':n,'n_unique_clauses':len(unique_cid),'label_counts':dict(counts),'span_counts':dict(span_counts),'malformed':malformed,'non_none':non_none,'gates':{'G_gold_non_none_500':non_none>=500,'G_unique_clauses_250':len(unique_cid)>=250,'G_parse_clean_99pct':malformed/max(n,1)<=.01}}

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--csv',type=Path,required=True); ap.add_argument('--out',type=Path,default=Path('lexdemod_audit.json')); a=ap.parse_args(); r=audit_csv(a.csv); a.out.write_text(json.dumps(r,indent=2),encoding='utf-8'); print(json.dumps(r,indent=2))
