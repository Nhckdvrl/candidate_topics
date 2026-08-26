from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from decision_state import transition

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True,help='CSV columns id,source,summary'); ap.add_argument('--out',type=Path,default=Path('state_scores.jsonl')); a=ap.parse_args(); rows=[]
    with a.input.open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            z={'id':r.get('id')}; z.update(transition(r['source'],r['summary'])); rows.append(z)
    a.out.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n',encoding='utf-8')
    if rows: print(json.dumps({k:sum(bool(x[k]) for x in rows)/len(rows) for k in ('upgrade','downgrade','conditionality_lost')}|{'n':len(rows)},indent=2))
