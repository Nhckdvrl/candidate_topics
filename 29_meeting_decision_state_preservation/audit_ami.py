from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter
from decision_state import classify_state

def iter_json_files(root: Path):
    for p in root.rglob('*.json'):
        try: x=json.loads(p.read_text(encoding='utf-8'))
        except Exception: continue
        yield p,x

def audit(root: Path) -> dict:
    n_decisions=n_linked=n_multi=n_temporal=n_stateful=0
    linked_turn_counts=Counter(); state_counts=Counter(); meetings=set(); examples=[]
    for p,obj in iter_json_files(root):
        if not isinstance(obj,list): continue
        for item in obj:
            if not isinstance(item,dict): continue
            a=item.get('abstractive'); ex=item.get('extractive')
            if not isinstance(a,dict) or not isinstance(ex,list): continue
            if str(a.get('type','')).lower()!='decisions': continue
            n_decisions+=1; meetings.add(p.stem)
            linked=[z for z in ex if isinstance(z,dict) and z.get('text')]
            n_linked+=int(bool(linked)); linked_turn_counts[len(linked)]+=1
            if len(linked)>=2: n_multi+=1
            times=[]
            for z in linked:
                try: times.append(float(z.get('starttime')))
                except Exception: pass
            if len(times)>=2 and max(times)-min(times)>=15: n_temporal+=1
            joined=' '.join(z['text'] for z in linked); st=classify_state(joined).state; state_counts[st]+=1
            if st!='OPEN': n_stateful+=1
            if len(examples)<12 and linked: examples.append({'decision':a.get('text'),'n_links':len(linked),'source_state':st,'source':[z['text'] for z in linked[:5]]})
    return {'root':str(root),'n_decision_abstracts':n_decisions,'n_with_linked_utterances':n_linked,'n_multiturn_linked':n_multi,'n_temporally_extended_ge15s':n_temporal,'n_with_explicit_state_cue':n_stateful,'linked_turn_hist':dict(linked_turn_counts),'source_state_hist':dict(state_counts),'meeting_files_with_decisions':len(meetings),'examples':examples,'gates':{'G_support_200':n_decisions>=200,'G_multiturn_100':n_multi>=100,'G_temporal_75':n_temporal>=75,'G_stateful_100':n_stateful>=100}}

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out',type=Path,default=Path('ami_audit.json')); a=ap.parse_args(); r=audit(a.root); a.out.write_text(json.dumps(r,indent=2,ensure_ascii=False),encoding='utf-8'); print(json.dumps(r['gates'],indent=2)); print('decision abstracts',r['n_decision_abstracts'])
