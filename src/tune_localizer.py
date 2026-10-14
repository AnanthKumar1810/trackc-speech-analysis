"""Select four fixed event thresholds using train speaker only."""
from pathlib import Path
from collections import defaultdict
from itertools import product
import numpy as np
from core import read, loadl, dump
from localizer import detect
from evaluate_localizer import overlap

ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=[m for m in loadl(ROOT/'metadata/manifest.jsonl') if m['split']=='train']
    labels={r['clip_id']:r for r in loadl(ROOT/'metadata/flaw_labels.jsonl')}
    words=defaultdict(list)
    for r in loadl(ROOT/'alignment_mfa/word_alignment_mfa.jsonl'):words[r['clip_id']].append(r)
    signals={}
    for m in manifest:
        signals[m['clip_id']]=(read(ROOT/m['audio_path']),dict(np.load(ROOT/m['feature_path'])))
    candidates=[]
    # Include thresholds below the known moderate drag ratio (~1.39), so the
    # train-only selection cannot make that labeled severity undetectable.
    for pause,quiet,rush,drag in product((.16,.25,.4),(-4.,-6.,-9.),(.70,.75,.82,.88),(1.10,1.18,1.22,1.30)):
        threshold={'pause_excess_s':pause,'quiet_db':quiet,'rush_ratio':rush,'drag_ratio':drag}
        tp=predicted=positive=0;group_total=defaultdict(int);group_hit=defaultdict(int)
        for m in manifest:
            clip=m['clip_id'];base=m['family_id']+'_base'
            bx,bf=signals[base];ux,uf=signals[clip]
            ev=detect(bx,ux,words[base],words[clip],same_setup=True,thresholds=threshold,
                      reference_features=bf,take_features=uf)['events']
            predicted+=len(ev)
            truth=labels.get(clip)
            if truth:
                positive+=1
                key=(truth['flaw_type'],truth['severity']);group_total[key]+=1
                interval=(truth['start_seconds'],truth['end_seconds'])
                if any(e['type']==truth['flaw_type'] and overlap((e['start_seconds'],e['end_seconds']),interval)>=.5 for e in ev):
                    tp+=1;group_hit[key]+=1
        precision=tp/predicted if predicted else 0;recall=tp/positive
        f1=2*precision*recall/(precision+recall) if precision+recall else 0
        severity_recall=np.mean([group_hit[k]/n for k,n in group_total.items()])
        # Balance aggregate precision/recall with equal weight for each
        # represented flaw/severity cell; do not tune on validation or test.
        objective=.5*f1+.5*severity_recall
        candidates.append((objective,f1,precision,recall,severity_recall,threshold))
    best=max(candidates,key=lambda x:(x[0],x[2]))
    out=ROOT/'localizer';out.mkdir(exist_ok=True)
    dump(out/'thresholds.json',{'selection':'Train-only grid search on Reagan; maximize equal blend of micro event F1 and macro recall over flaw type × severity at IoU >= 0.5. Validation/test are not used.',
                                'grid_size':len(candidates),'thresholds':best[5],
                                'train_objective':round(best[0],4),'train_f1':round(best[1],4),
                                'train_precision':round(best[2],4),'train_recall':round(best[3],4),
                                'train_macro_severity_recall':round(best[4],4)})
    print(best)

if __name__=='__main__':main()
