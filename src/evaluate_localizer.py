"""Evaluate predicted event intervals on speaker-disjoint digital-edit splits."""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
from core import read, loadl, dump
from localizer import detect, DEFAULTS

ROOT=Path(__file__).resolve().parents[1]

def overlap(a,b):
    left=max(a[0],b[0]);right=min(a[1],b[1])
    return max(0,right-left)/max(1e-9,max(a[1],b[1])-min(a[0],b[0]))

def main():
    threshold_file=ROOT/'localizer/thresholds.json'
    thresholds=json.loads(threshold_file.read_text(encoding='utf-8'))['thresholds'] if threshold_file.exists() else DEFAULTS
    manifest={r['clip_id']:r for r in loadl(ROOT/'metadata/manifest.jsonl')}
    labels={r['clip_id']:r for r in loadl(ROOT/'metadata/flaw_labels.jsonl')}
    words=defaultdict(list)
    for r in loadl(ROOT/'alignment_mfa/word_alignment_mfa.jsonl'):words[r['clip_id']].append(r)
    rows=[];cache={}
    def get(clip):
        if clip not in cache:
            m=manifest[clip]
            cache[clip]=(read(ROOT/m['audio_path']),dict(np.load(ROOT/m['feature_path'])))
        return cache[clip]
    for clip,m in manifest.items():
        base=m['family_id']+'_base'
        bx,bf=get(base);ux,uf=get(clip)
        result=detect(bx,ux,words[base],words[clip],same_setup=True,thresholds=thresholds,
                      reference_features=bf,take_features=uf)
        truth=labels.get(clip)
        predicted=result['events'];same=[e for e in predicted if truth and e['type']==truth['flaw_type']]
        true_interval=(truth['start_seconds'],truth['end_seconds']) if truth else None
        best=max(same,key=lambda e:overlap((e['start_seconds'],e['end_seconds']),true_interval),default=None) if truth else None
        iou=overlap((best['start_seconds'],best['end_seconds']),true_interval) if best else 0
        rows.append({'clip_id':clip,'split':m['split'],'true_type':m['flaw_type'],
                     'true_severity':m['severity'],'family_id':m['family_id'],'status':result['status'],
                     'predicted_events':predicted,'scores':result['scores'],
                     'best_same_type_iou':round(iou,4),
                     'type_detected':bool(same),'localized_at_iou_0_5':iou>=.5,
                     'localized_at_iou_0_3':iou>=.3,
                     'localized_with_200ms_collar':bool(best and abs(best['start_seconds']-true_interval[0])<=.2 and abs(best['end_seconds']-true_interval[1])<=.2) if truth else False,
                     'onset_error_seconds':round(abs(best['start_seconds']-true_interval[0]),3) if best else None,
                     'offset_error_seconds':round(abs(best['end_seconds']-true_interval[1]),3) if best else None})
    report={'scope':'Digital edits only; no human speech-quality accuracy is implied.',
            'thresholds':thresholds,'splits':{}}
    for split in ('train','validation','test'):
        subset=[r for r in rows if r['split']==split]
        positive=[r for r in subset if r['true_type']!='none']
        exact=[r for r in positive if r['localized_at_iou_0_5']]
        total_predictions=sum(len(r['predicted_events']) for r in subset)
        precision=len(exact)/total_predictions if total_predictions else 0
        recall=len(exact)/len(positive)
        report['splits'][split]={
            'clips':len(subset),'positive_clips':len(positive),
            'type_recall':round(sum(r['type_detected'] for r in positive)/len(positive),4),
            'localization_recall_iou_0_5':round(recall,4),
            'event_precision_iou_0_5':round(precision,4),
            'event_f1_iou_0_5':round(2*precision*recall/(precision+recall),4) if precision+recall else 0,
            'mean_best_same_type_iou':round(float(np.mean([r['best_same_type_iou'] for r in positive])),4),
            'mean_onset_error_s_localized':round(float(np.mean([r['onset_error_seconds'] for r in exact])),3) if exact else None,
            'mean_offset_error_s_localized':round(float(np.mean([r['offset_error_seconds'] for r in exact])),3) if exact else None,
            'false_positive_events':sum(len(r['predicted_events'])-(1 if r['localized_at_iou_0_5'] else 0) for r in subset),
            'base_clips_with_false_event':sum(bool(r['predicted_events']) for r in subset if r['true_type']=='none'),
            'by_type':{k:{'n':len(q),'type_recall':round(sum(r['type_detected'] for r in q)/len(q),3),
                          'localization_recall_iou_0_5':round(sum(r['localized_at_iou_0_5'] for r in q)/len(q),3)}
                       for k in ('pause','quiet','rush','drag') if (q:=[r for r in positive if r['true_type']==k])},
            'by_severity':{k:{'n':len(q),'localization_recall_iou_0_5':round(sum(r['localized_at_iou_0_5'] for r in q)/len(q),3)}
                           for k in ('minor','moderate','severe') if (q:=[r for r in positive if r['true_severity']==k])},
            'localization_recall_iou_0_3':round(sum(r['localized_at_iou_0_3'] for r in positive)/len(positive),4),
            'collar_200ms_recall':round(sum(r['localized_with_200ms_collar'] for r in positive)/len(positive),4),
            'collar_200ms_precision':round(sum(r['localized_with_200ms_collar'] for r in positive)/total_predictions,4) if total_predictions else 0
        }
    # Passage-family bootstrap respects the fact that 12 siblings share one
    # source passage. It quantifies variability within this eight-family test,
    # not generalization to a population of speakers.
    test=[r for r in rows if r['split']=='test']
    families=sorted({r['family_id'] for r in test});rng=np.random.default_rng(20261014)
    draws={'recall':[],'precision':[],'f1':[]}
    by_family={f:[r for r in test if r['family_id']==f] for f in families}
    for _ in range(2000):
        sampled=rng.choice(families,size=len(families),replace=True)
        q=[r for f in sampled for r in by_family[str(f)]]
        pos=[r for r in q if r['true_type']!='none']
        hits=sum(r['localized_at_iou_0_5'] for r in pos)
        preds=sum(len(r['predicted_events']) for r in q)
        rec=hits/max(1,len(pos));prec=hits/max(1,preds)
        draws['recall'].append(rec);draws['precision'].append(prec)
        draws['f1'].append(2*prec*rec/max(1e-12,prec+rec))
    report['splits']['test']['family_bootstrap_95_ci']={
        k:[round(float(v),4) for v in np.percentile(values,[2.5,97.5])]
        for k,values in draws.items()}
    report['family_bootstrap_note']='Percentile interval from 2,000 bootstrap draws resampling the eight held-out passage families; clips within each family remain together.'
    out=ROOT/'localizer';out.mkdir(exist_ok=True)
    dump(out/'evaluation.json',report)
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    (out/'predictions.js').write_text('const LOCALIZER_PREDICTIONS='+json.dumps({r['clip_id']:r for r in rows})+';\n',encoding='utf-8')
    print(json.dumps(report['splits'],indent=2))

if __name__=='__main__':main()
