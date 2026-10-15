"""Evaluate predicted event intervals on speaker-disjoint digital-edit splits."""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
from core import read, loadl, dump
from localizer import detect, DEFAULTS, gate_single_edit_events, _rubric
from infer_model import infer_pair, load_model

ROOT=Path(__file__).resolve().parents[1]

def overlap(a,b):
    left=max(a[0],b[0]);right=min(a[1],b[1])
    return max(0,right-left)/max(1e-9,max(a[1],b[1])-min(a[0],b[0]))

def main():
    threshold_file=ROOT/'localizer/thresholds.json'
    thresholds=json.loads(threshold_file.read_text(encoding='utf-8'))['thresholds'] if threshold_file.exists() else DEFAULTS
    classifier=load_model()
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
        signature=infer_pair(bx,ux,model=classifier,base_features=bf,variant_features=uf)
        truth=labels.get(clip)
        predicted=result['events']
        gated=gate_single_edit_events(predicted,signature['type']['predicted'])
        gated_scores=_rubric(gated,result.get('word_measurements',[]),True)
        same=[e for e in predicted if truth and e['type']==truth['flaw_type']]
        gated_same=[e for e in gated if truth and e['type']==truth['flaw_type']]
        true_interval=(truth['start_seconds'],truth['end_seconds']) if truth else None
        best=max(same,key=lambda e:overlap((e['start_seconds'],e['end_seconds']),true_interval),default=None) if truth else None
        iou=overlap((best['start_seconds'],best['end_seconds']),true_interval) if best else 0
        gated_best=max(gated_same,key=lambda e:overlap((e['start_seconds'],e['end_seconds']),true_interval),default=None) if truth else None
        gated_iou=overlap((gated_best['start_seconds'],gated_best['end_seconds']),true_interval) if gated_best else 0
        rows.append({'clip_id':clip,'split':m['split'],'true_type':m['flaw_type'],
                     'true_severity':m['severity'],'family_id':m['family_id'],'status':result['status'],
                     'predicted_events':predicted,'predicted_edit_type':signature['type']['predicted'],
                     'edit_type_confidence':max(signature['type']['probabilities'].values()),
                     'predicted_events_single_edit_gate':gated,'scores':result['scores'],
                     'scores_single_edit_type_gate':gated_scores,
                     'best_same_type_iou':round(iou,4),
                     'type_detected':bool(same),'localized_at_iou_0_5':iou>=.5,
                     'localized_at_iou_0_5_single_edit_gate':gated_iou>=.5,
                     'localized_at_iou_0_3':iou>=.3,
                     'localized_with_200ms_collar':bool(best and abs(best['start_seconds']-true_interval[0])<=.2 and abs(best['end_seconds']-true_interval[1])<=.2) if truth else False,
                     'onset_error_seconds':round(abs(best['start_seconds']-true_interval[0]),3) if best else None,
                     'offset_error_seconds':round(abs(best['end_seconds']-true_interval[1]),3) if best else None})
    report={'scope':'Digital edits only; no human speech-quality accuracy is implied.',
            'single_edit_gate_protocol':'The gate classifier is the released fit on train+validation. Only its test metrics are held out; its train and validation metrics below are in-sample diagnostics. The gate threshold and rule were selected with OOF train and validation only in reports/localizer_type_gate_experiment_20261015.json.',
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
        gated_hits=sum(r['localized_at_iou_0_5_single_edit_gate'] for r in positive)
        gated_predictions=sum(len(r['predicted_events_single_edit_gate']) for r in subset)
        gated_recall=gated_hits/max(1,len(positive))
        gated_precision=gated_hits/max(1,gated_predictions)
        report['splits'][split]['single_edit_type_gate']={
            'classifier_type_accuracy':round(sum(r['predicted_edit_type']==r['true_type'] for r in subset)/max(1,len(subset)),4),
            'localization_recall_iou_0_5':round(gated_recall,4),
            'event_precision_iou_0_5':round(gated_precision,4),
            'event_f1_iou_0_5':round(2*gated_precision*gated_recall/max(1e-12,gated_precision+gated_recall),4),
            'predicted_events':gated_predictions,
            'condition':'Use only when the benchmark guarantees one edited operation per clip; not suitable for natural speech with multiple concurrent issues.'}
    # Passage-family bootstrap respects the fact that 12 siblings share one
    # source passage. It quantifies variability within this eight-family test,
    # not generalization to a population of speakers.
    test=[r for r in rows if r['split']=='test']
    families=sorted({r['family_id'] for r in test});rng=np.random.default_rng(20261014)
    draws={'recall':[],'precision':[],'f1':[]}
    gated_draws={'recall':[],'precision':[],'f1':[]}
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
        gated_hits=sum(r['localized_at_iou_0_5_single_edit_gate'] for r in pos)
        gated_preds=sum(len(r['predicted_events_single_edit_gate']) for r in q)
        gated_rec=gated_hits/max(1,len(pos));gated_prec=gated_hits/max(1,gated_preds)
        gated_draws['recall'].append(gated_rec);gated_draws['precision'].append(gated_prec)
        gated_draws['f1'].append(2*gated_prec*gated_rec/max(1e-12,gated_prec+gated_rec))
    report['splits']['test']['family_bootstrap_95_ci']={
        k:[round(float(v),4) for v in np.percentile(values,[2.5,97.5])]
        for k,values in draws.items()}
    report['splits']['test']['single_edit_type_gate']['family_bootstrap_95_ci']={
        k:[round(float(v),4) for v in np.percentile(values,[2.5,97.5])]
        for k,values in gated_draws.items()}
    report['family_bootstrap_note']='Percentile interval from 2,000 bootstrap draws resampling the eight held-out passage families; clips within each family remain together.'
    out=ROOT/'localizer';out.mkdir(exist_ok=True)
    dump(out/'evaluation.json',report)
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    (out/'predictions.js').write_text('const LOCALIZER_PREDICTIONS='+json.dumps({r['clip_id']:r for r in rows})+';\n',encoding='utf-8')
    print(json.dumps(report['splits'],indent=2))

if __name__=='__main__':main()
