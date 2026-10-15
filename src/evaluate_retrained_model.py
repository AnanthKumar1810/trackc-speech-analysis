"""Final evaluation of the frozen retrained classifier on the held-out speaker."""
from pathlib import Path
import csv,json
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from core import loadl,read,dump
from infer_model import infer_pair,load_model
ROOT=Path(__file__).resolve().parents[1]
TYPE=['none','pause','quiet','rush','drag'];STRENGTH=['none','minor','moderate','severe']

def main():
    model=load_model();manifest=loadl(ROOT/'metadata/manifest.jsonl');lookup={m['clip_id']:m for m in manifest};cache={};rows=[]
    def load(m):
        if m['clip_id'] not in cache:
            cache[m['clip_id']]=(read(ROOT/m['audio_path']),dict(np.load(ROOT/m['feature_path'])))
        return cache[m['clip_id']]
    for m in manifest:
        base=lookup[m['family_id']+'_base'];a,af=load(base);b,bf=load(m)
        pred=infer_pair(a,b,model=model,base_features=af,variant_features=bf)
        rows.append({'clip_id':m['clip_id'],'family_id':m['family_id'],'split':m['split'],
          'true_edit_type':m['flaw_type'],'predicted_edit_type':pred['type']['predicted'],
          'true_edit_strength':m['severity'],'predicted_edit_strength':pred['strength']['predicted']})
    test_rows=[r for r in rows if r['split']=='test']
    baseline=json.loads((ROOT/'reports/classifier_baseline_logistic_20261015.json').read_text(encoding='utf-8'))
    report={'scope':'Held-out Obama synthetic edits only; no natural-speech quality accuracy is implied.',
      'test_used_for_selection':False,'trained_splits':model['trained_splits'],
      'selected_models':{k:v['selected_candidate'] for k,v in model['heads'].items()},
      'test_clips':len(test_rows),'test_families':len(set(r['family_id'] for r in test_rows)),
      'comparison_to_prior_logistic_model':{},'metrics':{}}
    for name,truth,pred,classes in [('type','true_edit_type','predicted_edit_type',TYPE),('strength','true_edit_strength','predicted_edit_strength',STRENGTH)]:
        y=[classes.index(r[truth]) for r in test_rows];p=[classes.index(r[pred]) for r in test_rows]
        report['metrics'][name]={'accuracy':float(accuracy_score(y,p)),
          'macro_f1':float(f1_score(y,p,average='macro',labels=range(len(classes)),zero_division=0)),
          'confusion_matrix_rows_true_cols_pred':confusion_matrix(y,p,labels=range(len(classes))).tolist(),
          'class_order':classes}
        report['comparison_to_prior_logistic_model'][name]={
          'prior_test_accuracy':baseline['metrics']['test'][name]['accuracy'],
          'prior_test_macro_f1':baseline['metrics']['test'][name]['macro_f1'],
          'accuracy_delta':report['metrics'][name]['accuracy']-baseline['metrics']['test'][name]['accuracy'],
          'macro_f1_delta':report['metrics'][name]['macro_f1']-baseline['metrics']['test'][name]['macro_f1']}
    with (ROOT/'model/predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump(ROOT/'model/evaluation.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
