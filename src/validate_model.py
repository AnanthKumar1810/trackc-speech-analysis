"""Validate frozen model against all released audio and grouped holdout labels."""
from pathlib import Path
import json,csv,collections
import numpy as np
from core import read,loadl,dump
from infer_model import infer_pair,load_model
from model_features import NAMES
ROOT=Path(__file__).resolve().parents[1]
def main():
 m=load_model();items=loadl(ROOT/'metadata/manifest.jsonl');known={x['clip_id']:x for x in items};by_split=collections.defaultdict(set)
 for x in items:by_split[x['split']].add(x['speaker_id'])
 assert all(not by_split[a]&by_split[b] for a,b in [('train','validation'),('train','test'),('validation','test')]);assert m['feature_names']==NAMES;assert m['training_speakers']==sorted(by_split['train']);assert len(m['center_train_only'])==len(NAMES)==len(m['scale_train_only'])
 with (ROOT/'model/predictions.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
 assert len(rows)==len(items) and {r['clip_id'] for r in rows}==set(known)
 evals=json.loads((ROOT/'model/evaluation.json').read_text(encoding='utf-8'));matrix={}
 cache={}
 def signal(x):
  if x['clip_id'] not in cache:cache[x['clip_id']]=(read(ROOT/x['audio_path']),dict(np.load(ROOT/x['feature_path'])))
  return cache[x['clip_id']]
 for row in rows:
  x=known[row['clip_id']];b=known[x['family_id']+'_base'];ap,af=signal(b);bp,bf=signal(x);p=infer_pair(ap,bp,model=m,base_features=af,variant_features=bf)
  assert row['predicted_edit_type']==p['type']['predicted'] and row['predicted_edit_strength']==p['strength']['predicted']
  assert row['true_edit_type']==x['flaw_type'] and row['true_edit_strength']==x['severity'] and row['split']==x['split']
  assert all(abs(sum(head['probabilities'].values())-1)<1e-10 for head in [p['type'],p['strength']])
 for s in ['train','validation','test']:
  q=[r for r in rows if r['split']==s]
  for head,truth,pred in [('type','true_edit_type','predicted_edit_type'),('strength','true_edit_strength','predicted_edit_strength')]:
   expected=evals['metrics'][s][head];accuracy=sum(r[truth]==r[pred] for r in q)/len(q);assert abs(accuracy-expected['accuracy'])<1e-12
   classes=expected['class_order'];conf=[[sum(r[truth]==a and r[pred]==b for r in q) for b in classes] for a in classes];assert conf==expected['confusion_matrix_rows_true_cols_pred']
 report={'status':'PASS','clips_reinferred':len(rows),'speaker_disjoint_splits':True,'training_speakers':m['training_speakers'],'model_predictions_match_frozen_csv':True,'accuracy_and_confusions_match_report':True,'model_scope':'Recognizes known digital edit signatures and programmed strengths; natural speaking quality unmeasured'};dump(ROOT/'model/validation.json',report);print(report)
if __name__=='__main__':main()
