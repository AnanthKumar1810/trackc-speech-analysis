"""Train an edit-signature classifier; tune on Kennedy only, evaluate Obama once.
Requires scikit-learn for fitting. Export JSON weights for numpy-only inference.
"""
from pathlib import Path
import json,csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from core import read,dump,loadl
from model_features import vector,NAMES
ROOT=Path(__file__).resolve().parents[1]
TYPE=['none','pause','quiet','rush','drag'];STRENGTH=['none','minor','moderate','severe']
def main():
 manifest=loadl(ROOT/'metadata/manifest.jsonl');groups={m['clip_id']:m for m in manifest};records=[];cache={}
 for m in manifest:
  base=groups[m['family_id']+'_base']
  def load(m):
   if m['clip_id'] not in cache:cache[m['clip_id']]=(read(ROOT/m['audio_path']),dict(np.load(ROOT/m['feature_path'])))
   return cache[m['clip_id']]
  a,af=load(base);b,bf=load(m);v=vector(a,b,af,bf);records.append({'clip_id':m['clip_id'],'family_id':m['family_id'],'speaker_id':m['speaker_id'],'split':m['split'],'edit_type':m['flaw_type'],'edit_strength':m['severity'],'x':v})
 X=np.stack([r['x'] for r in records]);typ=np.array([TYPE.index(r['edit_type']) for r in records]);sev=np.array([STRENGTH.index(r['edit_strength']) for r in records]);split=np.array([r['split'] for r in records]);tr=split=='train';va=split=='validation';te=split=='test';assert len(set(r['speaker_id'] for r in records if r['split']=='train')&set(r['speaker_id'] for r in records if r['split']=='test'))==0
 center=np.median(X[tr],axis=0);q1=np.percentile(X[tr],25,axis=0);q3=np.percentile(X[tr],75,axis=0);scale=np.where(q3-q1>1e-8,q3-q1,1.);Z=np.clip((X-center)/scale,-20,20)
 models={};preds={};tuning={}
 for target,y,classes in [('type',typ,TYPE),('strength',sev,STRENGTH)]:
  candidates=[]
  for C in [.01,.1,1.,10.]:
   model=LogisticRegression(C=C,max_iter=3000,class_weight='balanced',random_state=0)
   model.fit(Z[tr],y[tr]);p=model.predict(Z[va]);candidates.append({'C':C,'validation_macro_f1':float(f1_score(y[va],p,average='macro',labels=list(range(len(classes))),zero_division=0)),'validation_accuracy':float(accuracy_score(y[va],p)),'model':model})
  best=max(candidates,key=lambda c:(c['validation_macro_f1'],c['validation_accuracy'],-c['C']));model=best['model'];tuning[target]=[{k:v for k,v in c.items() if k!='model'} for c in candidates]
  models[target]={'classes':classes,'coef':model.coef_.tolist(),'intercept':model.intercept_.tolist(),'selected_C':best['C']}
  preds[target]=model.predict(Z)
 modeldata={'model_kind':'paired edit signature classifier','not_a_human_quality_score':True,'trained_split':'train','tuned_split':'validation','test_split':'test','feature_names':NAMES,'center_train_only':center.tolist(),'scale_train_only':scale.tolist(),'clip_standardized_features_to':[-20,20],'heads':models,'training_families':sorted(set(r['family_id'] for r in records if r['split']=='train')),'training_speakers':sorted(set(r['speaker_id'] for r in records if r['split']=='train')),'classes_definition':'Predicted values identify digitally applied operation and programmed magnitude. They are not listener judgments.','runtime':'numpy only for inference; scikit-learn used for fitting'}
 dump(ROOT/'model/model.json',modeldata)
 out={'dataset_counts':{s:int(sum(split==s)) for s in ['train','validation','test']},'families':{s:len(set(r['family_id'] for r in records if r['split']==s)) for s in ['train','validation','test']},'training_policy':'Only Reagan/train fitted scalers and weights. Kennedy/validation selected C separately for each head. Obama/test measured once after both decisions. Same family never crosses splits.','tuning':tuning,'metrics':{},'limitations':['Test is one speaker/recording style; the result is not a generalization estimate across speakers.','All labels are known digital operations; no natural flaws or human delivery ratings are represented.','Self-pair none class can be recognized by exact audio identity.','The model requires a suitable matching reference; performance on independently recorded participant speech is unmeasured.']}
 for s,mask in [('train',tr),('validation',va),('test',te)]:
  out['metrics'][s]={}
  for head,y,classes in [('type',typ,TYPE),('strength',sev,STRENGTH)]:
   p=preds[head][mask];truth=y[mask];out['metrics'][s][head]={'accuracy':float(accuracy_score(truth,p)),'macro_f1':float(f1_score(truth,p,average='macro',labels=list(range(len(classes))),zero_division=0)),'confusion_matrix_rows_true_cols_pred':confusion_matrix(truth,p,labels=list(range(len(classes)))).tolist(),'class_order':classes}
 predictions=[{'clip_id':r['clip_id'],'family_id':r['family_id'],'split':r['split'],'true_edit_type':r['edit_type'],'predicted_edit_type':TYPE[preds['type'][i]],'true_edit_strength':r['edit_strength'],'predicted_edit_strength':STRENGTH[preds['strength'][i]]} for i,r in enumerate(records)]
 with (ROOT/'model/predictions.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(predictions[0]));w.writeheader();w.writerows(predictions)
 dump(ROOT/'model/evaluation.json',out);print(json.dumps({'selected':{h:models[h]['selected_C'] for h in models},'test':out['metrics']['test']},indent=2))
if __name__=='__main__':main()
