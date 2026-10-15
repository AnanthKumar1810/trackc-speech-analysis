"""Fit the grouped-CV/validation-selected paired edit classifier.

Candidate choice comes from classifier_experiment_20261015.json. The final
estimators are refit on train+validation only; the held-out test is not read.
After this file finishes, run evaluate_retrained_model.py once for the frozen
test result.
"""
from pathlib import Path
import json
import re
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from core import loadl, read, dump
from model_features import vector, NAMES

ROOT=Path(__file__).resolve().parents[1]
CLASSES={'type':['none','pause','quiet','rush','drag'],
         'strength':['none','minor','moderate','severe']}

def estimator_from_name(name):
    match=re.fullmatch(r'extra_trees_leaf(\d+)_features(sqrt|1\.0)',name)
    if not match:
        raise ValueError(f'Unsupported selected model: {name}')
    leaf=int(match.group(1)); features=match.group(2)
    return ExtraTreesClassifier(n_estimators=400,min_samples_leaf=leaf,
        max_features='sqrt' if features=='sqrt' else 1.0,
        class_weight='balanced',random_state=17,n_jobs=1)

def serialize_forest(model):
    trees=[]
    for est in model.estimators_:
        tree=est.tree_
        trees.append({'children_left':tree.children_left.astype(int).tolist(),
                      'children_right':tree.children_right.astype(int).tolist(),
                      'feature':tree.feature.astype(int).tolist(),
                      'threshold':tree.threshold.astype(float).tolist(),
                      'values':tree.value[:,0,:].astype(float).tolist()})
    return trees

def main():
    manifest=loadl(ROOT/'metadata/manifest.jsonl')
    usable=[m for m in manifest if m['split'] in {'train','validation'}]
    lookup={m['clip_id']:m for m in usable};cache={};rows=[]
    def load(m):
        if m['clip_id'] not in cache:
            cache[m['clip_id']]=(read(ROOT/m['audio_path']),dict(np.load(ROOT/m['feature_path'])))
        return cache[m['clip_id']]
    for m in usable:
        base=lookup[m['family_id']+'_base'];a,af=load(base);b,bf=load(m)
        rows.append({'clip_id':m['clip_id'],'family_id':m['family_id'],
                     'speaker_id':m['speaker_id'],'split':m['split'],
                     'edit_type':m['flaw_type'],'edit_strength':m['severity'],
                     'x':vector(a,b,af,bf)})
    X=np.stack([r['x'] for r in rows])
    experiment=json.loads((ROOT/'model/classifier_experiment_20261015.json').read_text(encoding='utf-8'))
    center=np.median(X,axis=0);q1=np.percentile(X,25,axis=0);q3=np.percentile(X,75,axis=0)
    scale=np.where(q3-q1>1e-8,q3-q1,1.)
    model={'model_kind':'paired digital-edit signature classifier',
      'not_a_human_quality_score':True,'estimator_type':'extra_trees',
      'trained_splits':['train','validation'],'candidate_selection':'5-fold family-grouped CV on train; validation speaker used as a non-regression guard',
      'test_split':'test','feature_names':NAMES,'classes_definition':'Known digitally applied operation and programmed strength only; not listener judgments.',
      'center_fit_data':center.tolist(),'scale_fit_data':scale.tolist(),
      'clip_standardized_features_to':[-20,20],
      'note':'Trees use raw feature vectors; center and scale are stored for diagnostics only.',
      'heads':{},'training_families':sorted(set(r['family_id'] for r in rows)),
      'training_speakers':sorted(set(r['speaker_id'] for r in rows)),
      'training_clip_count':len(rows),'runtime':'NumPy-only inference; scikit-learn used for fitting'}
    for head in ['type','strength']:
        selected=experiment['selected_models'][head]
        clf=estimator_from_name(selected)
        y=np.array([CLASSES[head].index(r['edit_type'] if head=='type' else r['edit_strength']) for r in rows])
        clf.fit(X,y)
        model['heads'][head]={'estimator_type':'extra_trees','classes':CLASSES[head],
                              'selected_candidate':selected,'trees':serialize_forest(clf)}
    dump(ROOT/'model/model.json',model)
    print(json.dumps({'status':'fitted; test split was not loaded',
      'trained_clips':len(rows),'trained_families':len(model['training_families']),
      'selected_models':experiment['selected_models'],
      'serialized_tree_count':{h:len(v['trees']) for h,v in model['heads'].items()},
      'model_bytes':(ROOT/'model/model.json').stat().st_size},indent=2))

if __name__=='__main__':main()
