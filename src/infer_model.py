"""NumPy-only inference for paired 16 kHz audio and precomputed features."""
from pathlib import Path
import json
import numpy as np
from core import features
from model_features import vector
ROOT=Path(__file__).resolve().parents[1]
def load_model(path=None):return json.loads(Path(path or ROOT/'model/model.json').read_text(encoding='utf-8'))
def infer_pair(base_pcm,variant_pcm,model=None,base_features=None,variant_features=None):
 model=model or load_model();x=vector(base_pcm,variant_pcm,base_features if base_features is not None else features(base_pcm),variant_features if variant_features is not None else features(variant_pcm));out={}
 for name,head in model['heads'].items():
  if head.get('estimator_type')=='extra_trees':
   p=np.zeros(len(head['classes']),dtype=float)
   raw=x
   for tree in head['trees']:
    node=0
    while tree['children_left'][node] != -1:
     feature=tree['feature'][node]
     node=tree['children_left'][node] if raw[feature] <= tree['threshold'][node] else tree['children_right'][node]
    counts=np.asarray(tree['values'][node],dtype=float);p+=counts/max(counts.sum(),1e-12)
   p/=max(1,len(head['trees']))
  else:
   center=model.get('center_fit_data',model.get('center_train_only'))
   scale=model.get('scale_fit_data',model.get('scale_train_only'))
   z=np.clip((x-np.asarray(center))/np.asarray(scale),*model['clip_standardized_features_to'])
   scores=np.asarray(head['coef'])@z+np.asarray(head['intercept']);scores-=np.max(scores);p=np.exp(scores);p/=np.sum(p)
  out[name]={'predicted':head['classes'][int(np.argmax(p))],'probabilities':dict(zip(head['classes'],[float(v) for v in p]))}
 out['interpretation']='Digital edit signature only. No human speaking-quality rating. Uncalibrated model probabilities.';return out
