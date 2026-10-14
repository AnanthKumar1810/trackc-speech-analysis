"""NumPy-only inference for paired 16 kHz audio and precomputed features."""
from pathlib import Path
import json
import numpy as np
from core import features
from model_features import vector
ROOT=Path(__file__).resolve().parents[1]
def load_model(path=None):return json.loads(Path(path or ROOT/'model/model.json').read_text(encoding='utf-8'))
def infer_pair(base_pcm,variant_pcm,model=None,base_features=None,variant_features=None):
 model=model or load_model();x=vector(base_pcm,variant_pcm,base_features if base_features is not None else features(base_pcm),variant_features if variant_features is not None else features(variant_pcm));z=np.clip((x-np.asarray(model['center_train_only']))/np.asarray(model['scale_train_only']),*model['clip_standardized_features_to']);out={}
 for name,head in model['heads'].items():
  scores=np.asarray(head['coef'])@z+np.asarray(head['intercept']);scores-=np.max(scores);p=np.exp(scores);p/=np.sum(p);out[name]={'predicted':head['classes'][int(np.argmax(p))],'probabilities':dict(zip(head['classes'],[float(v) for v in p]))}
 out['interpretation']='Digital edit signature only. No human speaking-quality rating. Uncalibrated model probabilities.';return out
