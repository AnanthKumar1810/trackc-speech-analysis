"""Bounded MFCC-DTW transfer of reference word boundaries to a same-text take.

This is a fallback estimate, never transcript verification or phoneme alignment.
"""
import numpy as np
from core import SR
from dtw_pilot import path_for

def align_from_reference(reference_features,take_features,reference_words,take_samples):
    if not reference_words:return None,{'status':'missing_reference_words'}
    a=np.asarray(reference_features['mfcc']);b=np.asarray(take_features['mfcc'])
    if len(a)>2500 or len(b)>2500 or not .5<=len(b)/len(a)<=2:
        return None,{'status':'duration_out_of_range'}
    path,cost=path_for(a,b)
    if not np.isfinite(cost) or cost>.8:
        return None,{'status':'poor_acoustic_match','normalized_mfcc_cost':round(cost,3)}
    at=np.asarray(reference_features['time_seconds']);bt=np.asarray(take_features['time_seconds'])
    unique,starts=np.unique(path[:,0],return_index=True)
    ends=np.r_[starts[1:],len(path)]
    mapped=np.array([np.median(bt[path[s:e,1]]) for s,e in zip(starts,ends)])
    def point(sample):return int(np.clip(round(np.interp(sample/SR,at[unique],mapped)*SR),0,take_samples))
    result=[]
    for w in reference_words:
        s=point(w['start_sample']);e=point(w['end_sample'])
        if e-s<320:return None,{'status':'collapsed_word','normalized_mfcc_cost':round(cost,3)}
        result.append({'word':w['word'],'start_sample':s,'end_sample':e})
    if any(result[i]['end_sample']>result[i+1]['start_sample']+1600 for i in range(len(result)-1)):
        return None,{'status':'nonmonotonic_mapping','normalized_mfcc_cost':round(cost,3)}
    return result,{'status':'automatic_mfcc_dtw_estimate','normalized_mfcc_cost':round(cost,3),
                   'note':'Same transcript must be confirmed separately; these are uncertain mapped word boundaries.'}
