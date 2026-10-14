"""Pair-level acoustic features. Uses WAVs and frame arrays, never flaw labels or edit spans."""
from pathlib import Path
import numpy as np
from core import read, features
NAMES=['duration_ratio','duration_delta_s','sample_zero_fraction_delta','longest_zero_run_delta_s','rms_median_delta_db','rms_p10_delta_db','rms_p90_delta_db','relative_rms_shape_mae','relative_rms_shape_p90','relative_rms_shape_signed_p10','relative_rms_shape_signed_p90','spectral_centroid_mean_delta_hz','zcr_mean_delta','voiced_fraction_delta','f0_median_log2_ratio','f0_iqr_delta_hz','mfcc_mean_l2','mfcc_std_l2','loud_frame_fraction_delta','quiet_frame_fraction_delta']
def _longest_zero(x):
 v=(x==0).astype('int8');q=np.flatnonzero(np.diff(np.r_[0,v,0]));return int(np.max(q[1::2]-q[::2])) if len(q) else 0
def _iqr(x):return np.percentile(x,75)-np.percentile(x,25) if len(x) else 0.0
def _shape_delta(a,b):
 grid=np.linspace(0,1,128);u=np.interp(grid,np.linspace(0,1,len(a)),a);v=np.interp(grid,np.linspace(0,1,len(b)),b);return v-u
def vector(base_pcm,variant_pcm,base_features,variant_features):
 a,b=base_pcm,variant_pcm;af,bf=base_features,variant_features;ra=np.asarray(af['rms_dbfs']);rb=np.asarray(bf['rms_dbfs']);ar=np.asarray(af['relative_rms_db']);br=np.asarray(bf['relative_rms_db']);d=_shape_delta(ar,br)
 fa=np.asarray(af['f0_hz']);fb=np.asarray(bf['f0_hz']);fa=fa[np.isfinite(fa)];fb=fb[np.isfinite(fb)]
 ma=np.asarray(af['mfcc']);mb=np.asarray(bf['mfcc'])
 vals=[len(b)/len(a),(len(b)-len(a))/16000,np.mean(b==0)-np.mean(a==0),(_longest_zero(b)-_longest_zero(a))/16000,np.median(rb)-np.median(ra),np.percentile(rb,10)-np.percentile(ra,10),np.percentile(rb,90)-np.percentile(ra,90),np.mean(np.abs(d)),np.percentile(np.abs(d),90),np.percentile(d,10),np.percentile(d,90),np.mean(bf['spectral_centroid_hz'])-np.mean(af['spectral_centroid_hz']),np.mean(bf['zcr'])-np.mean(af['zcr']),len(fb)/len(bf['f0_hz'])-len(fa)/len(af['f0_hz']),np.log2(max(np.median(fb) if len(fb) else 1,1)/max(np.median(fa) if len(fa) else 1,1)),_iqr(fb)-_iqr(fa),np.linalg.norm(np.mean(mb,axis=0)-np.mean(ma,axis=0)),np.linalg.norm(np.std(mb,axis=0)-np.std(ma,axis=0)),np.mean(br>3)-np.mean(ar>3),np.mean(br< -12)-np.mean(ar< -12)]
 x=np.asarray(vals,dtype='float64');assert len(x)==len(NAMES) and np.isfinite(x).all();return x
