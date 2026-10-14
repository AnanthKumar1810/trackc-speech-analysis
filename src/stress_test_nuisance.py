"""Measure false alerts under global, label-preserving audio changes.

This is a small engineering negative-control suite, not evidence of
cross-speaker accuracy. It tests base clips only and never tunes thresholds.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt
from core import SR, read, features, loadl, dump
from localizer import detect

ROOT=Path(__file__).resolve().parents[1]

def pcm(x):
    return np.clip(np.rint(np.asarray(x)*32768),-32768,32767).astype(np.int16)

def pitch_shift(x,semitones):
    ratio=2**(semitones/12)
    y=resample_poly(x,1000,max(1,round(1000*ratio)))
    y=__import__('core').time_stretch(y.astype(float),1/ratio)
    if len(y)<len(x):y=np.pad(y,(0,len(x)-len(y)))
    return y[:len(x)]

def scale_words(words,rate):
    return [dict(w,start_sample=round(w['start_sample']/rate),
                 end_sample=round(w['end_sample']/rate)) for w in words]

def controls(x,seed):
    peak=max(float(np.max(np.abs(x))),1e-6)
    # Give the +/-6 dB controls headroom while applying identical scaling to
    # both sides of every comparison.
    ref=x*(.30/peak)
    rng=np.random.default_rng(seed)
    noise=rng.normal(0,.30/(10**(25/20)),len(ref))
    return {
      'speed_minus_10pct':(ref,__import__('core').time_stretch(ref,.9),1.),
      'speed_plus_10pct':(ref,__import__('core').time_stretch(ref,1.1),1.),
      'gain_minus_6db':(ref,ref*10**(-6/20),1.),
      'gain_plus_6db':(ref,ref*10**(6/20),1.),
      'pitch_minus_3st':(ref,pitch_shift(ref,-3),1.),
      'pitch_plus_3st':(ref,pitch_shift(ref,3),1.),
      'stationary_noise_25db_snr':(ref,ref+noise,1.),
      'phone_band_300_3400hz':(ref,sosfiltfilt(butter(4,[300,3400],btype='bandpass',fs=SR,output='sos'),ref),1.),
    }

def main():
    manifests=[m for m in loadl(ROOT/'metadata/manifest.jsonl') if m['flaw_type']=='none']
    alignments={}
    for w in loadl(ROOT/'alignment_mfa/word_alignment_mfa.jsonl'):
        alignments.setdefault(w['clip_id'],[]).append(w)
    thresholds=json.loads((ROOT/'localizer/thresholds.json').read_text(encoding='utf-8'))['thresholds']
    results=[]
    for m in manifests:
        base=np.asarray(read(ROOT/m['audio_path']),dtype=float)/32768
        words=alignments[m['clip_id']]
        seed=int.from_bytes(hashlib.sha256(m['clip_id'].encode()).digest()[:4],'big')
        base_reference=pcm(base*(.30/max(float(np.max(np.abs(base))),1e-6)))
        base_features=features(base_reference)
        for name,(reference,take,rate) in controls(base,seed).items():
            if name.startswith('speed_'):
                speed=.9 if name.endswith('10pct') and name.startswith('speed_minus') else 1.1
                take_words=scale_words(words,speed)
            else:take_words=words
            a=pcm(reference);b=pcm(take)
            result=detect(a,b,words,take_words,thresholds=thresholds,
                           reference_features=base_features,take_features=features(b))
            results.append({'clip_id':m['clip_id'],'split':m['split'],'control':name,
                            'false_event_count':len(result['events']),
                            'event_types':[e['type'] for e in result['events']]})
    summary={}
    for name in sorted({r['control'] for r in results}):
        subset=[r for r in results if r['control']==name]
        summary[name]={'clips':len(subset),'clips_with_any_false_event':sum(r['false_event_count']>0 for r in subset),
                       'false_events':sum(r['false_event_count'] for r in subset),
                       'mean_false_events_per_clip':round(float(np.mean([r['false_event_count'] for r in subset])),3)}
    report={'scope':'Global nuisance negative controls on unedited archive bases; not cross-speaker validation.',
            'method':'Every source clip is compared with a globally transformed version of itself. No threshold is tuned on these results.',
            'limits':['Phone-band filtering is used as a codec/channel proxy; MP3 encoding was not tested.',
                      'Local flaw preservation under nuisance transformations is not measured in this report.',
                      'Synthetic global pitch changes do not establish robustness to a new speaker.'],
            'summary':summary,'per_clip':results}
    dump(ROOT/'localizer/nuisance_controls.json',report)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
