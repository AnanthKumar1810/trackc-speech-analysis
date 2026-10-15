"""Reference-relative, word-grounded candidate flaw detector.

Only paired audio, estimated word timings and explicit recording setup enter
inference. Dataset edit labels and filenames are never inputs to this module.
"""
import numpy as np
from core import SR, features

VERSION = 'word-localizer-1.1'
DEFAULTS = {'pause_excess_s': .18, 'quiet_db': -4.0,
            'rush_ratio': .82, 'drag_ratio': 1.22, 'merge_gap_s': .12}


def _word_level(f, start, end):
    times = f['time_seconds']
    mask = (times >= start) & (times <= end)
    if not np.any(mask):
        mask[np.argmin(abs(times-(start+end)/2))] = True
    db = np.asarray(f['rms_dbfs'])[mask]
    return float(10*np.log10(np.mean(10**(db/10)) + 1e-12))


def _word_pitch_spread(f, start, end):
    """Robust F0 range in semitones relative to the clip's own median F0."""
    times=np.asarray(f['time_seconds'])
    pitch=np.asarray(f['relative_pitch_semitones'])
    mask=(times>=start)&(times<=end)&np.isfinite(pitch)
    values=pitch[mask]
    if len(values)<3:return None
    return float(np.percentile(values,90)-np.percentile(values,10))


def detect(reference, take, reference_words, take_words, *,
           same_setup=False, thresholds=None, reference_features=None,
           take_features=None):
    """Return predicted events on the take timeline, or abstain on bad input.

    Word intervals may be MFA or another forced aligner; they are estimates.
    Transcript matching must happen upstream. A reference from a different
    recording setup disables absolute level judgments.
    """
    t = dict(DEFAULTS, **(thresholds or {}))
    if not reference_words or len(reference_words) != len(take_words):
        return {'status': 'alignment_mismatch', 'events': [], 'scores': None}
    if [w['word'].lower() for w in reference_words] != [w['word'].lower() for w in take_words]:
        return {'status': 'transcript_mismatch', 'events': [], 'scores': None}
    if min(len(reference), len(take)) < SR:
        return {'status': 'insufficient_audio', 'events': [], 'scores': None}
    bf = reference_features if reference_features is not None else features(reference)
    uf = take_features if take_features is not None else features(take)
    raw_ratios=[]
    for a,b in zip(reference_words,take_words):
        ad=(a['end_sample']-a['start_sample'])/SR
        bd=(b['end_sample']-b['start_sample'])/SR
        if ad>=.025 and bd>=.025 and bd<=3.0:raw_ratios.append(bd/ad)
    # Normalize local pacing to the pair's robust overall tempo. This removes
    # speaker and global playback-rate differences while preserving local
    # within-utterance deviations. The raw global ratio remains reportable.
    global_tempo=float(np.median(raw_ratios)) if raw_ratios else 1.
    global_level_shift=float(np.median(uf['rms_dbfs'])-np.median(bf['rms_dbfs']))
    events = []
    rows = []
    for i, (a,b) in enumerate(zip(reference_words,take_words)):
        ad = (a['end_sample']-a['start_sample'])/SR
        bd = (b['end_sample']-b['start_sample'])/SR
        if ad < .025 or bd < .025 or bd > 3.0:
            rows.append({'index': i, 'word': b['word'], 'usable': False})
            continue
        raw_ratio = bd/ad
        ratio = raw_ratio/max(global_tempo,1e-6)
        adb = _word_level(bf,a['start_sample']/SR,a['end_sample']/SR)
        bdb = _word_level(uf,b['start_sample']/SR,b['end_sample']/SR)
        delta=(bdb-adb)-global_level_shift
        ref_pitch=_word_pitch_spread(bf,a['start_sample']/SR,a['end_sample']/SR)
        take_pitch=_word_pitch_spread(uf,b['start_sample']/SR,b['end_sample']/SR)
        rows.append({'index':i,'word':b['word'],'usable':True,
                     'duration_ratio':round(raw_ratio,3),
                     'tempo_normalized_duration_ratio':round(ratio,3),
                     'global_tempo_ratio':round(global_tempo,3),
                     'level_delta_db':round(delta,2),
                     'global_level_shift_db':round(global_level_shift,2),
                     'reference_pitch_spread_semitones':round(ref_pitch,2) if ref_pitch is not None else None,
                     'take_pitch_spread_semitones':round(take_pitch,2) if take_pitch is not None else None})
        start,end=b['start_sample']/SR,b['end_sample']/SR
        if ratio < t['rush_ratio']:
            events.append(_event('rush',start,end,i,ratio,1-ratio,
                                 f'Word duration is {ratio:.2f} times the reference duration.'))
        elif ratio > t['drag_ratio']:
            events.append(_event('drag',start,end,i,ratio,ratio-1,
                                 f'Word duration is {ratio:.2f} times the reference duration.'))
        if delta < t['quiet_db']:
            events.append(_event('quiet',start,end,i,delta,-delta,
                                 f'Word energy is {delta:.1f} dB below its clip-relative reference after removing global recording gain.'))
        if i+1 < len(reference_words):
            ag=(reference_words[i+1]['start_sample']-a['end_sample'])/SR
            bg=(take_words[i+1]['start_sample']-b['end_sample'])/SR
            # Compare pauses at the pair's normalized tempo. A uniform speed
            # change stretches gaps as well as words and should not create a
            # local pause alert by itself.
            normalized_bg=bg/max(global_tempo,1e-6)
            excess=normalized_bg-max(0,ag)
            if excess >= t['pause_excess_s'] and bg > .08:
                events.append(_event('pause',end,take_words[i+1]['start_sample']/SR,
                    i,round(excess,3),excess,
                    f'Normalized gap {normalized_bg:.2f} s − reference gap {max(0,ag):.2f} s = {excess:.2f} s excess.'))
    merged = _merge(events, max_gap=t['merge_gap_s'])
    return {'status':'candidate_events','version':VERSION,'events':merged,
            'scores':_rubric(merged,rows,same_setup),'word_measurements':rows,
            'normalization':{'global_tempo_ratio':round(global_tempo,3),
                             'global_level_shift_db':round(global_level_shift,2),
                             'note':'Local pacing and level candidates are normalized within each clip. A uniform whole-clip rate or level change is reported here but not localized as a word flaw.'},
            'caution':'Reference-relative acoustic differences are candidates, not human quality ratings. F0 spread is descriptive and is not yet used to issue a monotony label. Word boundaries are automatic estimates; pause placement may be rhetorically appropriate.'}


def _event(kind,start,end,index,value,magnitude,explanation):
    if kind == 'pause': level='minor' if magnitude<.45 else 'moderate' if magnitude<1.0 else 'severe'
    elif kind == 'quiet': level='minor' if magnitude<7 else 'moderate' if magnitude<12 else 'severe'
    else: level='minor' if magnitude<.30 else 'moderate' if magnitude<.55 else 'severe'
    return {'type':kind,'start_seconds':round(start,3),'end_seconds':round(end,3),
            'word_index':index,'measurement':round(float(value),3),
            'severity':level,'explanation':explanation}


def _merge(events, max_gap=.12):
    merged=[]
    for e in sorted(events,key=lambda x:(x['type'],x['start_seconds'])):
        if merged and merged[-1]['type']==e['type'] and e['start_seconds']-merged[-1]['end_seconds']<=max_gap:
            last=merged[-1]
            last['end_seconds']=max(last['end_seconds'],e['end_seconds'])
            last['severity']=max((last['severity'],e['severity']),key=lambda x:['minor','moderate','severe'].index(x))
            last.setdefault('word_indices',[last['word_index']])
            if e['word_index'] not in last['word_indices']:
                last['word_indices'].append(e['word_index'])
            if last['type'] == 'pause':
                last['measurement'] = round(float(last['measurement']) + float(e['measurement']), 3)
            last['explanation']+=' '+e['explanation']
        else:
            item=dict(e)
            item['word_indices']=[e['word_index']]
            merged.append(item)
    return sorted(merged,key=lambda x:x['start_seconds'])


def _rubric(events,rows,same_setup):
    """Operational rubric over displayed merged events, not a listener rating.

    Pacing and level use unique affected words so merging adjacent detections
    does not multiply their penalty. Pause cost accumulates measured excess
    seconds, so one long silence is not treated like one tiny event.
    """
    n=max(1,sum(r.get('usable',False) for r in rows))
    out={}
    for name,kinds in [('pacing',{'rush','drag'}),('pauses',{'pause'}),('local_level',{'quiet'})]:
        relevant=[e for e in events if e['type'] in kinds]
        if name == 'pauses':
            excess_seconds=sum(max(0.,float(e.get('measurement',0))) for e in relevant)
            score=max(0.,10.-3.*excess_seconds)
            out[name]={'score':round(score,1),'event_count':len(relevant),
                       'excess_pause_seconds':round(excess_seconds,3)}
        else:
            severity_weight={'minor':1,'moderate':2,'severe':3}
            burden_by_word={}
            for e in relevant:
                weight=severity_weight[e['severity']]
                for i in e.get('word_indices',[e['word_index']]):
                    burden_by_word[i]=max(weight,burden_by_word.get(i,0))
            burden=sum(burden_by_word.values())
            # Each word can contribute once per dimension, even if events merge.
            out[name]={'score':round(max(0.,10.-20.*burden/n),1),
                       'event_count':len(relevant),'affected_words':len(burden_by_word)}
    available=[v['score'] for v in out.values() if v['score'] is not None]
    out['combined']={'score':round(sum(available)/len(available),1) if available else None,
                     'basis':'Uncalibrated reference-relative operational rubric; higher means fewer detected deviations.'}
    return out
