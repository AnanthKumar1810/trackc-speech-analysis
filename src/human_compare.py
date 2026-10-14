"""Human speech comparison v1: pretrained VAD plus interpretable measurements.

No participant identity, intended take label, or classifier prediction enters
the measurements. Thresholds are engineering defaults, not learned ratings.
"""
import io
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from scipy.ndimage import binary_closing, binary_opening

SR = 16000
VERSION = 'human-comparison-1.0'
CONFIG = {'pace_fast_ratio': 1.15, 'pace_slow_ratio': .85,
          'pause_min_seconds': .30, 'pause_excess_seconds': .35,
          'volume_change_db': 3., 'pause_position_tolerance': .12}


def decode(source):
    """Decode locally; mono 16 kHz, no loudness normalization of saved audio."""
    try:
        x, sr = sf.read(io.BytesIO(source) if isinstance(source, bytes) else source)
        if x.ndim == 2: x = x.mean(axis=1)
        if sr != SR: x = resample_poly(x, SR, sr)
    except Exception:
        import av
        with av.open(io.BytesIO(source) if isinstance(source, bytes) else str(source)) as media:
            if not media.streams.audio: raise ValueError('The file contains no audio.')
            resampler = av.AudioResampler(format='fltp', layout='mono', rate=SR)
            chunks = []
            count = 0
            for frame in media.decode(audio=0):
                for converted in resampler.resample(frame):
                    chunks.append(converted.to_ndarray().reshape(-1))
                    count += converted.samples
                    if count > SR * 61: raise ValueError('Use audio between 1 and 60 seconds.')
            for converted in resampler.resample(None):
                chunks.append(converted.to_ndarray().reshape(-1))
            if not chunks: raise ValueError('No audio decoded.')
            x = np.concatenate(chunks)
    x = np.asarray(x, dtype=np.float32)
    if not SR <= len(x) <= 60 * SR: raise ValueError('Use audio between 1 and 60 seconds.')
    if not np.isfinite(x).all(): raise ValueError('Audio contains invalid values.')
    return x


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return list(zip(edges[::2], edges[1::2]))


def activity(x, method='auto'):
    """Normalize for detection only; retain original level for volume metrics."""
    peak = float(np.max(np.abs(x)))
    if peak < 1e-6: return [], 'no_signal'
    detector_audio = np.clip(x * min(100., .8 / peak), -1., 1.)
    if method != 'energy':
        try:
            from faster_whisper.vad import get_speech_timestamps, VadOptions
            options = VadOptions(threshold=.5, min_speech_duration_ms=100,
                                 min_silence_duration_ms=200, speech_pad_ms=0)
            segments = get_speech_timestamps(detector_audio, options, sampling_rate=SR)
            return [(s['start'], s['end']) for s in segments], 'Silero VAD (faster-whisper bundled)'
        except ImportError:
            if method == 'silero': raise
    # Dependency fallback: scale-invariant energy regions, not speech certainty.
    size = 320
    frames = x[:len(x)//size*size].reshape(-1, size)
    energy = np.sqrt(np.mean(frames**2, axis=1))
    high = float(np.quantile(energy, .9))
    if high < 1e-7: return [], 'no_signal'
    mask = energy > high * .15
    mask = binary_closing(mask, structure=np.ones(4))
    mask = binary_opening(mask, structure=np.ones(3))
    return [(a*size, b*size) for a,b in runs(mask)], 'adaptive energy fallback (less reliable)'


def measure(x, method='auto'):
    regions, detector = activity(x, method)
    quality = []
    peak = float(np.max(np.abs(x)))
    near_full_scale = float(np.mean(np.abs(x) >= .999))
    if near_full_scale > .001: quality.append('Possible clipping or limiter activity; inspect recording.')
    total = len(x)/SR
    base = {'duration_seconds': round(total,3), 'detector': detector,
            'near_full_scale_fraction': round(near_full_scale,6), 'quality_flags': quality}
    if not regions:
        return dict(base, status='no_speech_detected', pauses=[], speech_regions=[], waveform=[])
    voiced = sum(b-a for a,b in regions)/SR
    if voiced < .5: quality.append('Very little speech detected.')
    span = (regions[-1][1]-regions[0][0])/SR
    signal = np.concatenate([x[a:b] for a,b in regions])
    rms = 20*np.log10(max(float(np.sqrt(np.mean(signal**2))),1e-12))
    pauses=[]; elapsed=0.
    for (a,b),(c,d) in zip(regions,regions[1:]):
        elapsed += (b-a)/SR
        if (c-b)/SR >= CONFIG['pause_min_seconds']:
            pauses.append({'start_seconds':round(b/SR,3),'end_seconds':round(c/SR,3),
                           'seconds':round((c-b)/SR,3),'speech_progress':round(elapsed/max(voiced,1e-9),4)})
    step=800
    waveform=[{'seconds':round(i/SR,3),'rms':round(float(np.sqrt(np.mean(x[i:i+step]**2))),5)} for i in range(0,len(x),step)]
    return dict(base, status='measured', speech_start_seconds=round(regions[0][0]/SR,3),
                speech_end_seconds=round(regions[-1][1]/SR,3), speaking_span_seconds=round(span,3),
                detected_speech_seconds=round(voiced,3), internal_nonspeech_seconds=round(span-voiced,3),
                speech_rms_dbfs=round(float(rms),2), pauses=pauses,
                speech_regions=[[round(a/SR,3),round(b/SR,3)] for a,b in regions],waveform=waveform)


def compare(reference, take, same_words_confirmed=False, same_setup_confirmed=False,
            same_speaker_confirmed=False, method='auto'):
    a,b=measure(reference,method),measure(take,method)
    result={'version':VERSION,'status':'development_measurements','reference':a,'take':b,
            'thresholds':CONFIG,'same_words_confirmed':bool(same_words_confirmed),
            'same_setup_confirmed':bool(same_setup_confirmed),
            'same_speaker_confirmed':bool(same_speaker_confirmed),
            'human_accuracy':None,'severity_rating':None,'observations':[]}
    if a['status']!='measured' or b['status']!='measured':
        result['status']='insufficient_speech'; return result
    speed=a['detected_speech_seconds']/max(b['detected_speech_seconds'],.01)
    span_ratio=a['speaking_span_seconds']/max(b['speaking_span_seconds'],.01)
    delta=b['speech_rms_dbfs']-a['speech_rms_dbfs']
    paired=bool(same_words_confirmed and same_speaker_confirmed)
    pace='faster' if speed>CONFIG['pace_fast_ratio'] else 'slower' if speed<CONFIG['pace_slow_ratio'] else 'similar'
    delivery='faster' if span_ratio>CONFIG['pace_fast_ratio'] else 'slower' if span_ratio<CONFIG['pace_slow_ratio'] else 'similar'
    result['pace']={'speech_time_ratio':round(speed,3),'whole_span_rate_ratio':round(span_ratio,3),
                    'direction':pace if paired else 'requires_same_words_and_speaker_confirmation',
                    'provisional_direction':pace,'provisional_delivery_direction':delivery,
                    'delivery_direction':delivery if paired else 'requires_same_words_and_speaker_confirmation',
                    'basis':'Speech-time ratio excludes pauses; whole-span ratio includes internal pauses. Detector errors affect both.'}
    # Match gaps by cumulative speech progress, not absolute clock time.
    # This is only a coarse position correspondence, never a word alignment.
    used=set(); events=[]
    for p in b['pauses']:
        candidates=[(abs(p['speech_progress']-q['speech_progress']),i,q) for i,q in enumerate(a['pauses']) if i not in used]
        best=min(candidates,default=None)
        matched=best is not None and best[0]<=CONFIG['pause_position_tolerance']
        q=best[2] if matched else None
        if matched: used.add(best[1])
        extra=p['seconds']-(q['seconds'] if q else 0.)
        if extra>=CONFIG['pause_excess_seconds']:
            events.append(dict(p,extra_seconds=round(extra,3),reference_pause_seconds=q['seconds'] if q else 0.,
                               status='candidate_extra_pause' if paired else 'provisional_extra_pause'))
    result['pause_comparison']={'candidates':events,'total_internal_nonspeech_change_seconds':round(b['internal_nonspeech_seconds']-a['internal_nonspeech_seconds'],3),
                                'note':'Pauses can be appropriate rhetoric; candidates are not verified flaws or word boundaries.'}
    volume='lower' if delta<=-CONFIG['volume_change_db'] else 'higher' if delta>=CONFIG['volume_change_db'] else 'similar'
    result['volume']={'speech_level_change_db':round(delta,2),'direction':volume if same_setup_confirmed and same_speaker_confirmed else 'recording_setup_unconfirmed',
                      'provisional_direction':volume,'note':'Recorded level is affected by microphone gain, automatic gain control and distance; it does not prove vocal effort.'}
    if paired:
        result['observations'].append(f'Detected speech is {pace} relative to this speaker\'s baseline (ratio {speed:.2f}).')
    else:
        result['observations'].append('Confirm the same speaker and spoken words before interpreting pace or corresponding pause locations.')
    if events: result['observations'].append(f'{len(events)} candidate longer pause(s); listen to the highlighted intervals.')
    result['observations'].append(f'Recorded speech level differs by {delta:+.1f} dB; setup confirmation is required for a comparable level judgment.')
    return result
