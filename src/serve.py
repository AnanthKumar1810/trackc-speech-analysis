"""Local-only dataset review and same-transcript upload comparison. No external services."""
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
import json,base64,io,argparse,sys
import numpy as np,soundfile as sf
from scipy.signal import resample_poly
from core import *
from infer_model import infer_pair
from transcript_guard import check_pcm
from human_compare import decode, compare as compare_human
from localizer import detect
from align_dtw import align_from_reference
ROOT=Path(__file__).resolve().parents[1]
def reference_words(clip_id, expected):
 mfa=ROOT/'alignment_mfa/word_alignment_mfa.jsonl'
 if mfa.exists():
  rows=[w for w in loadl(mfa) if w['clip_id']==clip_id]
  if [w['word'] for w in rows]==expected:return rows,'MFA automatic'
 rows=[w for w in loadl(ROOT/'metadata/word_alignment.jsonl') if w['clip_id']==clip_id]
 return rows,'PocketSphinx automatic'
def analyze(payload):
 ms={m['clip_id']:m for m in loadl(ROOT/'metadata/manifest.jsonl')};m=ms.get(payload.get('reference_id'))
 if m is None or m['flaw_type']!='none':raise ValueError('Choose an included reference recording.')
 text=payload.get('transcript','');canonical=(ROOT/m['transcript_path']).read_text(encoding='utf-8')
 if norm(text)!=norm(canonical):raise ValueError('The upload transcript must match the selected reference text. Correct it or choose another reference.')
 data=base64.b64decode(payload['audio_base64'],validate=True)
 x,sr=sf.read(io.BytesIO(data));x=x.mean(axis=1) if x.ndim==2 else x
 if len(x)/sr>60 or len(x)/sr<1:raise ValueError('Use a recording between 1 and 60 seconds.')
 x=resample_poly(x,SR,sr);pcm=np.clip(np.rint(x*32768),-32768,32767).astype('int16')
 if payload.get('transcript_checked_by_user') is True:
  transcript_check={'status':'user_confirmed_spoken_words','note':'The operator reported listening to the words; this has not been independently verified.'}
 else:
  try: transcript_check=check_pcm(pcm,canonical)
  except Exception as e:transcript_check={'status':'asr_unavailable','detail':str(e),'note':'No automatic spoken-word verification was possible.'}
 if transcript_check['status']!='asr_agrees_provisionally' and payload.get('transcript_checked_by_user') is not True:
  return {'status':'needs_transcript_review','transcript_check':transcript_check,'words':[],
   'caution':'No word-level feedback was produced. Listen and correct the transcript or explicitly confirm the spoken words before retrying.'}
 bw,reference_timing=reference_words(m['clip_id'],norm(text));base=read(ROOT/m['audio_path']);bf=features(base);uf=features(pcm)
 ws=None;participant_timing='PocketSphinx automatic';alignment_diagnostic=None
 try:
  from select_crops import align
  ws=align(pcm,norm(text))
 except ImportError:pass
 if ws is None:
  ws,alignment_diagnostic=align_from_reference(bf,uf,bw,len(pcm))
  participant_timing='bounded MFCC-DTW word-boundary transfer from reference'
 if ws is None:raise ValueError('Automatic alignment failed quality checks. No word-level comparison was produced. Try a clearer recording and verify the transcript. '+str(alignment_diagnostic))
 threshold_file=ROOT/'localizer/thresholds.json'
 thresholds=json.loads(threshold_file.read_text(encoding='utf-8'))['thresholds'] if threshold_file.exists() else None
 localized=detect(base,pcm,bw,ws,same_setup=payload.get('same_setup_confirmed') is True,
                  thresholds=thresholds,reference_features=bf,take_features=uf)
 base_global=np.median(bf['rms_dbfs']);user_global=np.median(uf['rms_dbfs']);rows=[]
 for a,b in zip(bw,ws):
  ar=base[a['start_sample']:a['end_sample']].astype(float);br=pcm[b['start_sample']:b['end_sample']].astype(float);adb=20*np.log10(np.sqrt(np.mean(ar**2))+1e-6);bdb=20*np.log10(np.sqrt(np.mean(br**2))+1e-6)
  rows.append({'word':b['word'],'start_seconds':b['start_sample']/SR,'end_seconds':b['end_sample']/SR,'duration_ratio_to_reference':(b['end_sample']-b['start_sample'])/(a['end_sample']-a['start_sample']),'relative_energy_difference_db':(bdb-user_global)-(adb-base_global)})
 return {'status':'automatic comparison; not a speaking-quality rating','transcript_check':transcript_check,'reference_timing':reference_timing,'participant_timing':participant_timing,'alignment_diagnostic':alignment_diagnostic,'words':rows,'duration_ratio':len(pcm)/len(base),'features':compact(uf),'localizer':localized,'model_edit_signature':infer_pair(base,pcm,base_features=bf,variant_features=uf),'caution':'ASR agreement does not verify spoken words. Word boundaries and pitch are estimates. The model was trained on digital edits and has not been validated for natural speaking errors. Recording conditions can dominate energy differences.'}
def compact(f):
 return {k:[None if not np.isfinite(float(v)) else round(float(v),4) for v in f[k][::2]] for k in ['time_seconds','relative_rms_db','relative_pitch_semitones','rms_dbfs']}

def analyze_human(payload):
 reference=decode(base64.b64decode(payload['reference_audio_base64'],validate=True))
 take=decode(base64.b64decode(payload['audio_base64'],validate=True))
 result=compare_human(reference,take,
   same_words_confirmed=payload.get('same_words_confirmed') is True,
   same_setup_confirmed=payload.get('same_setup_confirmed') is True,
   same_speaker_confirmed=payload.get('same_speaker_confirmed') is True)
 if payload.get('check_transcript') is True:
  expected=payload.get('transcript','').strip()
  if not expected:raise ValueError('Enter the expected transcript for the optional speech-recognition check.')
  result['transcript_checks']={}
  for name,x in [('reference',reference),('take',take)]:
   pcm=np.clip(np.rint(x*32768),-32768,32767).astype('int16')
   try:result['transcript_checks'][name]=check_pcm(pcm,expected)
   except Exception as exc:result['transcript_checks'][name]={'status':'asr_unavailable','detail':str(exc)}
 return result
class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(ROOT),**kwargs)
 def do_POST(self):
  if self.path not in ('/api/analyze','/api/compare-human','/api/compare-pair'):self.send_error(404);return
  try:
   n=int(self.headers.get('Content-Length',0))
   if not 0<n<15_000_000:raise ValueError('Upload is too large (maximum request 15 MB).')
   payload=json.loads(self.rfile.read(n))
   result=(analyze_human if self.path in ('/api/compare-human','/api/compare-pair') else analyze)(payload);status=200
  except Exception as e:result={'error':str(e)};status=400
  body=json.dumps(result,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(body)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8765);args=ap.parse_args();print(f'Open http://127.0.0.1:{args.port}/evaluation.html',flush=True);ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
