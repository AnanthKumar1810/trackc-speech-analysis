"""Validate real PCM, exact edits, source crops, transcript families, split integrity, and features."""
from pathlib import Path
import argparse,json,itertools,collections
import numpy as np
import soundfile as sf
from core import *
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=ap.parse_args();r=args.root
 ms=loadl(r/'metadata/manifest.jsonl');by={m['clip_id']:m for m in ms};labels=loadl(r/'metadata/flaw_labels.jsonl');pairs=loadl(r/'metadata/pairs.jsonl');alignment=loadl(r/'metadata/word_alignment.jsonl');words=collections.defaultdict(list)
 for w in alignment:words[w['clip_id']].append(w)
 errors=[];checks=collections.Counter();signals={};warnings=[]
 def check(ok,category,detail):
  checks[category]+=1
  if not ok:errors.append(category+': '+detail)
 check(len(by)==len(ms),'unique_clip_ids','duplicate id')
 for m in ms:
  cid=m['clip_id'];p=r/m['audio_path'];info=sf.info(p);x=read(p);signals[cid]=x;ws=words[cid];text=(r/m['transcript_path']).read_text(encoding='utf-8')
  check(info.subtype=='PCM_16' and info.channels==1 and info.samplerate==SR,'wav_format',cid)
  check(len(x)==m['num_samples'] and abs(len(x)/SR-m['duration_seconds'])<1e-9,'duration',cid)
  check(sha(p)==m['audio_sha256'],'audio_checksum',cid)
  check(norm(text)==[w['word'] for w in ws],'canonical_word_coverage',cid)
  check(all(0<=w['start_sample']<w['end_sample']<=len(x) for w in ws),'word_bounds',cid)
  check(all(a['end_sample']<=b['start_sample'] for a,b in zip(ws,ws[1:])),'word_monotonicity',cid)
  check(np.max(abs(x.astype(float)))<32767,'no_digital_clipping',cid)
  check(np.sqrt(np.mean((x.astype(float)/32768)**2))>1e-4,'audible_energy',cid)
  f=np.load(r/m['feature_path']);check(len(f['time_seconds'])>0 and len(f['time_seconds'])==len(f['mfcc']) and np.isfinite(f['mfcc']).all(),'feature_dimensions',cid)
  check(m['review_status']=='pending' and not m['word_alignment_human_verified'] and not m['tts_used'],'provenance_flags',cid)
 # Independently decode originals and verify every reference crop against provenance.
 from scipy.signal import resample_poly
 for source in json.loads((r/'metadata/sources.json').read_text(encoding='utf-8')):
  p=r/source['audio_path'];check(sha(p)==source['sha256'],'original_source_hash',source['source_id']);x,sr=sf.read(p);x=x.mean(axis=1) if x.ndim==2 else x;x=resample_poly(x,SR,sr);gain=min(1.,.75/max(float(np.max(np.abs(x))),1e-12));pcm=np.clip(np.rint(x*gain*32768),-32768,32767).astype('int16')
  for m in ms:
   if m['source_id']==source['source_id'] and m['flaw_type']=='none':
    expected=pcm[m['source_start_sample_16k']:m['source_end_sample_16k']];check(np.array_equal(expected,signals[m['clip_id']]) and gain==m['source_linear_gain'],'reference_source_crop',m['clip_id'])
 for lab in labels:
  cid=lab['clip_id'];b=signals[lab['baseline_clip_id']];v=signals[cid];e=lab['edit'];a,z,c=e['baseline_start_sample'],e['baseline_end_sample'],e['variant_end_sample'];kind=lab['flaw_type'];level=lab['severity']
  check(np.array_equal(b[:a],v[:a]) and np.array_equal(b[z:],v[c:]),'unaltered_outside_edit',cid)
  expected,ee=edit(b,kind,level,a,z);check(np.array_equal(expected,v) and ee==e,'operator_reconstruction',cid)
  if kind=='pause':check(c-a==round(LEVELS[level]['pause']*SR) and not np.any(v[a:c]),'exact_silence',cid)
  if kind=='quiet':
   interior=slice(a+160,z-160);observed=20*np.log10(np.linalg.norm(v[interior].astype(float))/np.linalg.norm(b[interior].astype(float)));check(abs(observed-e['parameter'])<.02,'measured_gain_db',cid)
  if kind in ['rush','drag']:check(c-a==round((z-a)/e['parameter']),'pace_sample_count',cid)
  mapped=map_words(words[lab['baseline_clip_id']],e,kind);check(all(a['start_sample']==b['start_sample'] and a['end_sample']==b['end_sample'] for a,b in zip(mapped,words[cid])),'word_time_mapping',cid)
 for p in pairs:
  a,b=by[p['baseline_clip_id']],by[p['variant_clip_id']];check(a['family_id']==b['family_id'] and a['split']==b['split'] and a['normalized_text_sha256']==b['normalized_text_sha256'],'pair_consistency',p['pair_id'])
 for group in ['speaker_id','source_id','family_id','normalized_text_sha256','audio_sha256']:
  groups=collections.defaultdict(set)
  for m in ms:groups[m[group]].add(m['split'])
  check(all(len(s)==1 for s in groups.values()),'split_disjoint_'+group,'group crossed split')
 bases=[m for m in ms if m['flaw_type']=='none'];similar=[]
 for a,b in itertools.combinations(bases,2):
  if a['split']==b['split']:continue
  ta=norm((r/a['transcript_path']).read_text(encoding='utf-8'));tb=norm((r/b['transcript_path']).read_text(encoding='utf-8'));ga=set(zip(ta,ta[1:],ta[2:],ta[3:],ta[4:]));gb=set(zip(tb,tb[1:],tb[2:],tb[3:],tb[4:]));jac=len(ga&gb)/max(1,len(ga|gb))
  check(jac<.25,'cross_split_text_5gram_similarity',a['clip_id']+' '+b['clip_id'])
  if jac>0:similar.append({'a':a['clip_id'],'b':b['clip_id'],'jaccard_5gram':jac})
 for source in set(m['source_id'] for m in bases):
  q=sorted([m for m in bases if m['source_id']==source],key=lambda m:m['source_start_sample_16k']);check(all(a['source_end_sample_16k']<=b['source_start_sample_16k'] for a,b in zip(q,q[1:])),'source_crop_nonoverlap',source)
 for split in ['train','validation','test']:
  sm=loadl(r/f'splits/{split}.jsonl');check({m['clip_id'] for m in sm}=={m['clip_id'] for m in ms if m['split']==split},'split_file_membership',split)
  for fid in set(m['family_id'] for m in sm):check({(m['flaw_type'],m['severity']) for m in sm if m['family_id']==fid}=={('none','none')}|{(k,l) for k in ['pause','quiet','rush','drag'] for l in LEVELS},'balanced_family_operators',fid)
 # Independent signal test: duration changes while a steady tone retains its frequency.
 n=np.arange(SR*2);tone=.25*np.sin(2*np.pi*140*n/SR)
 pitch_tests=[]
 for rate in [.52,.9,1.12,1.7]:
  y=time_stretch(tone,rate);mid=y[2000:-2000];peak=np.fft.rfftfreq(len(mid),1/SR)[np.argmax(abs(np.fft.rfft(mid*np.hanning(len(mid)))))];check(abs(peak-140)<2,'phase_vocoder_tone_pitch',str(rate));pitch_tests.append({'rate':rate,'measured_hz':peak})
 report={'status':'PASS' if not errors else 'FAIL','check_counts':dict(checks),'total_checks':sum(checks.values()),'errors':errors,'limitations':['Machine validation does not certify words match the spoken audio.','The requesting user reported positive general listening feedback; detailed per-clip naturalness ratings and timing corrections remain unrecorded.','No claim of zero conceivable leakage: tested speaker/source/family/exact-audio/exact-text and 5-gram overlap; source channel and era remain confounded with split.','Pace word times are linear estimates inside phase-vocoder edits, not newly aligned ground truth.'],'cross_split_nonzero_text_similarity':similar,'independent_tone_pitch_tests':pitch_tests};dump(r/'reports/validation.json',report);print(json.dumps(report,indent=2));raise SystemExit(bool(errors))
if __name__=='__main__':main()
