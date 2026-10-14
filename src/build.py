"""Rebuild the release offline from included source audio and frozen crop/word records."""
from pathlib import Path
import argparse,json,csv,platform,importlib.metadata
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from core import *
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=ap.parse_args();root=args.root
 for f in ['audio','features','transcripts','metadata','reports','splits']:(root/f).mkdir(exist_ok=True)
 sources=json.loads((root/'metadata/sources.json').read_text(encoding='utf-8'));crops=json.loads((root/'metadata/crops.json').read_text(encoding='utf-8'));decoded={};source_gains={}
 for s in sources:
  p=root/s['audio_path'];assert sha(p)==s['sha256'];x,sr=sf.read(p);x=x.mean(axis=1) if x.ndim==2 else x;x=resample_poly(x,SR,sr);gain=min(1.,.75/max(float(np.max(np.abs(x))),1e-12));source_gains[s['source_id']]=gain;x=x*gain;decoded[s['source_id']]=np.clip(np.rint(x*32768),-32768,32767).astype('int16')
 manifest=[];pairs=[];labels=[];alignments=[]
 for c in crops:
  fid=c['family_id'];base=fid+'_base';x=decoded[c['source_id']][c['source_start_sample_16k']:c['source_end_sample_16k']];words=c['words'];(root/f'transcripts/{fid}.txt').write_text(c['text']+'\n',encoding='utf-8')
  variants=[(base,'none','none',x,words,None)]
  # Middle third target, separated from crop edges. Pause placed between words.
  i=max(2,len(words)//3);j=min(len(words)-2,i+max(4,len(words)//4));a=words[i]['start_sample'];b=words[j-1]['end_sample'];pause_at=(words[i-1]['end_sample']+words[i]['start_sample'])//2
  for kind in ['pause','quiet','rush','drag']:
   for level in LEVELS:
    aa=pause_at if kind=='pause' else a;bb=aa if kind=='pause' else b;y,e=edit(x,kind,level,aa,bb);cid=f'{fid}_{kind}_{level}';mapped=map_words(words,e,kind);variants.append((cid,kind,level,y,mapped,e))
    label={'clip_id':cid,'baseline_clip_id':base,'family_id':fid,'split':c['split'],'flaw_type':kind,'severity':level,'severity_basis':'ordered digital edit magnitude; human rating unknown','edit':e,'start_seconds':e['variant_start_sample']/SR,'end_seconds':e['variant_end_sample']/SR,'timing_basis':'exact half-open PCM sample interval of edit; semantic word boundary is estimated','human_rating':None,'causal_explanation':{'pause':f'Inserted {e["parameter"]:.2f} seconds of digital silence; subsequent words shift by that exact amount.','quiet':f'Interior amplitude multiplied by 10^({e["parameter"]}/20); interior RMS decreases by {abs(e["parameter"])} dB; 10 ms ramps connect boundaries.','rush':f'Local output duration is round(input samples / {e["parameter"]})/16000 seconds. Phase-vocoder frequency tracking targets stable pitch; naturalness is unverified.','drag':f'Local output duration is round(input samples / {e["parameter"]})/16000 seconds. This increases duration without intentionally lowering pitch; artifacts may remain.'}[kind]};labels.append(label)
    pairs.append({'pair_id':cid,'family_id':fid,'split':c['split'],'baseline_clip_id':base,'variant_clip_id':cid,'transcript_path':f'transcripts/{fid}.txt','same_canonical_transcript':True})
  for cid,kind,level,y,ws,e in variants:
   path=f'audio/{cid}.wav';write(root/path,y);f=features(y);np.savez_compressed(root/f'features/{cid}.npz',**f)
   record={'clip_id':cid,'family_id':fid,'source_id':c['source_id'],'speaker_id':c['speaker_id'],'split':c['split'],'condition':'human_archival_reference_candidate' if kind=='none' else 'digitally_modified_human_recording','flaw_type':kind,'severity':level,'audio_path':path,'audio_sha256':sha(root/path),'transcript_path':f'transcripts/{fid}.txt','normalized_text_sha256':__import__('hashlib').sha256(' '.join(norm(c['text'])).encode()).hexdigest(),'feature_path':f'features/{cid}.npz','source_linear_gain':source_gains[c['source_id']],'sample_rate_hz':SR,'channels':1,'pcm_bits':16,'num_samples':len(y),'duration_seconds':len(y)/SR,'source_start_sample_16k':c['source_start_sample_16k'],'source_end_sample_16k':c['source_end_sample_16k'],'word_count':len(ws),'whole_clip_words_per_minute':len(ws)*60/(len(y)/SR),'baseline_quality_human_verified':False,'transcript_human_verified':False,'word_alignment_human_verified':False,'review_status':'pending','tts_used':False};manifest.append(record)
   for k,w in enumerate(ws):alignments.append({'clip_id':cid,'word_index':k,**w,'start_seconds':w['start_sample']/SR,'end_seconds':w['end_sample']/SR,'timing_method':'PocketSphinx 5.0.4 10ms estimates'+(' mapped through known edit' if e else ''),'human_verified':False})
  print('Built',fid,len(variants),'WAVs',flush=True)
 for name,rows in [('manifest',manifest),('pairs',pairs),('flaw_labels',labels),('word_alignment',alignments)]:jsonl(root/f'metadata/{name}.jsonl',rows)
 with (root/'metadata/manifest.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
 for split in ['train','validation','test']:jsonl(root/f'splits/{split}.jsonl',[m for m in manifest if m['split']==split])
 review=root/'metadata/human_review.csv'
 if not review.exists():
  with review.open('w',newline='',encoding='utf-8') as f:
   fields=['clip_id','reviewer_id','transcript_correct','word_times_correct','reference_appropriate','perceived_severity_0_to_3','naturalness_1_to_5','corrected_transcript','notes'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({'clip_id':m['clip_id']} for m in manifest)
 dump(root/'reports/build_environment.json',{'python':platform.python_version(),'platform':platform.platform(),'dependencies':{x:importlib.metadata.version(x) for x in ['numpy','scipy','soundfile']},'seed':'none; deterministic operations','generation':'offline source decoding + frozen source crops/estimated word alignments + exact signal operations'})
 dump(root/'reports/statistics.json',{'families':len(crops),'wav_files':len(manifest),'pairs':len(pairs),'speakers':len(set(m['speaker_id'] for m in manifest)),'baseline_minutes':sum(m['duration_seconds'] for m in manifest if m['flaw_type']=='none')/60,'total_audio_minutes':sum(m['duration_seconds'] for m in manifest)/60,'splits':{s:{'families':len(set(m['family_id'] for m in manifest if m['split']==s)),'wav_files':sum(m['split']==s for m in manifest),'pairs':sum(p['split']==s for p in pairs)} for s in ['train','validation','test']}})
if __name__=='__main__':main()
