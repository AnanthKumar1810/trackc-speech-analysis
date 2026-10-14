"""Create candidate source crops using published sentences and independent ASR anchors.
Used once for selection; frozen inputs in metadata permit exact ordinary rebuilds.
"""
from pathlib import Path
import re,json,difflib,sys
import numpy as np
import soundfile as sf
from pocketsphinx import Decoder
sys.path.insert(0,str(Path(__file__).resolve().parent))
from core import norm,dump,SR

def align(pcm,words):
 d=Decoder();missing=[w for w in words if not d.lookup_word(w)]
 if missing:return None
 d.set_align_text(' '.join(words));d.start_utt();d.process_raw(np.r_[pcm,np.zeros(8000,dtype=np.int16)].astype('<i2').tobytes(),full_utt=True);d.end_utt()
 if d.seg() is None:return None
 seg=[{'word':re.sub(r'\(\d+\)$','',s.word),'start_sample':s.start_frame*160,'end_sample':(s.end_frame+1)*160} for s in d.seg() if not s.word.startswith(('<','['))]
 if [s['word'] for s in seg]!=words:return None
 if any(s['end_sample']>len(pcm) or not 320<=s['end_sample']-s['start_sample']<=19200 for s in seg):return None
 return seg

def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=ap.parse_args();p=args.inputs;root=args.root;out=[];decisions=[]
 for name,want,split,speaker in [('reagan',12,'train','ronald_reagan'),('obama',8,'test','barack_obama')]:
  text=(p/(name+'_canonical.txt')).read_text(encoding='utf-8').replace('1917','nineteen seventeen').replace('4','four')
  asr=json.loads((p/(name+'_asr.json')).read_text(encoding='utf-8'));asr=[dict(s,word=re.sub(r'\(\d+\)$','',s['word'])) for s in asr if not s['word'].startswith(('<','['))]
  cw=norm(text);aw=[s['word'] for s in asr];mapping={}
  for m in difflib.SequenceMatcher(None,cw,aw,autojunk=False).get_matching_blocks():
   for i in range(m.size):mapping[m.a+i]=m.b+i
  pcm,sr=sf.read(p/(name+'_16k.wav'),dtype='int16');offset=0;candidates=[];last_end=0
  for sentence in re.split(r'(?<=[.!?])\s+',text):
   words=norm(sentence);n=len(words);idx=offset;offset+=n
   if not 18<=n<=55:continue
   matched=[i for i in range(idx,idx+n) if i in mapping]
   if len(matched)<n*(.30 if name=='obama' else .50) or min(matched,default=99999)>idx+2 or max(matched,default=0)<idx+n-3:continue
   l=matched[0];r=matched[-1];a=max(0,asr[mapping[l]]['start_sample']-round((l-idx)*.4*SR)-4000);b=min(len(pcm),asr[mapping[r]]['end_sample']+round((idx+n-1-r)*.4*SR)+4000)
   if a<last_end or not 6*SR<b-a<36*SR:continue
   seg=align(pcm[a:b],words)
   if seg is None:decisions.append({'source':name,'sentence':sentence,'status':'excluded_alignment_check'});continue
   left=a+max(0,seg[0]['start_sample']-800);right=a+min(b-a,seg[-1]['end_sample']+800)
   if left<last_end:continue
   if max([seg[i+1]['start_sample']-seg[i]['end_sample'] for i in range(n-1)])>SR*1.5:continue
   candidates.append({'source_id':name,'speaker_id':speaker,'split':split,'source_start_sample_16k':left,'source_end_sample_16k':right,'text':sentence.strip(),'words':[dict(s,start_sample=a+s['start_sample']-left,end_sample=a+s['end_sample']-left) for s in seg],'asr_anchor_word_coverage':len(matched)/n,'transcript_basis':'published source transcript; matched by automatic ASR anchors; listening review pending'})
   last_end=right
  pick=np.unique(np.linspace(0,len(candidates)-1,min(want,len(candidates))).round().astype(int)) if candidates else []
  out.extend(candidates[i] for i in pick);print(name,len(candidates),'candidates',len(pick),'selected',flush=True)
 # JFK excerpts avoid the uncertain final word in the moon download.
 for name,start,end in [('jfk_moon',0,25),('jfk_moon',25,52),('jfk_return',0,31)]:
  d=json.loads((p/(name+'_aligned.json')).read_text(encoding='utf-8'));seg=d['segments'][start:end];pcm,sr=sf.read(p/(name+'_16k.wav'),dtype='int16');a=max(0,seg[0]['start_sample']-400);b=min(len(pcm),seg[-1]['end_sample']+400)
  out.append({'source_id':name,'speaker_id':'john_f_kennedy','split':'validation','source_start_sample_16k':a,'source_end_sample_16k':b,'text':' '.join(re.sub(r'\(\d+\)$','',s['word']) for s in seg),'words':[dict(s,word=re.sub(r'\(\d+\)$','',s['word']),start_sample=s['start_sample']-a,end_sample=s['end_sample']-a) for s in seg],'asr_anchor_word_coverage':None,'transcript_basis':'canonical historical quotation adapted to NASA excerpt; automatic alignment; listening review pending'})
 # Remove padding overlap for adjacent crops (never words).
 for source in set(c['source_id'] for c in out):
  ordered=sorted([c for c in out if c['source_id']==source],key=lambda c:c['source_start_sample_16k'])
  for before,after in zip(ordered,ordered[1:]):
   if before['source_end_sample_16k']>after['source_start_sample_16k']:
    boundary=(before['source_end_sample_16k']+after['source_start_sample_16k'])//2;shift=boundary-after['source_start_sample_16k'];before['source_end_sample_16k']=boundary;after['source_start_sample_16k']=boundary
    for w in after['words']:w['start_sample']-=shift;w['end_sample']-=shift
 for i,c in enumerate(out):c['family_id']=f"{c['source_id']}_{i+1:03d}"
 dump(root/'metadata/crops.json',out);dump(root/'reports/selection_exclusions.json',decisions)
if __name__=='__main__':main()
