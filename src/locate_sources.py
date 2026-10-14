"""Optional slower ASR anchor extraction. All bundled full originals are provenance only."""
from pathlib import Path
import argparse,json,numpy as np,soundfile as sf
from scipy.signal import resample_poly
from pocketsphinx import Decoder
from core import dump
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=ap.parse_args();p=args.root/'sources'
for name in ['reagan','obama']:
 x,sr=sf.read(p/(name+'.ogg'));x=x.mean(1) if x.ndim==2 else x;x=np.clip(np.rint(resample_poly(x,16000,sr)*32768),-32768,32767).astype('<i2');sf.write(p/(name+'_16k.wav'),x,16000,subtype='PCM_16');segs=[]
 for start in range(0,min(len(x),480*16000),40*16000):
  d=Decoder(loglevel='ERROR');d.start_utt();d.process_raw(x[start:start+40*16000].tobytes(),full_utt=True);d.end_utt()
  if d.seg():segs.extend({'word':s.word,'start_sample':start+s.start_frame*160,'end_sample':start+(s.end_frame+1)*160} for s in d.seg())
  print(name,start/16000,flush=True)
 dump(p/(name+'_asr.json'),segs)
# JFK frozen alignment inputs are included; build their temporary resampled audio for select_crops.
for name in ['jfk_moon','jfk_return']:
 x,sr=sf.read(p/(name+'.mp3'));x=x.mean(1) if x.ndim==2 else x;x=np.clip(np.rint(resample_poly(x,16000,sr)*32768),-32768,32767).astype('<i2');sf.write(p/(name+'_16k.wav'),x,16000,subtype='PCM_16')
print('Optional next step: python src/select_crops.py --inputs sources')
