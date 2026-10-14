"""Shared deterministic PCM edits and transparent acoustic measurements."""
from pathlib import Path
import hashlib, json, re
import numpy as np
import soundfile as sf
from scipy.fft import dct
SR=16000
LEVELS={'minor':{'pause':.25,'quiet':-3.,'rush':1.12,'drag':.90},'moderate':{'pause':.75,'quiet':-8.,'rush':1.35,'drag':.72},'severe':{'pause':1.50,'quiet':-15.,'rush':1.70,'drag':.52}}
def norm(t):return re.findall(r"[a-z]+(?:'[a-z]+)?",t.lower().replace('\u2019',"'"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 x,sr=sf.read(p,dtype='int16');assert sr==SR and x.ndim==1
 return x

def write(p,x):sf.write(p,np.asarray(x,dtype=np.int16),SR,subtype='PCM_16')
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def jsonl(p,rows):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
def loadl(p):return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines()]
def time_stretch(x,rate):
 # STFT phase vocoder: interpolate magnitudes, preserve instantaneous frequency.
 n=512;hop=128;window=np.hanning(n);pad=np.pad(x,(n//2,n//2+n))
 frames=np.lib.stride_tricks.sliding_window_view(pad,n)[::hop]
 spectrum=np.fft.rfft(frames*window,axis=1);steps=np.arange(0,(len(x)+n//2)/hop,rate)
 phase=np.angle(spectrum[0]);advance=2*np.pi*hop*np.arange(n//2+1)/n
 y=np.zeros((len(steps)-1)*hop+n);weight=np.zeros_like(y)
 for j,t in enumerate(steps):
  i=int(t);alpha=t-i;left=spectrum[min(i,len(spectrum)-1)];right=spectrum[min(i+1,len(spectrum)-1)]
  mag=(1-alpha)*abs(left)+alpha*abs(right);frame=np.fft.irfft(mag*np.exp(1j*phase),n)*window
  y[j*hop:j*hop+n]+=frame;weight[j*hop:j*hop+n]+=window**2
  delta=np.angle(right)-np.angle(left)-advance;delta-=2*np.pi*np.round(delta/(2*np.pi));phase+=advance+delta
 y/=np.maximum(weight,1e-10);length=round(len(x)/rate)
 return np.pad(y[n//2:n//2+length],(0,max(0,length-len(y[n//2:n//2+length]))))
def edit(x,kind,level,a,b):
 value=LEVELS[level][kind]
 if kind=='pause': y=np.zeros(round(value*SR),dtype=np.int16);b=a
 elif kind=='quiet':
  # Raised-cosine 10ms ramps avoid a sharp gain discontinuity.
  gain=10**(value/20);env=np.full(b-a,gain);r=min(160,(b-a)//4)
  ramp=.5*(1-np.cos(np.linspace(0,np.pi,r)));env[:r]=1+(gain-1)*ramp;env[-r:]=env[:r][::-1]
  y=np.rint(x[a:b].astype(float)*env).astype(np.int16)
 else:
  f=time_stretch(x[a:b].astype(np.float64)/32768.,value)
  # Keep only this span changed. Crossfade at the two boundaries without shifting its length.
  r=min(160,len(f)//4,b-a);ref=x[a:b].astype(float)/32768
  w=np.linspace(0,1,r);f[:r]=ref[:r]*(1-w)+f[:r]*w;f[-r:]=f[-r:]*(1-w)+ref[-r:]*w
  y=np.clip(np.rint(f*32768),-32768,32767).astype(np.int16)
 return np.concatenate([x[:a],y,x[b:]]),{'baseline_start_sample':a,'baseline_end_sample':b,'variant_start_sample':a,'variant_end_sample':a+len(y),'parameter':value,'parameter_unit':{'pause':'seconds inserted','quiet':'dB gain in interior','rush':'playback rate with pitch-preserving phase vocoder','drag':'playback rate with pitch-preserving phase vocoder'}[kind],'boundary_ramp_samples':0 if kind=='pause' else 160}
def map_words(words,e,kind):
 a,b,c=e['baseline_start_sample'],e['baseline_end_sample'],e['variant_end_sample'];delta=c-b
 def point(t,end=False):
  if kind=='pause':return t+delta if t>a or (t==a and not end) else t
  if t<=a:return t
  if t>=b:return t+delta
  return a+round((t-a)*(c-a)/(b-a))
 return [dict(w,start_sample=point(w['start_sample']),end_sample=point(w['end_sample'],True)) for w in words]
def features(x):
 x=np.asarray(x,dtype=float)/32768;frame=640;hop=320;nfft=1024
 z=np.lib.stride_tricks.sliding_window_view(x,frame)[::hop].copy();spec=np.abs(np.fft.rfft(z*np.hanning(frame),nfft));power=spec**2
 freq=np.fft.rfftfreq(nfft,1/SR);mel=lambda f:2595*np.log10(1+f/700);edges=700*(10**(np.linspace(0,mel(8000),28)/2595)-1)
 bank=np.array([np.maximum(0,np.minimum((freq-a)/(b-a),(c-freq)/(c-b))) for a,b,c in zip(edges[:-2],edges[1:-1],edges[2:])]);mfcc=dct(np.log(np.maximum(power@bank.T,1e-12)),type=2,norm='ortho',axis=1)[:,:13]
 rms=np.sqrt(np.mean(z*z,axis=1));db=20*np.log10(np.maximum(rms,1e-8));center=(np.arange(len(z))*hop+frame/2)/SR
 f0=[];conf=[]
 for frame_x,r in zip(z,rms):
  v=frame_x-frame_x.mean();ac=np.correlate(v,v,'full')[len(v)-1:];lag=46+np.argmax(ac[46:229]);q=ac[lag]/(ac[0]+1e-12);f0.append(SR/lag if q>.45 and r>1e-4 else np.nan);conf.append(q)
 f0=np.array(f0);voiced=f0[np.isfinite(f0)];median=float(np.median(voiced)) if len(voiced) else 1
 return {'time_seconds':center,'rms_dbfs':db,'relative_rms_db':db-np.median(db),'f0_hz':f0,'relative_pitch_semitones':12*np.log2(f0/median),'f0_periodicity':np.array(conf),'spectral_centroid_hz':(spec@freq)/(spec.sum(axis=1)+1e-12),'zcr':np.mean(np.signbit(z[:,1:])!=np.signbit(z[:,:-1]),axis=1),'mfcc':mfcc,'log_fft_power':np.log(np.maximum(power,1e-12)).astype(np.float32)}
