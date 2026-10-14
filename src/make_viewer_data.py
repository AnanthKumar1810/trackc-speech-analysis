from pathlib import Path
import json,collections,numpy as np
from core import *
def compact(f):return {k:[None if not np.isfinite(float(v)) else round(float(v),3) for v in f[k][::2]] for k in ['time_seconds','relative_rms_db','relative_pitch_semitones','rms_dbfs']}
r=Path(__file__).resolve().parents[1];ms=loadl(r/'metadata/manifest.jsonl');labels={l['clip_id']:l for l in loadl(r/'metadata/flaw_labels.jsonl')};ws=collections.defaultdict(list)
alignment=r/'alignment_mfa/word_alignment_mfa.jsonl'
if not alignment.exists():alignment=r/'metadata/word_alignment.jsonl'
for w in loadl(alignment):ws[w['clip_id']].append(w)
clips={m['clip_id']:{**m,'label':labels.get(m['clip_id']),'words':ws[m['clip_id']],'features':compact(np.load(r/m['feature_path']))} for m in ms}
data={'families':json.loads((r/'metadata/crops.json').read_text(encoding='utf-8')),'clips':clips};(r/'viewer_data.js').write_text('const DATA = '+json.dumps(data,ensure_ascii=False,allow_nan=False)+';\n',encoding='utf-8');print('Wrote reviewer data',len(clips))
