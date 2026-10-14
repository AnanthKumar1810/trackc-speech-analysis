"""Download missing original recordings; refuse bytes that differ from frozen hashes."""
from pathlib import Path
import json,urllib.request
from core import sha
root=Path(__file__).resolve().parents[1]
for s in json.loads((root/'metadata/sources.json').read_text(encoding='utf-8')):
 p=root/s['audio_path']
 if p.exists() and sha(p)==s['sha256']:print('Verified',s['source_id']);continue
 request=urllib.request.Request(s['download_url'],headers={'User-Agent':'TrackCResearchDataset/2.0 (educational research)'})
 data=urllib.request.urlopen(request,timeout=120).read();tmp=p.with_suffix(p.suffix+'.download');tmp.write_bytes(data)
 if sha(tmp)!=s['sha256']:raise RuntimeError('Source changed; download retained for inspection, original not replaced: '+str(tmp))
 tmp.replace(p);print('Downloaded',s['source_id'])
