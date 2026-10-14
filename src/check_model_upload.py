"""Exercise the real local upload path on each generated edit category."""
from pathlib import Path
import sys,json,base64
sys.path.insert(0,str(Path(__file__).resolve().parent))
from serve import analyze
from core import loadl,dump
ROOT=Path(__file__).resolve().parents[1]
ms=loadl(ROOT/'metadata/manifest.jsonl');rows=[]
for kind in ['none','pause','quiet','rush','drag']:
 item=next(x for x in ms if x['flaw_type']==kind and x['severity'] in ['none','moderate']);payload={'reference_id':item['family_id']+'_base','transcript':(ROOT/item['transcript_path']).read_text(encoding='utf-8'),'audio_base64':base64.b64encode((ROOT/item['audio_path']).read_bytes()).decode()}
 guarded=analyze(payload);assert guarded['transcript_check']['status'] in ('asr_agrees_provisionally','needs_transcript_review','asr_unavailable')
 if guarded['status']=='needs_transcript_review':assert not guarded['words']
 payload['transcript_checked_by_user']=True;result=analyze(payload);pred=result['model_edit_signature']['type']['predicted'];rows.append({'clip_id':item['clip_id'],'true_edit_type':kind,'model_edit_type':pred,'word_count':len(result['words']),'asr_status':guarded['transcript_check']['status'],'guard_withheld_word_feedback':guarded['status']=='needs_transcript_review','upload_alignment_succeeded':True});assert pred==kind
report={'status':'PASS','test_kind':'Local analyze() with ASR guard and explicit simulated transcript confirmation on each same-script WAV; no browser or network needed','cases':rows,'limitation':'The test confirmation is simulated for pipeline coverage, not a human review. Only included digital clips; no independently recorded participant was evaluated.'};dump(ROOT/'reports/model_upload_smoke.json',report);print(report)
