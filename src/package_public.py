"""Create the shareable package, excluding private human audio and local caches."""
from pathlib import Path
import hashlib,zipfile

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT.parent/'TrackC_Public_Submission_2026-10-14.zip'
EXCLUDE_TOP={'.deps','.pip-cache','chrome-print-profile','edge-print-profile',
             'human_pilot','real_evaluation'}
EXCLUDE_FILES={'human.html','human_app.js','HUMAN_LAB_README.md',
               'HUMAN_MODEL_CARD.md','INSTALL_PRIVATE_ADDON.md',
               'START_HUMAN_LAB.cmd','evaluation_script.js',
               'TECHNICAL_REPORT.md','TECHNICAL_REPORT.pdf',
               'SHA256SUMS.txt','SHA256SUMS_PUBLIC.txt'}

def included(path):
    rel=path.relative_to(ROOT)
    if path.is_symlink() or not path.is_file():return False
    if rel.parts[0] in EXCLUDE_TOP or path.name in EXCLUDE_FILES:return False
    if '__pycache__' in rel.parts or path.suffix in ('.pyc','.pyo'):return False
    if rel.parts[0]=='reports' and ('human' in path.name or path.name=='user_review.json'):
        return False
    return True

def main():
    files=sorted((p for p in ROOT.rglob('*') if included(p)),key=lambda p:p.relative_to(ROOT).as_posix())
    for p in files:
        rel=p.relative_to(ROOT).as_posix().lower()
        if 'human_pilot/' in rel:
            raise RuntimeError('Private audio path entered the public package: '+rel)
    sums=[]
    for p in files:
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
        sums.append(h.hexdigest()+'  '+p.relative_to(ROOT).as_posix())
    checksum='\n'.join(sums)+'\n'
    (ROOT/'SHA256SUMS_PUBLIC.txt').write_text(checksum,encoding='utf-8')
    with zipfile.ZipFile(DEST,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
        z.write(ROOT/'SHA256SUMS_PUBLIC.txt','SHA256SUMS_PUBLIC.txt')
    with zipfile.ZipFile(DEST) as z:
        names=set(z.namelist())
        assert 'evaluation.html' in names and 'TECHNICAL_REPORT_2026-10-14.pdf' in names
        assert not any(n.startswith('human_pilot/') or n.startswith('.deps/') for n in names)
        assert z.testzip() is None
    print(f'{DEST} | files={len(files)+1} | bytes={DEST.stat().st_size}')

if __name__=='__main__':main()
