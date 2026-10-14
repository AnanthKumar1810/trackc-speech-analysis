"""Prepare a clean, audio-free project checkout for GitHub."""
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT.parent/'trackc-speech-analysis'
DIRS={'src','metadata','splits','transcripts','localizer','model','reports',
      'alignment_mfa','.github'}
FILES={'evaluation.html','pair.html','review.html','viewer_data.js','README.md',
       'DATASET_CARD.md','RIGHTS.md','LICENSE-CODE','PROJECT_STATUS.md',
       'PUBLISHING.md','TECHNICAL_REPORT_2026-10-14.pdf',
       'TECHNICAL_REPORT_2026-10-14.html','requirements.txt',
       'requirements-model.txt','requirements-human.txt','MFA_WORKFLOW.md',
       'MODEL_CARD.md','RELEASE_NOTES.md','Makefile','Dockerfile',
       '.dockerignore'}
PRIVATE_SOURCE_FILES={'prepare_human_pilot.py','validate_human_lab.py',
                      'evaluate_human_holdout.py'}

def main():
    DEST.mkdir(exist_ok=True)
    for item in ROOT.iterdir():
        if item.name in FILES and item.is_file():
            shutil.copy2(item,DEST/item.name)
        elif item.name in DIRS and item.is_dir():
            for p in item.rglob('*'):
                if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts or p.suffix=='.pyc':continue
                if item.name=='alignment_mfa' and p.name not in {
                    'word_alignment_mfa.jsonl','STATUS.md','alignment_report.json',
                    'run_environment.json'
                }:continue
                if item.name=='src' and p.name in PRIVATE_SOURCE_FILES:continue
                if item.name=='reports' and ('human' in p.name or p.name=='user_review.json'):continue
                target=DEST/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    # Keep only the small alignment results needed for the dashboard and API.
    alignment=DEST/'alignment_mfa';alignment.mkdir(exist_ok=True)
    (DEST/'.gitignore').write_text(
        'audio/\nfeatures/\nsources/\nhuman_pilot/\nreal_evaluation/\n'
        '.deps/\n.pip-cache/\n__pycache__/\n*.py[cod]\n'
        '.pytest_cache/\n.DS_Store\nThumbs.db\n',encoding='utf-8')
    shutil.copy2(DEST/'LICENSE-CODE',DEST/'LICENSE')
    # Replace the old archive note with explicit instructions for this code-only repo.
    readme=DEST/'README.md';existing=readme.read_text(encoding='utf-8')
    marker='\n## Dataset files\n'
    if marker in existing: existing=existing.split(marker)[0]
    readme.write_text(existing+marker+
        '\nThe repository intentionally excludes WAV recordings, extracted feature arrays, '
        'original source media, and participant recordings. The paired-upload page works '
        'with files you select locally. The archive evaluator and full dataset rebuild '
        'require the separate public dataset package described in `PUBLISHING.md`. '
        'Private human-participant recordings and their identifying filenames are not '
        'part of this repository.\n',encoding='utf-8')
    print(f'Prepared {DEST} with {sum(p.is_file() for p in DEST.rglob("*"))} files')

if __name__=='__main__':main()
