# Track C: Contrastive Speech Analysis

An offline-first prototype that compares a spoken take with a matching reference and highlights candidate changes in pauses, local pace, and relative loudness. It also includes a reproducible dataset pipeline for 21 public-speech passage families and 252 digitally modified examples. The modified clips are synthetic signal edits, not naturally flawed human performances.

## Try the local demo

Use Python 3.13. Install the core requirements and start the local server:

```powershell
python -m pip install -r requirements.txt
python src/serve.py
```

Open `http://127.0.0.1:8765/pair.html` to compare two WAV recordings you choose. This page reports coarse speech activity, pause, and level differences; it does not provide word-level flaw localization. The `evaluation.html` page supports aligned comparisons from the included archive after the dataset audio is installed. Audio stays on the local machine when using the local server.

## Results and limits

Thresholds were selected on the Reagan training speaker; the Obama speaker is held out for the reported test. On 96 digitally modified test clips, event localization reached **55.2% recall, 33.3% precision, and 41.6% F1** at interval IoU ≥ 0.5. The passage-family bootstrap 95% interval for F1 is 30.7–52.7%. Recall was 21.9% for minor, 65.6% for moderate, and 78.1% for severe edits. These are synthetic-edit results, not accuracy on natural speech. The operational 0–10 rubric is not a human speaking-quality rating.

The detector normalizes local word duration by the pair's median tempo ratio and local loudness by each clip's median level. F0 spread is measured descriptively but monotony detection is not validated. The nuisance-control suite had zero false events on transformed versions of 21 unedited references; it does not test local flaw preservation or prove cross-speaker accuracy. Automatic word boundaries have not been measured against manually verified timing.

## Dataset and rebuilding

This Git repository is code and documentation only. It excludes WAVs, original source media, feature arrays, and all participant recordings. The separate public dataset archive contains the licensed source clips, 273 generated WAV files, labels, and derived features. See `DATASET_CARD.md`, `RIGHTS.md`, and `PUBLISHING.md` before downloading or sharing that archive. The full build and aligned archive dashboard require those dataset files; the arbitrary-pair page can be tried with your own WAVs.

Once the dataset archive is extracted into this project directory, run:

```powershell
python -m pip install -r requirements.txt
python src/build.py
python src/validate.py
python -m unittest discover -s src -p 'test_*.py' -v
python src/tune_localizer.py
python src/evaluate_localizer.py
python src/make_viewer_data.py
python src/render_technical_pdf.py
```

Run `python src/stress_test_nuisance.py` to reproduce the global nuisance negative controls. Run `python src/serve.py` and open `http://127.0.0.1:8765/evaluation.html` to use the archive evaluator. The separate `requirements-human.txt` adds optional transcription and audio-decoding tools; `requirements-model.txt` is for the digital-edit classifier.

## Project contents

- `src/`: dataset construction, validation, feature extraction, alignment, local comparison, evaluation, and serving code.
- `metadata/`, `splits/`, and `transcripts/`: archive manifests, edit labels, source provenance, canonical text, and leakage-resistant split assignments.
- `localizer/` and `model/`: frozen thresholds, predictions, and evaluation outputs for the rule-based detector and digital-edit classifier.
- `alignment_mfa/`: imported automatic alignment results and workflow status. Coverage is complete; timing accuracy is not human-verified.
- `reports/`: dataset statistics, validation, reproducibility, and nuisance-control records.
- `evaluation.html`, `pair.html`, and `review.html`: local browser interfaces.

The 22 recordings from four private development participants are excluded because public redistribution consent is not documented. They have influenced development and are not a blind test set. Natural-speech accuracy and independent human ratings remain unmeasured. See `PROJECT_STATUS.md` and `TECHNICAL_REPORT_2026-10-14.pdf` for details.
