# Track C: contrastive speech analysis pilot

This package contains 21 public-speech passage families and 273 WAV files: 21 reference clips and 252 labeled digital edits. Each family has a canonical transcript, automatic word timings, source and rights records, and a fixed train/validation/test assignment. The edited clips are signal-processed examples, not recordings of people naturally making these delivery changes.

## Try the evaluator

Install the listed dependencies, then run `python src/serve.py` and open **http://127.0.0.1:8765/evaluation.html**. The aligned archive evaluator compares a take with a matching reference and shows candidate regions, word-level measurements, and pitch/energy curves. It accepts a WAV upload and transcript; listen and verify the actual words first.

Open **http://127.0.0.1:8765/pair.html** to compare two arbitrary recordings. This estimates speech timing, longer pauses, and recorded level without requiring an archive passage. It does not provide word-level localization. Both pages process audio through the local server; recordings are not sent to a cloud service. The generated measurements are candidates, not a validated speaking-quality score.

## Reproduce and test

Use Python 3.13. Install `requirements.txt`; use `requirements-human.txt` for optional VAD and broader audio decoding. The core build is offline:

```text
python -m pip install -r requirements.txt
python src/build.py
python src/validate.py
python -m unittest discover -s src -p 'test_*.py' -v
python src/tune_localizer.py
python src/evaluate_localizer.py
python src/experiment_classifier.py
python src/retrain_model.py
python src/evaluate_retrained_model.py
python src/validate_model.py
python src/make_viewer_data.py
python src/serve.py
```

The classifier pipeline compares logistic, support-vector, and tree models
using five-fold grouped cross-validation on training passage families and a
validation-speaker guard. It then refits the selected type and strength models
on train plus validation and evaluates the frozen test speaker. The current
synthetic-edit classifier reaches 104/104 type accuracy and 101/104 programmed
strength accuracy on that fixed test set. Read `MODEL_CARD.md`: these scores
measure recognition of generated digital operations, not natural speech.
`python src/train_model.py` runs the full classifier pipeline. Do not use the
already inspected test set to select future models.

First run `python src/tune_localizer.py` for the train-only baseline, then `python src/optimize_localizer.py` to compare thresholds and same-type event merging on Reagan train and Kennedy validation only. The optimizer never loads the Obama test speaker. `python src/evaluate_localizer.py` evaluates the frozen selection on that held-out test. At interval IoU ≥ 0.5, the current test result is **60.4% event recall, 42.3% event precision, and 49.8% F1**. The passage-family bootstrap 95% interval for F1 is 39.1–61.1%. At IoU ≥ 0.3 recall is 67.7%; with a ±200 ms onset/offset collar, recall is 47.9% and precision 33.6%. Minor recall is 7/32, moderate 23/32, and severe 28/32. These are results on synthetic edits, not accuracy on natural speech. The 0–10 rubric is an operational score, not listener preference or MOS.

The detector normalizes local word duration by the pair's median tempo ratio and local loudness by each clip's median level. This reduces sensitivity to uniform speaker-rate and recording-gain differences. It also means a uniform whole-clip rate or gain change is reported as a global difference, not localized as a word flaw. Current thresholds are 0.25 seconds for excess pause, -6 dB for local level, 0.82 for normalized rush ratio and 1.30 for drag ratio; same-type regions separated by at most 0.60 seconds are merged. This candidate improved the validation and held-out synthetic results, but the wide merge gap may combine separate events in natural speech and requires human review. F0 spread is descriptive; monotony detection is not validated. Word boundaries and F0 are automatic estimates.

Run `python src/stress_test_nuisance.py` for global speed, gain, pitch, stationary-noise, and phone-band negative controls on all 21 unedited references. `localizer/nuisance_controls.json` reports false alerts; it does not test whether edited flaws survive those transformations or prove cross-speaker accuracy.

## Data, privacy, and limits

- `audio/`, `transcripts/`, `metadata/`, `splits/`, `features/`, and `sources/` contain the dataset, labels, provenance, and derived measurements.
- `src/` contains generation, validation, feature extraction, model, alignment, and local app code. `model/` contains a separate digital-edit classifier; read `MODEL_CARD.md` before interpreting its results.
- `alignment_mfa/` contains automatic MFA TextGrids. Coverage is complete, but timing accuracy has not been scored against manually checked boundaries.
- The 22 recordings from four private development participants are excluded from this public package because public redistribution consent is not documented. Their results are not a blind test set.

Source and license details are in [RIGHTS.md](RIGHTS.md) and [DATASET_CARD.md](DATASET_CARD.md). Never treat automatic alignment, ASR agreement, or exact edit labels as human verification. Natural-speech accuracy, independent ratings, cross-speaker validation, and human timing accuracy remain unmeasured. Read [PROJECT_STATUS.md](PROJECT_STATUS.md), [MODEL_CARD.md](MODEL_CARD.md), and the updated `TECHNICAL_REPORT.md` for current results. The included PDF predates the latest classifier retraining and records its previous logistic baseline.

## AI assistance disclosure

OpenAI Codex and ChatGPT assisted with code drafting and refinement, debugging, dataset-processing and evaluation scripts, and documentation. The team is responsible for the submitted implementation and claims; the automated checks and their scope are documented in `reports/` and `PROJECT_STATUS.md`. The natural-speech pilot remains exploratory and has not been human-labeled for accuracy.

## Built with

Python, NumPy, SciPy, SoundFile, HTML, CSS, JavaScript, Montreal Forced Aligner, PocketSphinx, OpenAI Codex, and ChatGPT.

## Dataset files

The repository intentionally excludes WAV recordings, extracted feature arrays, original source media, and participant recordings. The paired-upload page works with files you select locally. The archive evaluator and full dataset rebuild require the separate public dataset package described in `PUBLISHING.md`. Private human-participant recordings and their identifying filenames are not part of this repository.
