# Track C: Contrastive Speech Analysis

An offline-first prototype for comparing a spoken take with a reference recording of the **same words**. It measures timing and acoustic differences, highlights candidate regions for review, and includes a reproducible benchmark built from public archival speeches and controlled digital edits.

> **Scope:** This is a research prototype, not a validated speech-quality judge. Its reported benchmark results are from digitally modified archival recordings. They do not establish accuracy on naturally occurring speech or human speaking quality.

## What the project does

The system has two related components:

1. **Reference-relative event detector.** It aligns the reference and take by transcript, compares word duration, pauses and recorded energy, and displays candidate time regions with their acoustic evidence. The dashboard also presents pitch and energy measurements where available. A candidate is a prompt for listening and review, not a confirmed flaw.
2. **Digital-edit signature classifier.** A separate paired-audio model predicts which known signal edit was applied—pause insertion, local level reduction, faster delivery or slower delivery—and its programmed strength. It is evaluated only on the controlled synthetic-edit task described below.

The feature pipeline uses short-time spectral measurements, MFCCs, estimated fundamental frequency (F0), RMS energy, word durations and inter-word gaps. Forced alignment supplies estimated word boundaries. The detector uses acoustic differences to propose intervals; exact edit labels are used for evaluation, not as model inputs at inference.

## Results

All figures below are measured on the **fixed Obama-speaker test split of digitally edited recordings**. The test contains eight passage families (104 total clips); the event detector is scored on the 96 edited variants. No natural-speech accuracy or human-rated quality score has been established.

| Evaluation task | Held-out result | How to interpret it |
|---|---:|---|
| Digital-edit type classification | **104/104 correct (100.0% accuracy; macro F1 1.000)** | Recognizes the known edit operation in this controlled dataset. |
| Programmed edit-strength classification | **101/104 correct (97.1% accuracy; macro F1 0.977)** | Predicts the intervention level used to generate an edit; strength levels are not listener ratings. |
| Event localization, interval IoU ≥ 0.5 | **42.3% precision, 60.4% recall, 49.8% F1** | Finds intervals overlapping the known synthetic edit region. There were 79 extra or insufficient-overlap predictions. |
| Event-localization F1 uncertainty | **39.1–61.1% family-bootstrap 95% interval** | Uncertainty across the eight held-out passage families. |

The classifier’s previous logistic baseline scored 97.1% edit-type accuracy and 73.1% strength accuracy on this same test set. The newer ExtraTrees model was selected without using test outcomes, but this is **not a new independent benchmark**: the test set had already been used to report the baseline. The localizer improved from 33.3% precision, 55.2% recall and 41.6% F1 to the results shown above after threshold selection on training and validation data.

These high classifier scores concern recovery of **programmed digital operations**, not whether a real person speaks clearly, naturally or effectively. The localizer’s precision means a considerable number of highlighted regions may be false alarms. See [MODEL_CARD.md](MODEL_CARD.md), [PROJECT_STATUS.md](PROJECT_STATUS.md) and the [technical report](TECHNICAL_REPORT_2026-10-15.pdf) for protocols and caveats.

## Run the demo

Use Python 3.13. From the repository directory, install the core dependencies and start the local server:

```powershell
python -m pip install -r requirements.txt
python src/serve.py
```

Then open:

- `http://127.0.0.1:8765/pair.html` to compare two WAV files. This page provides coarse speech-activity, pause and recorded-level comparisons; it does not provide word-level flaw localization.
- `http://127.0.0.1:8765/evaluation.html` to inspect an aligned same-text comparison and the archived benchmark. The full archive view requires the separate dataset package.

The pages run through the local server; uploaded audio is not sent to a cloud service by this application. Listen to both recordings and confirm the transcript before interpreting word-level feedback. Different microphones, gain control, background noise or spoken words can make comparisons unreliable.

## Dataset

The benchmark contains **273 mono 16 kHz WAV clips** from 21 public-speech passage families: 21 reference clips and 252 digitally modified variants. Each passage has four edit types at three programmed strengths. Edit intervals and sample mappings are recorded precisely; severity names describe edit magnitude, not human judgments. The canonical transcripts and word timings are available, but the word timings are automatic estimates and have not been measured against a human-verified timing set.

The split is by speaker and passage family:

| Split | Speaker | Passage families | Clips |
|---|---|---:|---:|
| Train | Reagan | 10 | 130 |
| Validation | Kennedy | 3 | 39 |
| Test | Obama | 8 | 104 |

No passage family crosses splits. Because each split uses a different speaker, recording channel, era and speaker are confounded; this design does not establish broad speaker generalization. Variants from the same reference passage are related examples, not independent speakers.

The repository contains source code, metadata, model artifacts, reports and interfaces, but **does not include the WAV dataset or original source recordings**. The separate dataset archive and its redistribution details are described in [PUBLISHING.md](PUBLISHING.md), [RIGHTS.md](RIGHTS.md) and [DATASET_CARD.md](DATASET_CARD.md). Private recordings from local participants are excluded because public redistribution consent is not documented. They are development material, not a blind test set.

## Reproduce checks and experiments

Install the project dependencies first. The full build, archive evaluation and model validation require the separate dataset package to be installed in this directory.

Run the unit tests:

```powershell
python -m unittest discover -s src -p 'test_*.py' -v
```

With the dataset available, rebuild and validate the archive, then evaluate the frozen detector and classifier:

```powershell
python src/build.py
python src/validate.py
python src/tune_localizer.py
python src/optimize_localizer.py
python src/evaluate_localizer.py
python src/experiment_classifier.py
python src/retrain_model.py
python src/evaluate_retrained_model.py
python src/validate_model.py
python src/make_viewer_data.py
```

Threshold selection uses Reagan training passage families and Kennedy validation data; the Obama test speaker is held out until evaluation. Do not use this already inspected test set to select future models. For a new evaluation, add an independently collected speaker-disjoint test set with consent and verified labels.

## Repository guide

- `src/` — dataset construction, alignment, feature extraction, modeling, evaluation and local server.
- `evaluation.html`, `pair.html`, `review.html` — browser interfaces served locally.
- `metadata/`, `transcripts/`, `splits/` — manifests, canonical text and split assignments.
- `model/`, `localizer/` — classifier artifacts, detector settings, predictions and evaluation outputs.
- `alignment_mfa/` — imported automatic Montreal Forced Aligner (MFA) TextGrids. Coverage does not imply timing accuracy.
- `reports/` — validation and reproducibility reports.
- `TECHNICAL_REPORT_2026-10-15.pdf` — detailed methods, evaluation and limitations.

## Limitations and next validation work

- Natural-speech event detection has not been evaluated against independently human-labeled intervals.
- No independent human ratings calibrate the programmed severity levels or the dashboard’s operational score.
- Word boundaries and F0 are automatic estimates; their accuracy has not been established here.
- Only three archival speakers are represented, and recording conditions differ across the data splits.
- Quietness is a recorded-level comparison and can be confounded by microphone distance, input gain and automatic gain control.
- Dynamic time warping (DTW) can help transfer timing only within a trusted same-transcript comparison; it cannot verify that the words match.

The intended next step is a consented, speaker-diverse natural-speech evaluation with independently checked transcripts, word boundaries, flaw intervals and severity ratings. Until then, use the system to explore controlled comparisons and support human review—not to make high-stakes judgments about speakers.

## AI assistance

OpenAI Codex and ChatGPT assisted with code drafting and refinement, debugging, data-processing and evaluation scripts, and documentation. The project team is responsible for the implementation and all submitted claims. Automated checks and their scope are recorded in the reports and project documentation.

## Built with

Python, NumPy, SciPy, SoundFile, HTML, CSS, JavaScript, Montreal Forced Aligner, PocketSphinx, OpenAI Codex and ChatGPT.
