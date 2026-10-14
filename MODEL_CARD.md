# First trained model: digital edit signature

This release includes a trained prototype in `model/model.json`. It predicts the **type of digital edit** (none, pause, quiet, rush, drag) and its **programmed strength** (none, minor, moderate, severe) from a recording paired with the corresponding human reference. It does not predict whether an unaided human listener finds the delivery good, nor does it identify naturally occurring flaws. It requires a suitable reference of the same words. A new participant's independently recorded speech is outside the measured test distribution.

Training recipe: `python src/train_model.py` after the dataset build. The model derives 20 summary measurements from the paired WAVs and frame features. These include total duration change, silent-sample runs, relative energy differences, pitch and spectral changes. No flaw labels, severity settings, edit timestamps, source IDs or split names enter the feature vector. A train-only median and interquartile scale feeds two balanced multinomial logistic regressions. Model selection uses only the Kennedy validation set. The frozen test is Barack Obama. The stored JSON contains coefficients, scaler and class definitions; inference in `src/infer_model.py` needs only NumPy and the dataset's basic audio dependencies. No pickle or TTS system is included.

Results: 10 Reagan families / 130 clips trained; 3 Kennedy families / 39 clips for model selection; 8 Obama families / 104 clips evaluated once. On the 104 held-out clips: edit type accuracy **101/104 = 97.1%**, macro F1 **0.967**; programmed strength accuracy **76/104 = 73.1%**, macro F1 **0.749**. Full confusion matrices and validation results are in `model/evaluation.json`; each clip's result is in `model/predictions.csv`. These denominators count edited siblings and therefore overstate the independent sample size. Only one speaker appears in each split, so speaker, era, channel and split remain confounded.

A self-pair of an exact audio file represents `none` and is easy to recognize. Silence insertion and phase-vocoder artifacts can also make the synthetic edits easy to classify. Performance on natural hesitations, inconsistent real speaking, other microphones or new languages is **unknown**. Probabilities are uncalibrated. Do not present these figures as speaking-quality accuracy.

The local comparison app now displays the model's edit-signature output for a WAV upload that passes its same-transcript alignment check, alongside the existing measured differences. The app explicitly labels it as a digital edit classifier. Human listening and independent ratings remain separate future evaluation work.

The app now runs an independent PocketSphinx ASR pass before word alignment.
If its words disagree with the selected passage, it withholds word-level
feedback until a listener confirms the actual spoken words. ASR disagreement
is a review flag, not proof of a spoken error; ASR agreement is not human
verification. This guard was added after the frozen model evaluation above
and does not change its reported synthetic-edit metrics. See
`EXTRACTION_PIPELINE.md` for the flow and the separate DTW pilot.

Natural-speech evaluation has not been completed. The public package excludes
private participant audio and any private manifest. `src/evaluate_real_speech.py`
can evaluate a future consented, independently labeled holdout, but it requires
new audio and verified labels before it can produce a meaningful result.

To reproduce and inspect: run `python src/train_model.py`, then compare the regenerated `model/model.json`, `model/evaluation.json`, and `model/predictions.csv` to the release. `python src/validate_model.py` checks split separation, training metadata, all model predictions and reported metrics without retraining.
