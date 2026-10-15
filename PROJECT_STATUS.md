# Project status — 15 October 2026

## Combined one-edit benchmark result

- Added a benchmark-only type filter: the paired classifier predicts the
  digitally applied operation, then the event detector retains candidate
  intervals of that type. The rule was selected using passage-family
  out-of-fold train predictions and a validation-speaker guard; the test split
  was not used to choose it.
- On the fixed Obama synthetic test set, the combined pipeline reached
  **65.2% event precision, 60.4% recall and 62.7% F1** at interval IoU ≥ 0.5,
  compared with **42.3%, 60.4% and 49.8%** for the event detector alone. It
  reduced candidate intervals from 137 to 89 while retaining 58 correctly
  localized edits.
- Family-bootstrap 95% interval for combined F1: **50.5–73.3%**, based on
  resampling the eight test passage families. This is not uncertainty across
  speakers.
- The dashboard enables this filter only for the controlled archive, where
  each variant contains one programmed edit. Uploaded takes remain ungated;
  natural speech can contain multiple simultaneous issues.
- The classifier’s results on this same fixed test set were reported earlier.
  Although the gate rule was selected without test results, this combined
  score is exploratory and is not a fresh independent benchmark.

## 15 October detector and dashboard update

- Reworked `evaluation.html` into a responsive analysis dashboard with a clear
  benchmark-to-upload flow, acoustic evidence chart, candidate table, honest
  benchmark metrics, and an explicit synthetic-data / human-validation note.
- Added a 1,024-candidate threshold and event-merging search using Reagan train
  families and Kennedy validation only. The Obama test speaker is not loaded by
  the optimizer. A 0.60-second same-type merge gap and 1.30 drag threshold were
  selected; pause, quiet, and rush thresholds stayed at their previous values.
- On validation, event precision rose from 22.4% to 30.3%, recall from 47.2% to
  55.6%, and F1 from 30.4% to 39.2%. This was the selection/guard set, so it is
  not an independent final result.
- The frozen candidate was evaluated on the held-out Obama speaker once:
  **42.3% precision, 60.4% recall, and 49.8% F1**, compared with the old
  **33.3%, 55.2%, and 41.6%**. Extra or insufficient-overlap events fell from
  106 to 79. Results remain limited to digital edits from one test speaker.
- Updated the dashboard and benchmark documentation to report the new result.
- Re-ran all 18 unit tests and all 4,187 structural dataset checks: passed.
- Built refreshed code-only and public-dataset ZIPs for this detector version.

## 15 October classifier retraining

- Compared logistic regression, RBF SVM, Random Forest, ExtraTrees and histogram
  gradient boosting with five-fold grouped CV on the ten Reagan training
  families. Kennedy validation was a guard; the Obama test split was not read
  by candidate selection.
- Selected separate 400-tree ExtraTrees heads for edit type and programmed
  strength. Refit on Reagan+Kennedy (169 clips / 13 families), then evaluated
  the frozen model on Obama (104 clips / 8 families).
- Test results: **104/104 edit-type accuracy (100.0%, macro F1 1.000)** and
  **101/104 programmed-strength accuracy (97.1%, macro F1 0.977)**. The previous
  logistic baseline scored 101/104 (97.1%) and 76/104 (73.1%), respectively.
- This is the same fixed synthetic test set previously reported for the
  logistic baseline. Candidate selection did not use test results, but this is
  not a new independent benchmark. Natural-speech accuracy is still unknown.
- The NumPy inference implementation reproduced predictions for all 273 clips;
  validation passed. `MODEL_CARD.md` describes the protocol and limitations.

## Built and checked

- Public-speech contrastive dataset: 21 passage families, 273 WAV files, 252
  exact labeled digital edits, canonical transcripts, source and rights records,
  manifests and speaker-disjoint train/validation/test splits.
- Imported MFA TextGrids for all 273 archival clips. Their timing is automatic,
  not human-verified. The 170 alignment-review flags are disagreements with a
  previous aligner, not a measured MFA error count.
- Whole-clip digital-edit classifier, private 22-take human development lab,
  transcript gate, local API and aligned dashboard.
- Word-grounded candidate-event detector with pair-wide tempo and loudness
  normalization, train-family selection with a validation guard, configurable
  same-type event merging, a corrected
  merged-event rubric, arbitrary-pair comparison page, held-out evaluation and
  updated technical report.
- Structural validation: 4,187 checks passed. Unit tests: 18 passed, including
  whole-clip speed/gain nuisance cases and pause/merged-event rubric cases.
- Global nuisance controls: 21 reference clips × 8 transformations produced
  zero false events. Local flaw preservation and cross-speaker performance
  were not tested by this control suite.

## Measured detector results

On the Obama test speaker's 96 digital variants, 58 events were correctly
localized at IoU >= 0.5: **60.4% recall**, **42.3% event precision**, and
**49.8% event F1**. There were 79 extra or insufficient-overlap predictions.
Pause localization was 23/24; quiet 9/24; rush 14/24; drag 12/24. Minor,
moderate and severe recall were 21.9%, 71.9% and 87.5%. A family-bootstrap
95% interval for test F1 is 39.1–61.1%. These are synthetic edit results, not
natural speaking-quality accuracy. The larger 0.60-second merge gap can
combine separate real events and needs human review on natural recordings.

## Work that cannot be claimed complete from the available evidence

1. Verify the spoken words and word boundaries by listening and checking a
   sampled timing sheet. MFA coverage is not timing accuracy.
2. Obtain independent, consented human ratings and flaw intervals for natural
   same-text recordings, ideally from new speakers kept outside development.
   The four current speakers have already influenced development and cannot be
   treated as a blind test set. Do not publish their audio without consent.
3. Re-evaluate the detector on that natural holdout. Its current 42.3% event
   precision makes it a candidate-highlighting prototype rather than a reliable
   judge. A score above 90% is not supported by present evidence.
4. Publish a public code repository and permissible public dataset package,
   then provide the links in the hackathon entry. Check final submission rules.
5. The user will produce the required submission video separately.

## Reproduce this release

Run `python -m unittest discover -s src -p 'test_*.py' -v`,
`python src/validate.py`, `python src/tune_localizer.py`,
`python src/optimize_localizer.py`, `python src/evaluate_localizer.py`,
`python src/make_viewer_data.py`, and
`python src/render_technical_pdf.py`. Then run `python src/serve.py` and open
`http://127.0.0.1:8765/evaluation.html`. Re-running threshold tuning uses
train and validation data only; test is scored after the candidate is frozen.
The `human_pilot/` directory is private and excluded from
the public ZIP.
