# Project status — 14 October 2026

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
  normalization, severity-balanced train-only threshold selection, a corrected
  merged-event rubric, arbitrary-pair comparison page, held-out evaluation and
  updated technical report.
- Structural validation: 4,187 checks passed. Unit tests: 17 passed, including
  whole-clip speed/gain nuisance cases and pause/merged-event rubric cases.
- Global nuisance controls: 21 reference clips × 8 transformations produced
  zero false events. Local flaw preservation and cross-speaker performance
  were not tested by this control suite.

## Measured detector results

On the Obama test speaker's 96 digital variants, 53 events were correctly
localized at IoU >= 0.5: **55.2% recall**, **33.3% event precision**, and
**41.6% event F1**. There were 106 extra or insufficient-overlap predictions.
Pause localization was 23/24; quiet 7/24; rush 12/24; drag 11/24. Minor,
moderate and severe recall were 21.9%, 65.6% and 78.1%. A family-bootstrap
95% interval for test F1 is 30.7–52.7%. Severity balancing improved drag
detection but reduced quiet detection. These are synthetic edit results, not
natural speaking-quality accuracy.

## Work that cannot be claimed complete from the available evidence

1. Verify the spoken words and word boundaries by listening and checking a
   sampled timing sheet. MFA coverage is not timing accuracy.
2. Obtain independent, consented human ratings and flaw intervals for natural
   same-text recordings, ideally from new speakers kept outside development.
   The four current speakers have already influenced development and cannot be
   treated as a blind test set. Do not publish their audio without consent.
3. Re-evaluate the detector on that natural holdout. Its current 33.3% event
   precision makes it a candidate-highlighting prototype rather than a reliable
   judge. A score above 90% is not supported by present evidence.
4. Publish a public code repository and permissible public dataset package,
   then provide the links in the hackathon entry. Check final submission rules.
5. The user will produce the required submission video separately.

## Reproduce this release

Run `python -m unittest discover -s src -p 'test_*.py' -v`,
`python src/validate.py`, `python src/tune_localizer.py`,
`python src/evaluate_localizer.py`, `python src/make_viewer_data.py`, and
`python src/render_technical_pdf.py`. Then run `python src/serve.py` and open
`http://127.0.0.1:8765/evaluation.html`. Re-running threshold tuning uses
train data only. The `human_pilot/` directory is private and excluded from
the public ZIP.
