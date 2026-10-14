# Release v2 - 11 October 2026

## Evaluator update — 14 October 2026

- Added word-grounded candidate flaw intervals, train-only threshold selection,
  speaker-disjoint event evaluation and a reference-relative operational rubric.
- Added `evaluation.html`, upload alignment fallback and current three-page
  technical report. The public package excludes the 22 private human takes.
- On 96 test-speaker digital edits, event recall at IoU >= 0.5 is 54.2%,
  precision 32.9%, and F1 40.9%. These are not natural speaking-quality scores.
- Read `PROJECT_STATUS.md` for remaining human validation and publication work.

Supersedes the earlier SAPI prototype. No audio from that prototype is included.

- 21 unique reference passages, 3 real human speakers.
- 273 actual WAV files and 252 reference-variant pairs.
- 3.44 minutes distinct reference audio; 46.18 minutes including all variants.
- Four edit operators, three strengths each, on every passage.
- Train/validation/test: 10/3/8 families, separated by speaker and source.
- 4187 machine checks passed; every WAV reproduced byte-for-byte on an independent second build.
- Local API smoke test passed, including rejection of mismatched text; browser playback controls and variant selection inspected.
- Trained paired edit-signature model: held-out type accuracy 101/104, programmed-strength accuracy 76/104; see MODEL_CARD.md for limits.
- Technical report PDF included and visually inspected.

Remaining: structured transcript and timing verification/corrections; independent reference/severity/naturalness ratings; diverse consented natural variants; your publication and demo submission. Labels do not claim these tasks were completed.

## User review update - 11 October 2026

The requesting user reported: "I manually review them. they are good". This is recorded as positive general listening feedback in reports/user_review.json. The reviewed clip IDs and detailed judgments were not supplied. Per-clip transcript/word-boundary verification, numerical severity/naturalness ratings and independent agreement remain unrecorded. Existing verification flags and blank rating fields are unchanged. The technical PDF and automated reports describe the earlier build-stage checks; this update records the subsequent user feedback.

Known constraints: only 3.44 minutes of distinct human content; historical English-language male presidential speeches; speaker/channel/era confounded with split; digital edits may be detectable from processing artifacts. This is a pilot for controlled experiments, not a human-validated benchmark.
