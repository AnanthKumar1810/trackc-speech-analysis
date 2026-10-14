# MFA alignment status — 2026-10-13

The user supplied `D:\TrackC_MFA_TextGrids.zip` (SHA-256
`635E231A02D7270D99A0B1B8DB452FA77423A87B5EB34250336F14E93A7298A0`).
The ZIP contained 273 distinct, safely named TextGrids matching all 273
prepared clip IDs. `src/mfa_workflow.py import` accepted **273/273** files and
produced word and phone alignment tables.

- MFA clip coverage: **100%** (273/273)
- Clips flagged for >100 ms disagreement with the older automatic aligner: **170**
- Clips still needing transcript/listening review: **273**
- Human-checked word-boundary accuracy: **unavailable** (0 checked gold rows)

The largest clip-level boundary disagreement was 1,998.44 ms, in
`obama_018_drag_severe`; this is a review priority, not an MFA error rate.
MFA output and the old PocketSphinx estimate are both automatic. A matching
TextGrid word sequence verifies file consistency, not what was spoken.

The user also supplied the completed Colab notebook. It shows successful
`mfa validate` and `mfa align` execution, an explicit request for MFA 3.4.3,
and downloads of `english_mfa` and `english_us_mfa`. The notebook did not
retain the printed `mfa version` output or model revision/hash information.
See `run_environment.json`, `alignment_report.json`, `quality_flags.jsonl`,
and `IMPORT_AUDIT.md`.

A frozen six-clip, 171-word manual timing pilot is prepared in
`pilot_gold_selection.json` and `pilot_gold_template.jsonl`. No checked rows
exist yet. `timing_review.html` collects direct-listening boundaries; its
scorer rejects an incomplete pilot. The separate `real_evaluation/` folder
contains an empty manifest template because no independent speaker recordings
or verified real-flaw labels have been supplied.
