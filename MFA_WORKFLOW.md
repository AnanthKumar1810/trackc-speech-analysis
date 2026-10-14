# MFA word alignment workflow

This is an **offline annotation stage**. It does not overwrite the published
PocketSphinx estimates or the exact sample-based edit labels. All MFA outputs
remain automatic until a human checks them.

## 1. Prepare corpus

From the package root, run `python src/mfa_workflow.py prepare`. This copies
all 273 released WAV files into `alignment_mfa/corpus/<speaker>/` and writes
paired `.lab` transcripts. It also writes `input_manifest.jsonl` and a
preflight report. Original canonical scripts are used provisionally. If a
speaker said something different, put the **actual spoken words** in
`transcripts_spoken/<clip_id>.txt` and rerun prepare. Do not silently edit the
canonical script. For our existing digital variants, words are intended to
match the baseline, but baseline transcripts have not been word-verified.

## 2. Install and run MFA

Use MFA's official Miniforge/conda installation instructions on a computer
with internet access. Pin and record the MFA version, acoustic model version,
dictionary version, their hashes, and the complete command in
`alignment_mfa/run_environment.json`. Confirm command flags with
`mfa align --help` for the installed version. The MFA 3.x legacy command is:

```text
mfa validate alignment_mfa/corpus DICTIONARY --acoustic_model_path ACOUSTIC_MODEL
mfa align alignment_mfa/corpus DICTIONARY ACOUSTIC_MODEL alignment_mfa/raw --output_format long_textgrid
```

Resolve OOVs and validation failures before alignment. Review any generated
pronunciations. Keep `alignment_mfa/raw` and MFA's analysis/log files. Do not
interpret an acoustic score as a calibrated accuracy percentage. MFA 3.4+
also offers a newer model/command format; do not mix formats without pinning.

## 3. Import and review

```text
python src/mfa_workflow.py import --aligned alignment_mfa/raw
```

The importer rejects missing/invalid outputs and word-sequence mismatches.
It writes `word_alignment_mfa.jsonl`, `phone_alignment_mfa.jsonl`,
`quality_flags.jsonl`, and `alignment_report.json`. Disagreement with old
PocketSphinx times over 100 ms is a **review trigger**, not an error metric.
Check baseline and edited clips across all speakers and edit severities by
listening and in Praat. The exact digital edit positions are already known;
word and phone positions remain estimates. Silence/VAD may flag pause
disagreements, but should never automatically override those exact digital
edit labels or become a human-verified speech boundary.

## 4. Accuracy measurement

Run `python src/mfa_workflow.py template` to generate a 20-clip selection and
`alignment_mfa/manual_gold_template.jsonl`. For every word in those clips,
listen and mark its start/end in seconds. Fill the blank `start_seconds` and
`end_seconds` fields and set `human_verified` to `true` only after checking
that word; save the completed file as `alignment_mfa/manual_gold_checked.jsonl`.
The sample includes all three speakers and baseline/minor/moderate/severe
conditions. Only then compute median and 90th-percentile absolute start
and end errors in milliseconds, plus the fraction within a declared tolerance.
Do not call corpus coverage, edit detection, or agreement with another
automatic aligner "word-boundary accuracy." No human gold boundaries are in
this release, so measured MFA boundary accuracy is currently unavailable.
Once checked boundaries exist, run
`python src/mfa_workflow.py score --gold alignment_mfa/manual_gold_checked.jsonl` to
produce `alignment_mfa/gold_accuracy_report.json` with median error, 90th
percentile error, and the fraction of boundaries within 50 ms. This report
uses the checked word rows only; it does not label unchecked clips accurate.

## Known constraints

The baseline set contains 21 passages and 3.44 minutes of unique speech.
Speaker adaptation may help or hurt on such small per-speaker material; test
both settings on the checked sample before selecting one. Keep the severe
1.70x rush and 0.52x drag variants as stress tests even if alignment fails.
Do not tune edits to make an aligner look better. Live dashboard alignment
requires the actual spoken transcript or an explicit transcript-correction
step; the supplied script alone cannot detect omitted words.

Official reference: https://montreal-forced-aligner.readthedocs.io/en/stable/user_guide/workflows/alignment.html

Validation: https://montreal-forced-aligner.readthedocs.io/en/stable/user_guide/data_validation.html
