# Dataset card

## Evaluator update (15 October 2026)

`evaluation.html` and `src/localizer.py` provide predicted candidate flaw
intervals and an operational reference-relative rubric. `pair.html` also
compares arbitrary uploaded recordings at a coarse speech-activity level.
Threshold and same-type event-merge selection uses a 1,024-candidate search on
Reagan training passage families, with a Kennedy validation guard. The Obama
test speaker is not loaded during selection. Current Obama test-set event
precision is 42.3%, recall 60.4%, and F1 49.8% at IoU >= 0.5; its
passage-family bootstrap 95% F1 interval is 39.1–61.1%. These controlled
digital-edit results are not natural-speech accuracy. See `PROJECT_STATUS.md`
and the technical report for limitations.

## Purpose and status

A small, English-language controlled contrastive speech dataset for Track C prototyping: compare an authentic public-speaking reference candidate with digitally altered delivery of the same canonical text. Use it for operator localization, paired acoustic comparisons, and pipeline development. It is **not** a validated universal speaking-quality measure, a representative speech corpus, or a collection of naturally occurring mistakes.

The actual inventory and durations are generated in `reports/statistics.json`. Every family contains a human reference and four edit types at three severity levels. Audio uses 16,000 Hz mono PCM16. Each full decoded source is attenuated, only if needed, to a peak of 0.75 before quantization; the gain is recorded per clip. This prevents new clipping while leaving edit headroom, but does not repair distortion already present in an archival recording. Originals are bundled for provenance and offline reproduction.

## Collection and source selection

Sources: Reagan's 1981 inaugural; Obama's 2009 inaugural; two Kennedy presidential speech excerpts from NASA. Published transcripts supply Reagan/Obama canonical sentences. Kennedy text uses the historical quotation corresponding to the excerpts. Exact spoken-word agreement remains unreviewed. Optional ASR locates candidate transcript anchors; PocketSphinx 5.0.4 forced alignment estimates word positions. Short passages passing token coverage, duration and monotonicity checks are selected. Source offset and ASR anchor coverage are retained per crop. These are automatic selection signals, not measured transcript accuracy.

Oath/ceremony lead-ins and unmatched passages are excluded from final crops. Complete originals still contain ceremony material and must not be counted as labeled examples. An uncertain final word in the NASA moon excerpt is excluded by using earlier subpassages. Source recordings retain historical noise, crowd sounds and recording limitations; there is no claim of clean studio capture.

## Generation and severity

| Edit | Minor | Moderate | Severe |
|---|---:|---:|---:|
| Inserted silent pause | 0.25 s | 0.75 s | 1.50 s |
| Local energy reduction | -3 dB | -8 dB | -15 dB |
| Local pace increase | 1.12× | 1.35× | 1.70× |
| Local pace decrease | 0.90× | 0.72× | 0.52× |

“Minor/moderate/severe” are ordered intervention magnitudes, not listener-calibrated scores. References are assigned condition `human_archival_reference_candidate`; variants are `digitally_modified_human_recording`. No TTS or invented human evaluations are present. Pace uses an STFT phase vocoder with a 512-sample window and 128-sample hop plus 10ms edge blending; it targets pitch preservation, but can introduce processing artifacts. Quiet edits use 10ms gain ramps. Inserted zero-valued silence is deliberately artificial. These artifacts can make classification easier than real delivery assessment.

## Timing and labels

All intervals use half-open `[start_sample, end_sample)` coordinates at 16 kHz; divide by 16000 for seconds. Source offsets refer to the downmixed/resampled original. Edit boundaries and inserted sample counts are exact. Word labels originate from a 10ms-frame aligner and are *estimates*: quantization resolution is not demonstrated accuracy. Pace mappings are piecewise linear inside the edit; phase-vocoder phonetic events may deviate. Pause mappings keep the preceding word end before inserted silence. Canonical text, pair membership, cause and severity are separately represented. Human ratings are null and human-verification flags are false.

## Splits and leakage

Reagan = train; Kennedy = validation; Obama = test. All crops from each speaker/source and all variants of every text family remain together. Crops do not overlap within a source. Validation checks speaker/source/family/exact-transcript/exact-audio disjointness, pair membership and cross-split 5-word-shingle overlap. Every family has every operator/severity combination, so operator classes are balanced across splits.

There are only three speakers. Speaker, historical era, venue, recording channel and split are strongly confounded. Validation is very small, especially for selecting thresholds. Scores have high variance and do not establish demographic or domain generalization. Do not mix the full held-out original recordings into training. Do not randomly split the per-WAV manifest. Artificial siblings are not independent observations; evaluate and bootstrap by family/speaker, never individual variant alone.

## Features and normalization

Frames: 40ms window, 20ms hop. Included: RMS dBFS, RMS relative to each clip median, zero-crossing rate, spectral centroid, log FFT power (1024 FFT), 13 orthonormal-DCT MFCCs from 26 triangular mel filters, autocorrelation pitch (roughly 70–350Hz), periodicity and pitch relative to clip median in semitones. Unvoiced/low-periodicity pitch is NaN. Whole-clip words/minute includes pauses and is not articulation rate. Historical applause/music/noise can corrupt pitch and energy. Relative normalization is exploratory and not perceptual calibration; keep raw features as well. Fit any learned scalers using training data only.

## Trained model

The packaged prototype predicts digital edit type and programmed edit strength from a matched reference/recording pair. It is trained on Reagan, tuned on Kennedy and tested on Obama; the one-speaker-per-split design limits conclusions. Test edit-type accuracy is 101/104 and programmed-strength accuracy is 76/104. These are controlled-edit recognition results, not performance on natural speech mistakes or human-rated quality. Details, confusion matrices and inference instructions are in `MODEL_CARD.md` and `model/evaluation.json`.

## Validation and appropriate claims

Read the generated validation report, not a generic assertion of completeness. It checks actual PCM sample counts/hashes, unmodified spans, exact operator reconstruction, measured quiet gain, tone pitch retention, word ranges/token coverage, feature arrays and split grouping. Independent rebuilding checks audio bytes. These do not replace listening review, independent transcript transcription, inter-rater agreement or held-out natural-performance evaluation.

Human reviewers still need to verify transcripts/crop boundaries, correct word timing, assess reference suitability, rate severity/naturalness and adjudicate disagreements. Add consented natural same-text performances and more speakers, languages, styles and recording conditions before making broad claims. Do not use this pilot for hiring, grading people, clinical assessment, identity inference or judgments about a public speaker's real abilities.
