# Model card: paired digital-edit signature classifier

## Purpose and limits

The released classifier predicts which **known digital operation** was applied to a recording (none, pause, quiet, rush, or drag) and the **programmed edit strength** (none, minor, moderate, or severe). It compares a recording with a reference containing the same words. It does not judge whether a person sounds clear, natural, confident, or good, and it does not yet detect naturally occurring delivery problems. A participant recording without a matching reference is outside the evaluated use case.

## Data and split design

The dataset has 21 source-passage families and 273 clips. Candidate models were compared using five-fold stratified group cross-validation over the ten Reagan training families. Each fold held complete passage families out. The Kennedy validation speaker served as a regression guard and helped select model settings. The selected models were then refit on the combined 13 train and validation families (169 clips). The Obama test speaker (eight families, 104 clips) was not loaded during candidate comparison or fitting and was evaluated only after the model was frozen. No passage family crosses splits.

This is not a broad speaker-generalization benchmark: one speaker is assigned to each split, so speaker, recording channel, era, and split are confounded. The test clips are siblings made from eight base passages; 104 clips do not represent 104 independent speakers or passages. The same fixed test set was previously used to evaluate the earlier logistic baseline. The new candidate was chosen without using test outcomes, but this is not a new independent benchmark; future iterations need a fresh external speaker holdout.

## Model and fitting

The feature vector has 20 paired-audio summaries: duration and silence changes, relative energy shape, pitch, spectral centroid, zero-crossing rate, voiced fraction, and MFCC summary differences. It does not include edit labels, programmed strength, edit timestamps, source IDs, or split names. Two ExtraTrees classifiers were selected independently: 400 trees for edit type (minimum leaf size 1, square-root feature sampling) and 400 for strength (minimum leaf size 2, all features considered). Candidate selection prioritized grouped training-family macro-F1, with the validation-speaker macro-F1 kept within two points of the logistic baseline when a qualifying candidate existed. The final estimators were fitted on train plus validation data.

`model/model.json` stores the tree arrays and class order. `src/infer_model.py` executes them using NumPy; scikit-learn is needed for training, not inference. Model probabilities are not calibrated.

## Results

On the fixed 104-clip synthetic test set:

| Prediction target | Accuracy | Macro F1 | Previous logistic baseline |
|---|---:|---:|---:|
| Digital edit type | 104/104 = **100.0%** | **1.000** | 101/104 = 97.1%; macro F1 0.967 |
| Programmed edit strength | 101/104 = **97.1%** | **0.977** | 76/104 = 73.1%; macro F1 0.749 |

Strength errors were two moderate edits predicted as minor and one severe edit predicted as moderate. In grouped cross-validation on the ten training families, strength macro-F1 was 0.911 for the selected ExtraTrees model versus 0.738 for the logistic baseline. The Kennedy validation strength macro-F1 was 0.979 versus 0.809. These improvements concern recovery of the dataset's known edit labels, not human judgments.

## Interpretation and risks

Very high test scores are plausible because the task is to identify controlled signal-processing operations, not subtle natural behavior. Exact self-pairs make the `none` class easy. Pause insertion, time-stretch artifacts, and large local amplitude changes may expose the operation. A new microphone, natural speech, different language, mismatched reference, codec, or real spontaneous delivery may change performance sharply. Do not report these figures as natural-speech accuracy, speech-quality accuracy, or human preference.

## Reproduction

Install `requirements-model.txt` and run:

```text
python src/train_model.py
python src/validate_model.py
```

The training command compares models using train and validation only, refits the selected models on those splits, then evaluates the frozen test set. The validation script re-runs inference for all 273 clips and verifies the released test metrics and confusion matrices. Do not use this already inspected test set to tune another candidate; acquire a new, speaker-disjoint natural-speech holdout for future evaluation.
