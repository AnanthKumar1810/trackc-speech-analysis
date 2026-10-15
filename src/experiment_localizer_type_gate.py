"""Test a single-edit type gate using out-of-fold train and validation only.

This experiment asks whether a paired classifier can suppress cross-type
candidate events in this one-edit-per-clip synthetic benchmark. It never loads
the held-out test split and does not change the released localizer.
"""
from collections import defaultdict
from pathlib import Path
import json
import re

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold

from core import loadl, read
from evaluate_localizer import overlap
from localizer import detect
from infer_model import infer_pair, load_model
from model_features import vector

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['none', 'pause', 'quiet', 'rush', 'drag']


def estimator_from_name(name):
    m = re.fullmatch(r'extra_trees_leaf(\d+)_features(sqrt|1\.0)', name)
    if not m:
        raise ValueError(f'Unsupported selected model: {name}')
    return ExtraTreesClassifier(n_estimators=400, min_samples_leaf=int(m.group(1)),
        max_features='sqrt' if m.group(2) == 'sqrt' else 1.0,
        class_weight='balanced', random_state=17, n_jobs=1)


def score(rows):
    positives = [r for r in rows if r['truth']]
    hits = sum(r['hit'] for r in positives)
    predicted = sum(len(r['events']) for r in rows)
    recall = hits / max(1, len(positives))
    precision = hits / max(1, predicted)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {'clips': len(rows), 'positive_clips': len(positives),
            'hits': hits, 'predicted_events': predicted,
            'precision': round(precision, 4), 'recall': round(recall, 4),
            'f1': round(f1, 4)}


def main():
    manifest = [m for m in loadl(ROOT/'metadata/manifest.jsonl')
                if m['split'] in {'train', 'validation'}]
    lookup = {m['clip_id']: m for m in manifest}
    labels = {r['clip_id']: r for r in loadl(ROOT/'metadata/flaw_labels.jsonl')}
    thresholds = json.loads((ROOT/'localizer/thresholds.json').read_text(encoding='utf-8'))['thresholds']
    words = defaultdict(list)
    for r in loadl(ROOT/'alignment_mfa/word_alignment_mfa.jsonl'):
        words[r['clip_id']].append(r)

    audio_cache, feature_cache = {}, {}
    def load(m):
        cid = m['clip_id']
        if cid not in audio_cache:
            audio_cache[cid] = read(ROOT/m['audio_path'])
            feature_cache[cid] = dict(np.load(ROOT/m['feature_path']))
        return audio_cache[cid], feature_cache[cid]

    rows, X = [], []
    for m in manifest:
        base = lookup[m['family_id']+'_base']
        ap, af = load(base)
        bp, bf = load(m)
        X.append(vector(ap, bp, af, bf))
        truth = labels.get(m['clip_id'])
        result = detect(ap, bp, words[base['clip_id']], words[m['clip_id']],
                        same_setup=True, thresholds=thresholds,
                        reference_features=af, take_features=bf)
        rows.append({'manifest': m, 'truth': truth, 'events': result['events']})
    X = np.stack(X)

    experiment = json.loads((ROOT/'model/classifier_experiment_20261015.json').read_text(encoding='utf-8'))
    estimator = estimator_from_name(experiment['selected_models']['type'])
    split = np.array([r['manifest']['split'] for r in rows])
    train_mask, val_mask = split == 'train', split == 'validation'
    y = np.array([CLASSES.index(r['manifest']['flaw_type']) for r in rows])
    groups = np.array([r['manifest']['family_id'] for r in rows])
    probabilities = np.zeros((len(rows), len(CLASSES)), dtype=float)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261015)
    for fit_idx, held_idx in cv.split(X[train_mask], y[train_mask], groups[train_mask]):
        train_rows = np.flatnonzero(train_mask)
        model = estimator.fit(X[train_rows[fit_idx]], y[train_rows[fit_idx]])
        probabilities[train_rows[held_idx]] = model.predict_proba(X[train_rows[held_idx]])
    model = estimator.fit(X[train_mask], y[train_mask])
    probabilities[val_mask] = model.predict_proba(X[val_mask])

    results = []
    for threshold in [round(x, 2) for x in np.arange(0, 1.001, .05)] + [1.01]:
        gated_rows = []
        for i, row in enumerate(rows):
            m, truth = row['manifest'], row['truth']
            prob = probabilities[i]
            pred = CLASSES[int(np.argmax(prob))]
            selected = row['events']
            if float(np.max(prob)) >= threshold:
                selected = [] if pred == 'none' else [e for e in selected if e['type'] == pred]
            hit = False
            if truth:
                gold = (truth['start_seconds'], truth['end_seconds'])
                hit = any(e['type'] == truth['flaw_type'] and
                    overlap((e['start_seconds'], e['end_seconds']), gold) >= .5
                    for e in selected)
            gated_rows.append({'family_id': m['family_id'],
                'split': m['split'], 'truth': truth is not None,
                'hit': hit, 'events': selected})
        tr = [r for r in gated_rows if r['split'] == 'train']
        va = [r for r in gated_rows if r['split'] == 'validation']
        family_f1 = [score([r for r in tr if r['family_id'] == f])['f1']
                     for f in sorted({r['family_id'] for r in tr})]
        results.append({'confidence_threshold': threshold,
            'train': score(tr), 'train_macro_family_f1': round(float(np.mean(family_f1)), 4),
            'validation': score(va)})

    baseline = next(r for r in results if r['confidence_threshold'] == 1.01)
    eligible = [r for r in results if
        r['validation']['f1'] >= baseline['validation']['f1'] and
        r['validation']['precision'] >= baseline['validation']['precision']]
    selected = max(eligible, key=lambda r:(r['train_macro_family_f1'],
                                            r['validation']['f1'],
                                            r['validation']['precision'])) if eligible else None
    # Freeze the gate using train OOF and validation, then evaluate it once on
    # the held-out test speaker. This test result is reporting only.
    gate_threshold = selected['confidence_threshold'] if selected else 1.01
    test_manifest = [m for m in loadl(ROOT/'metadata/manifest.jsonl') if m['split'] == 'test']
    test_lookup = {m['clip_id']: m for m in test_manifest}
    final_model = load_model(ROOT/'model/model.json')
    baseline_test, gated_test = [], []
    test_type_correct = []
    for m in test_manifest:
        base = test_lookup[m['family_id']+'_base']
        ap, af = load(base)
        bp, bf = load(m)
        result = detect(ap, bp, words[base['clip_id']], words[m['clip_id']],
                        same_setup=True, thresholds=thresholds,
                        reference_features=af, take_features=bf)
        pred = infer_pair(ap, bp, model=final_model,
                          base_features=af, variant_features=bf)['type']
        test_type_correct.append(pred['predicted'] == m['flaw_type'])
        truth = labels.get(m['clip_id'])
        gold = (truth['start_seconds'], truth['end_seconds']) if truth else None
        for target, use_gate in ((baseline_test, False), (gated_test, True)):
            events = result['events']
            if use_gate and float(max(pred['probabilities'].values())) >= gate_threshold:
                kind = pred['predicted']
                events = [] if kind == 'none' else [e for e in events if e['type'] == kind]
            hit = bool(truth and any(e['type'] == truth['flaw_type'] and
                overlap((e['start_seconds'], e['end_seconds']), gold) >= .5 for e in events))
            target.append({'family_id':m['family_id'],'truth':truth is not None,
                           'hit':hit,'events':events})
    test_baseline = score(baseline_test)
    test_gated = score(gated_test)
    families = sorted({r['family_id'] for r in baseline_test})
    by_family = {f:[r for r in baseline_test if r['family_id']==f] for f in families}
    gated_by_family={f:[r for r in gated_test if r['family_id']==f] for f in families}
    rng=np.random.default_rng(20261015); draws={'baseline':[],'gated':[]}
    for _ in range(2000):
        chosen=rng.choice(families,size=len(families),replace=True)
        sampled_base=[r for f in chosen for r in by_family[str(f)]]
        sampled_gate=[r for f in chosen for r in gated_by_family[str(f)]]
        draws['baseline'].append(score(sampled_base)['f1'])
        draws['gated'].append(score(sampled_gate)['f1'])
    test_report={'baseline_ungated':test_baseline,'classifier_gated':test_gated,
        'gate_confidence_threshold':gate_threshold,
        'classifier_type_accuracy_on_test':float(np.mean(test_type_correct)),
        'family_bootstrap_f1_95_ci':{k:[round(float(x),4) for x in np.percentile(v,[2.5,97.5])]
                                     for k,v in draws.items()},
        'test_used_for_selection':False,
        'limitation':'Same fixed synthetic test set was reported in earlier work; this combined-pipeline result is exploratory, not an independent benchmark.'}
    report = {'scope':'Train-family out-of-fold predictions and validation were used for selection; the fixed test split was evaluated once after selection.',
        'condition':'One digitally edited operation per clip. A global type gate may suppress valid simultaneous events in natural speech.',
        'classifier':experiment['selected_models']['type'],
        'training_prediction':'5-fold passage-family out-of-fold predictions for Reagan train; classifier fit on Reagan only for Kennedy validation.',
        'baseline_ungated':baseline, 'selected_candidate':selected,
        'best_validation_only':max(results,key=lambda r:(r['validation']['f1'],r['validation']['precision'])),
        'test_evaluation_after_selection':test_report,
        'grid':results}
    out = ROOT/'reports/localizer_type_gate_experiment_20261015.json'
    out.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'baseline_ungated':baseline, 'selected_candidate':selected,
                      'best_validation_only':report['best_validation_only'],
                      'output':str(out)}, indent=2))


if __name__ == '__main__':
    main()
