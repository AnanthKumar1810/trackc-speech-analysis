"""Compare small paired-audio classifiers without reading the held-out test split.

Uses grouped cross-validation over training passage families, then reports the
separate validation-speaker result as a selection guard. This is an experiment
driver; it does not modify the released model or predictions.
"""
from pathlib import Path
import json
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from core import loadl, read
from model_features import vector

ROOT = Path(__file__).resolve().parents[1]
TYPE = ['none', 'pause', 'quiet', 'rush', 'drag']
STRENGTH = ['none', 'minor', 'moderate', 'severe']


def candidates(seed=17):
    out = []
    for C in [.1, 1., 10., 100.]:
        out.append((f'logistic_C{C:g}', make_pipeline(
            RobustScaler(), LogisticRegression(C=C, max_iter=3000,
            class_weight='balanced', random_state=seed))))
        out.append((f'rbf_svc_C{C:g}', make_pipeline(
            RobustScaler(), SVC(C=C, kernel='rbf', class_weight='balanced'))))
    for family, cls in [('extra_trees', ExtraTreesClassifier), ('random_forest', RandomForestClassifier)]:
        for leaf in [1, 2, 4]:
            for max_features in ['sqrt', 1.0]:
                out.append((f'{family}_leaf{leaf}_features{max_features}', cls(
                    n_estimators=400, min_samples_leaf=leaf,
                    max_features=max_features, class_weight='balanced',
                    random_state=seed, n_jobs=-1)))
    for leaves in [3, 7, 15]:
        out.append((f'hist_gradient_leaves{leaves}', HistGradientBoostingClassifier(
            max_leaf_nodes=leaves, learning_rate=.08, max_iter=100,
            l2_regularization=1., random_state=seed, class_weight='balanced')))
    return out


def score(y, p, class_count):
    return {
        'accuracy': float(accuracy_score(y, p)),
        'macro_f1': float(f1_score(y, p, average='macro', labels=range(class_count), zero_division=0)),
    }


def main():
    manifest = [m for m in loadl(ROOT/'metadata/manifest.jsonl') if m['split'] in {'train', 'validation'}]
    lookup = {m['clip_id']: m for m in manifest}
    cache = {}

    def load(m):
        if m['clip_id'] not in cache:
            cache[m['clip_id']] = (read(ROOT/m['audio_path']), dict(np.load(ROOT/m['feature_path'])))
        return cache[m['clip_id']]

    rows = []
    for m in manifest:
        base = lookup[m['family_id']+'_base']
        ap, af = load(base)
        bp, bf = load(m)
        rows.append({'family': m['family_id'], 'split': m['split'],
                     'type': m['flaw_type'], 'strength': m['severity'],
                     'x': vector(ap, bp, af, bf)})
    X = np.stack([r['x'] for r in rows])
    split = np.array([r['split'] for r in rows])
    groups = np.array([r['family'] for r in rows])
    train = split == 'train'
    validation = split == 'validation'
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261015)

    result = {
        'scope': 'Only Reagan training and Kennedy validation were loaded; Obama test was not read.',
        'cv': '5-fold StratifiedGroupKFold by passage family over the 10 training families.',
        'metrics': {},
    }
    selection = {}
    for head, labels, classes in [('type', TYPE, TYPE), ('strength', STRENGTH, STRENGTH)]:
        y = np.array([classes.index(r[head]) for r in rows])
        ytrain, yval = y[train], y[validation]
        fold_scores = []
        for name, estimator in candidates():
            oof = np.full(len(ytrain), -1, dtype=int)
            for tri, vai in cv.split(X[train], ytrain, groups[train]):
                model = clone(estimator)
                model.fit(X[train][tri], ytrain[tri])
                oof[vai] = model.predict(X[train][vai])
            cv_score = score(ytrain, oof, len(classes))
            fitted = clone(estimator).fit(X[train], ytrain)
            val_pred = fitted.predict(X[validation])
            val_score = score(yval, val_pred, len(classes))
            fold_scores.append({'name': name, 'cv': cv_score, 'validation': val_score,
                                'estimator': estimator})
        fold_scores.sort(key=lambda x: (x['cv']['macro_f1'], x['cv']['accuracy'],
                                        x['validation']['macro_f1']), reverse=True)
        # Prefer grouped-CV performance; the validation result is diagnostic and
        # helps reject clear regressions, rather than selecting on one tiny split.
        baseline = next(x for x in fold_scores if x['name'] == 'logistic_C10')
        guarded = [x for x in fold_scores
                   if x['validation']['macro_f1'] >= baseline['validation']['macro_f1'] - .02]
        chosen = max(guarded or fold_scores,
                     key=lambda x: (x['cv']['macro_f1'], x['cv']['accuracy'],
                                    x['validation']['macro_f1']))
        selection[head] = {'name': chosen['name'], 'estimator': chosen['estimator']}
        result['metrics'][head] = {
            'baseline': {'name': baseline['name'], 'cv': baseline['cv'],
                         'validation': baseline['validation']},
            'selected': {'name': chosen['name'], 'cv': chosen['cv'],
                         'validation': chosen['validation']},
            'candidates': [{'name': x['name'], 'cv': x['cv'], 'validation': x['validation']}
                           for x in fold_scores],
        }
    result['selected_models'] = {k: v['name'] for k, v in selection.items()}
    out = ROOT/'model/classifier_experiment_20261015.json'
    out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['selected_models'], indent=2))
    for head, values in result['metrics'].items():
        print(head, json.dumps({'baseline': values['baseline'], 'selected': values['selected']}, indent=2))
    print('Saved', out)


if __name__ == '__main__':
    main()
