"""Compare detector thresholds and event merge gaps using train/validation only.

The held-out test speaker is deliberately never loaded by this script. Select a
candidate on train-family behavior, use validation as a guard, then freeze it
before running evaluate_localizer.py for the final test report.
"""
from collections import defaultdict
from itertools import product
from pathlib import Path
import json
import numpy as np

from core import loadl, read
from evaluate_localizer import overlap
from localizer import DEFAULTS, detect, _merge

ROOT = Path(__file__).resolve().parents[1]
BASELINE_THRESHOLDS = {'pause_excess_s': .25, 'quiet_db': -6.0,
                       'rush_ratio': .82, 'drag_ratio': 1.22,
                       'merge_gap_s': .12}


def metrics(rows):
    positives = [r for r in rows if r['truth']]
    tp = sum(r['hit'] for r in positives)
    predictions = sum(len(r['events']) for r in rows)
    recall = tp / max(1, len(positives))
    precision = tp / max(1, predictions)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    negatives = [r for r in rows if not r['truth']]
    by_type = {}
    for kind in ('pause', 'quiet', 'rush', 'drag'):
        class_rows = [r for r in rows if r['truth_type'] == kind]
        class_tp = sum(r['hit'] for r in class_rows)
        class_predictions = sum(e['type'] == kind for r in rows for e in r['events'])
        by_type[kind] = {
            'positive_clips': len(class_rows),
            'true_positive_events': class_tp,
            'predicted_events': class_predictions,
            'precision': round(class_tp/max(1, class_predictions), 4),
            'recall': round(class_tp/max(1, len(class_rows)), 4),
        }
    return {'clips': len(rows), 'positive_clips': len(positives),
            'true_positive_events': tp, 'predicted_events': predictions,
            'precision': round(precision, 4), 'recall': round(recall, 4),
            'f1': round(f1, 4),
            'negative_clips_with_alert': sum(bool(r['events']) for r in negatives),
            'by_type': by_type}


def main():
    manifest = [r for r in loadl(ROOT/'metadata/manifest.jsonl')
                if r['split'] in ('train', 'validation')]
    labels = {r['clip_id']: r for r in loadl(ROOT/'metadata/flaw_labels.jsonl')}
    words = defaultdict(list)
    for r in loadl(ROOT/'alignment_mfa/word_alignment_mfa.jsonl'):
        words[r['clip_id']].append(r)

    # Load only train and validation audio. The test speaker is not touched.
    signal_cache = {}
    rows_by_split = defaultdict(list)
    for m in manifest:
        clip = m['clip_id']
        if clip not in signal_cache:
            signal_cache[clip] = (read(ROOT/m['audio_path']),
                                  dict(np.load(ROOT/m['feature_path'])))
        truth = labels.get(clip)
        base = m['family_id']+'_base'
        bx, bf = signal_cache[base]
        ux, uf = signal_cache[clip]
        # Precompute each word's candidate evidence once; the grid then only
        # applies thresholds and event merging. This keeps the experiment fast.
        common = {'pause_excess_s': -1e6, 'quiet_db': 1e6,
                  'merge_gap_s': -1.0}
        all_rush = detect(bx, ux, words[base], words[clip], same_setup=True,
                          thresholds={**common, 'rush_ratio': 10., 'drag_ratio': 1e6},
                          reference_features=bf, take_features=uf)['events']
        all_drag = detect(bx, ux, words[base], words[clip], same_setup=True,
                          thresholds={**common, 'rush_ratio': -1e6, 'drag_ratio': -1.},
                          reference_features=bf, take_features=uf)['events']
        candidates = [e for e in all_rush if e['type'] in ('pause', 'quiet', 'rush')]
        candidates += [e for e in all_drag if e['type'] == 'drag']
        rows_by_split[m['split']].append({'manifest': m, 'truth': truth,
                                          'candidates': candidates})

    grids = product(
        (.16, .25, .35, .45),       # extra pause duration, seconds
        (-4., -6., -8., -10.),      # local level difference, dB
        (.70, .75, .82, .88),       # normalized local word duration, rush
        (1.10, 1.18, 1.22, 1.30),   # normalized local word duration, drag
        (.12, .25, .40, .60),       # merge same-type flagged spans, seconds
    )
    candidates = []
    for pause, quiet, rush, drag, merge_gap in grids:
        t = {'pause_excess_s': pause, 'quiet_db': quiet,
             'rush_ratio': rush, 'drag_ratio': drag,
             'merge_gap_s': merge_gap}
        scored = {}
        for split in ('train', 'validation'):
            split_rows = []
            for item in rows_by_split[split]:
                m, truth = item['manifest'], item['truth']
                ev = []
                for e in item['candidates']:
                    value = e['measurement']
                    if e['type'] == 'pause' and value >= pause:
                        ev.append(e)
                    elif e['type'] == 'quiet' and value < quiet:
                        ev.append(e)
                    elif e['type'] == 'rush' and value < rush:
                        ev.append(e)
                    elif e['type'] == 'drag' and value > drag:
                        ev.append(e)
                ev = _merge(ev, max_gap=merge_gap)
                hit = False
                if truth:
                    gold = (truth['start_seconds'], truth['end_seconds'])
                    hit = any(e['type'] == truth['flaw_type'] and
                              overlap((e['start_seconds'], e['end_seconds']), gold) >= .5
                              for e in ev)
                split_rows.append({'family_id': m['family_id'],
                                   'truth': truth is not None,
                                   'truth_type': truth['flaw_type'] if truth else 'none', 'hit': hit,
                                   'events': ev})
            scored[split] = split_rows

        train = metrics(scored['train'])
        val = metrics(scored['validation'])
        family_f1 = []
        for family in sorted({r['family_id'] for r in scored['train']}):
            family_f1.append(metrics([r for r in scored['train']
                                      if r['family_id'] == family])['f1'])
        candidates.append({'thresholds': t, 'train': train, 'validation': val,
                           'train_macro_family_f1': round(float(np.mean(family_f1)), 4)})

    baseline_t = dict(BASELINE_THRESHOLDS)
    baseline = next((c for c in candidates if c['thresholds'] == baseline_t), None)
    if baseline is None:
        # DEFAULTS' merge gap is .12, and the other values are explicitly in the grid.
        baseline = candidates[0]
    # Candidate selection prioritizes family-macro training F1. Validation is
    # an independent guard: only retain candidates that improve both F1 and
    # precision over today's validation baseline, then maximize F1, precision.
    baseline_val = baseline['validation']
    eligible = [c for c in candidates
                if c['validation']['f1'] >= baseline_val['f1'] and
                   c['validation']['precision'] >= baseline_val['precision']]
    eligible.sort(key=lambda c:(c['train_macro_family_f1'],
                                c['validation']['f1'],
                                c['validation']['precision']), reverse=True)
    ranked_val = sorted(candidates,
                        key=lambda c:(c['validation']['f1'],
                                      c['validation']['precision']), reverse=True)
    selected = eligible[0] if eligible else None
    result = {
        'scope': 'Only Reagan train and Kennedy validation splits were loaded; Obama test split was not accessed.',
        'selection_rule': 'Grid-search threshold and same-type merge-gap candidates. Rank by mean family F1 on train; validation is a guard requiring no regression in either F1 or precision. No candidate was chosen using test results.',
        'grid_size': len(candidates),
        'baseline_validation': baseline_val,
        'best_validation_only': ranked_val[0],
        'best_train_family_candidate': selected,
        'eligible_improving_both': len(eligible),
        'top_validation_candidates': ranked_val[:20],
    }
    out_dir = ROOT/'localizer'
    out_dir.mkdir(exist_ok=True)
    out = out_dir/'tuning_experiment_20261015.json'
    out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    if selected:
        chosen = selected['thresholds']
        selection = {
            'selection': '1,024-candidate grid; selected by mean train family F1, with a validation-speaker guard requiring F1 and precision not below the fixed pre-experiment baseline. Test speaker was not used.',
            'grid_size': len(candidates), 'thresholds': chosen,
            'baseline_validation': baseline_val,
            'selected_train': selected['train'],
            'selected_train_macro_family_f1': selected['train_macro_family_f1'],
            'selected_validation': selected['validation'],
            'test_used_for_selection': False,
        }
        (out_dir/'thresholds.json').write_text(
            json.dumps(selection, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in
                      ('grid_size','baseline_validation','best_validation_only',
                       'best_train_family_candidate','eligible_improving_both')}, indent=2))
    print('Detailed candidate table:', out)


if __name__ == '__main__':
    main()
