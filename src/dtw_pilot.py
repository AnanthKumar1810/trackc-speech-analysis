"""Exploratory MFCC/DTW timing comparison on known digital edits.

This does not produce public flaw labels.  Inspect validation results before
considering it for the live feedback model.
"""
import argparse
import json
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def path_for(reference, variant):
    # Exclude coefficient zero (mostly total spectral energy). Normalize each
    # dimension jointly so large-valued coefficients do not dominate the path.
    a, b = np.asarray(reference[:, 1:13], float), np.asarray(variant[:, 1:13], float)
    scale = np.maximum(np.std(np.vstack((a, b)), axis=0), 1e-3)
    a = np.clip((a - np.mean(a, axis=0)) / scale, -5, 5)
    b = np.clip((b - np.mean(b, axis=0)) / scale, -5, 5)
    n, m = len(a), len(b)
    # Band around the line between endpoints, with room for local edits.
    radius = max(25, abs(n - m) + 20)
    acc = np.full((n + 1, m + 1), np.inf)
    step = np.zeros((n + 1, m + 1), np.uint8)
    acc[0, 0] = 0
    for i in range(1, n + 1):
        center = round(i * m / n)
        for j in range(max(1, center - radius), min(m, center + radius) + 1):
            cost = float(np.mean((a[i-1] - b[j-1]) ** 2))
            candidates = (acc[i-1, j-1], acc[i-1, j], acc[i, j-1])
            k = int(np.argmin(candidates))
            acc[i, j] = cost + candidates[k]
            step[i, j] = k + 1
    if not np.isfinite(acc[n, m]):
        raise ValueError("DTW band excluded all endpoint paths")
    pairs = []
    i, j = n, m
    while i and j:
        pairs.append((i-1, j-1))
        k = step[i, j]
        if k == 1: i, j = i-1, j-1
        elif k == 2: i -= 1
        elif k == 3: j -= 1
        else: raise ValueError("Invalid DTW path")
    pairs.reverse()
    return np.asarray(pairs, dtype=int), float(acc[n, m] / len(pairs))


def compare_pair(root, label):
    cid, base = label["clip_id"], label["baseline_clip_id"]
    with np.load(root / "features" / f"{base}.npz") as x, np.load(root / "features" / f"{cid}.npz") as y:
        path, cost = path_for(x["mfcc"], y["mfcc"])
        tbase, tvariant = x["time_seconds"], y["time_seconds"]
        mapped = np.full(len(tvariant), np.nan)
        for j in np.unique(path[:, 1]):
            mapped[j] = np.median(tbase[path[path[:, 1] == j, 0]])
        if np.isnan(mapped).any():
            raise ValueError("Unmapped DTW frames")
        offset = tvariant - mapped
        sample_times = tvariant[::5]
        sample_offset = offset[::5]
        change = np.abs(np.diff(sample_offset))
        inside = (sample_times[:-1] >= label["start_seconds"]) & (sample_times[:-1] < label["end_seconds"])
        inside_change = float(np.mean(change[inside])) if np.any(inside) else None
        outside_change = float(np.mean(change[~inside])) if np.any(~inside) else None
        # Report a path-derived timing curve; avoid inventing a phoneme label.
        return {"clip_id": cid, "split": label["split"], "flaw_type": label["flaw_type"],
                "severity": label["severity"], "known_edit_start_seconds": label["start_seconds"],
                "known_edit_end_seconds": label["end_seconds"], "normalized_path_cost": round(cost, 4),
                "maximum_absolute_offset_seconds": round(float(np.max(np.abs(offset))), 3),
                "mean_abs_offset_change_per_100ms_inside_edit_seconds": round(inside_change, 4) if inside_change is not None else None,
                "mean_abs_offset_change_per_100ms_outside_edit_seconds": round(outside_change, 4) if outside_change is not None else None,
                "timing_curve": [{"variant_seconds": round(float(t), 3),
                                   "reference_seconds": round(float(r), 3),
                                   "offset_seconds": round(float(o), 3)}
                                  for t, r, o in zip(tvariant[::5], mapped[::5], offset[::5])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--clip-id")
    args = ap.parse_args()
    labels = rows(args.root / "metadata/flaw_labels.jsonl")
    selected = [x for x in labels if x["clip_id"] == args.clip_id] if args.clip_id else [x for x in labels if x["split"] == "validation" and x["severity"] == "moderate" and x["flaw_type"] in ("pause", "rush", "drag")]
    if not selected: raise SystemExit("No matching clips")
    report = {"status": "exploratory_only", "feature": "joint-normalized MFCC 1-12", "hop_seconds": .02,
              "note": "Path offsets are not verified flaw locations or phoneme boundaries.",
              "pairs": [compare_pair(args.root, x) for x in selected]}
    dest = args.root / "reports/dtw_pilot.json"
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected)} comparisons to {dest}")


if __name__ == "__main__": main()
