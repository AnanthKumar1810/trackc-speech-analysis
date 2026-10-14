"""Evaluate the frozen digital-edit classifier on independent, reviewed WAVs.

Never treats digital variants as independent people or inferred labels as gold.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from core import SR, read, loadl
from infer_model import infer_pair

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("none", "pause", "quiet", "rush", "drag")
FIELDS = ("recording_id", "participant_id", "reference_clip_id", "audio_path",
          "human_primary_flaw", "human_label_verified", "spoken_transcript_verified",
          "consent_for_local_evaluation", "single_primary_flaw")


def yes(value):
    return value.strip().lower() in ("true", "yes", "1")


def load_cases(root, manifest):
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(x not in reader.fieldnames for x in FIELDS):
            raise ValueError("Real-evaluation manifest is missing required columns")
        cases = list(reader)
    if not cases:
        raise ValueError("No independently recorded, human-labeled cases have been supplied")
    known = {m["clip_id"]: m for m in loadl(root / "metadata/manifest.jsonl")}
    source_hashes = {m["audio_sha256"] for m in known.values()}
    seen = set()
    for row in cases:
        rid = row["recording_id"].strip()
        if not rid or rid in seen: raise ValueError("Recording IDs must be unique and nonempty")
        seen.add(rid)
        if not row["participant_id"].strip(): raise ValueError(f"Missing participant ID: {rid}")
        reference = known.get(row["reference_clip_id"])
        if not reference or reference["flaw_type"] != "none": raise ValueError(f"Reference must be a released baseline: {rid}")
        if row["participant_id"] == reference["speaker_id"]: raise ValueError(f"Archival speaker is not an independent participant: {rid}")
        if row["human_primary_flaw"] not in KINDS: raise ValueError(f"Unknown human primary flaw: {rid}")
        if not all(yes(row[x]) for x in ("human_label_verified", "spoken_transcript_verified", "consent_for_local_evaluation", "single_primary_flaw")):
            raise ValueError(f"Unverified, non-consented, or multi-flaw case: {rid}")
        rel = Path(row["audio_path"])
        if rel.is_absolute() or ".." in rel.parts: raise ValueError(f"Audio path must stay inside project: {rid}")
        wav = (root / rel).resolve()
        if not wav.is_relative_to(root.resolve()) or not wav.is_file(): raise ValueError(f"Audio file missing or outside project: {rid}")
        if hashlib.sha256(wav.read_bytes()).hexdigest() in source_hashes:
            raise ValueError(f"Released dataset audio cannot be used as independent evaluation: {rid}")
        row["_wav"] = wav
        row["_reference"] = reference
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--manifest", type=Path, default=ROOT / "real_evaluation/manifest.csv")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    cases = load_cases(args.root, args.manifest)
    if args.validate_only:
        print(f"Validated {len(cases)} independent labeled recordings from {len({x['participant_id'] for x in cases})} participants")
        return
    results = []
    for row in cases:
        pcm, sr = sf.read(row["_wav"])
        if pcm.ndim == 2: pcm = pcm.mean(axis=1)
        if not 1 <= len(pcm) / sr <= 60: raise ValueError(f"Recording duration outside 1–60 seconds: {row['recording_id']}")
        pcm = np.clip(np.rint(resample_poly(pcm, SR, sr) * 32768), -32768, 32767).astype("int16")
        ref = read(args.root / row["_reference"]["audio_path"])
        prediction = infer_pair(ref, pcm)["type"]["predicted"]
        results.append({"recording_id": row["recording_id"], "participant_id": row["participant_id"],
                        "reference_clip_id": row["reference_clip_id"], "human_primary_flaw": row["human_primary_flaw"],
                        "predicted_digital_edit_signature": prediction, "match": prediction == row["human_primary_flaw"]})
    labels = sorted({x["human_primary_flaw"] for x in results})
    f1 = {}
    for kind in labels:
        tp = sum(x["human_primary_flaw"] == kind and x["predicted_digital_edit_signature"] == kind for x in results)
        fp = sum(x["human_primary_flaw"] != kind and x["predicted_digital_edit_signature"] == kind for x in results)
        fn = sum(x["human_primary_flaw"] == kind and x["predicted_digital_edit_signature"] != kind for x in results)
        f1[kind] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    report = {"status": "exploratory_external_domain_evaluation", "clip_count": len(results),
              "participant_count": len({x["participant_id"] for x in results}),
              "human_label_counts": dict(Counter(x["human_primary_flaw"] for x in results)),
              "edit_type_match_percent": round(100 * sum(x["match"] for x in results) / len(results), 2),
              "per_observed_class_f1": f1, "results": results,
              "limits": "Frozen model was trained on digital edits. Natural flaws may overlap, labels are single-primary-flaw judgments, and this score is not speaking-quality accuracy. Report sample and speaker counts alongside any percentage."}
    dest = args.root / "real_evaluation/report.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote exploratory report for {len(results)} clips to {dest}")


if __name__ == "__main__":
    try: main()
    except ValueError as exc: raise SystemExit(str(exc)) from None
