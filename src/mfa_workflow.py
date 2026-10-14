"""Prepare, import and audit MFA alignments without overwriting source labels.

Requires only Python's standard library for preparation/import. MFA itself is run
in a separately pinned environment; see MFA_WORKFLOW.md.
"""
import argparse
import csv
import json
import math
import re
import shutil
import statistics
import wave
from collections import defaultdict
from pathlib import Path

SR = 16000


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in values), encoding="utf-8")


def tokens(text):
    if re.search(r"\d", text):
        raise ValueError("Write numbers as the words actually spoken before alignment")
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower().replace("\u2019", "'"))


def wav_samples(path):
    with wave.open(str(path), "rb") as f:
        if (f.getframerate(), f.getnchannels(), f.getsampwidth()) != (SR, 1, 2):
            raise ValueError(f"Unexpected WAV format: {path}")
        return f.getnframes()


def prepare(root, out):
    manifest = rows(root / "metadata/manifest.jsonl")
    previous = rows(root / "metadata/word_alignment.jsonl")
    by_clip = defaultdict(list)
    for w in previous:
        by_clip[w["clip_id"]].append(w)
    excluded = []
    included = []
    for m in manifest:
        cid = m["clip_id"]
        spoken = root / "transcripts_spoken" / f"{cid}.txt"
        script = (root / m["transcript_path"]).read_text(encoding="utf-8").strip()
        text = spoken.read_text(encoding="utf-8").strip() if spoken.exists() else script
        # A script is only a provisional spoken transcript, not listening verification.
        provenance = "explicit_spoken_transcript_unverified" if spoken.exists() else "canonical_script_provisional"
        if not tokens(text):
            excluded.append({"clip_id": cid, "reason": "empty_transcript"})
            continue
        if not spoken.exists() and tokens(script) != [w["word"].lower() for w in by_clip[cid]]:
            excluded.append({"clip_id": cid, "reason": "script_previous_word_mismatch"})
            continue
        audio = root / m["audio_path"]
        n = wav_samples(audio)
        if n != m["num_samples"]:
            excluded.append({"clip_id": cid, "reason": "wav_manifest_duration_mismatch"})
            continue
        folder = out / "corpus" / m["speaker_id"]
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(audio, folder / f"{cid}.wav")
        (folder / f"{cid}.lab").write_text(" ".join(tokens(text)) + "\n", encoding="utf-8")
        included.append({"clip_id": cid, "speaker_id": m["speaker_id"], "family_id": m["family_id"], "split": m["split"], "flaw_type": m["flaw_type"], "severity": m["severity"], "samples": n, "transcript_source": provenance, "spoken_transcript_human_verified": False, "script_differs": tokens(script) != tokens(text), "script": script, "spoken_transcript": text})
    write_jsonl(out / "input_manifest.jsonl", included)
    write_json(out / "prepare_report.json", {"included": len(included), "excluded": excluded, "provisional_transcript_count": sum(x["transcript_source"] == "canonical_script_provisional" for x in included), "note": "A matching script or automatic alignment does not verify spoken words."})
    return included


def parse_textgrid(path):
    # MFA long TextGrid: preserve named word/phone tiers. Fail closed on format changes.
    data = path.read_text(encoding="utf-8-sig")
    tier = None
    result = {"words": [], "phones": []}
    current = {}
    for line in data.splitlines():
        s = line.strip()
        if re.match(r"item \[\d+\]:$", s):
            tier = None
        elif tier is None and s.startswith("name = "):
            name = s.split("=", 1)[1].strip().strip('"').lower()
            tier = "words" if name in ("words", "word") else "phones" if name in ("phones", "phone") else "skip"
        elif s.startswith("intervals ["):
            current = {}
        elif s.startswith("xmin = ") and tier in result:
            current["start_seconds"] = float(s.split("=", 1)[1])
        elif s.startswith("xmax = ") and tier in result:
            current["end_seconds"] = float(s.split("=", 1)[1])
        elif s.startswith("text = ") and tier in result and "start_seconds" in current:
            current["label"] = s.split("=", 1)[1].strip().strip('"')
            if current["label"].strip():
                result[tier].append(current)
            current = {}
    if not result["words"]:
        raise ValueError("No words tier found")
    return result


def validate_intervals(items, duration):
    previous = -1.0
    for w in items:
        a, b = w["start_seconds"], w["end_seconds"]
        if not all(math.isfinite(v) for v in (a, b)) or a < 0 or b <= a or b > duration + .02 or a < previous - 1e-5:
            return False
        previous = b
    return True


def import_alignments(root, out, aligned):
    source = rows(out / "input_manifest.jsonl")
    by_id = {m["clip_id"]: m for m in source}
    old = defaultdict(list)
    for w in rows(root / "metadata/word_alignment.jsonl"):
        old[w["clip_id"]].append(w)
    imported, phones, quality = [], [], []
    for cid, m in by_id.items():
        candidates = list(aligned.rglob(f"{cid}.TextGrid"))
        flags = []
        if len(candidates) != 1:
            quality.append({"clip_id": cid, "accepted": False, "flags": ["missing_or_duplicate_mfa_output"]})
            continue
        try:
            tiers = parse_textgrid(candidates[0])
        except (ValueError, OSError) as e:
            quality.append({"clip_id": cid, "accepted": False, "flags": ["textgrid_parse_error"], "detail": str(e)})
            continue
        duration = m["samples"] / SR
        ww = [w for w in tiers["words"] if w["label"].lower() not in ("sil", "sp", "spn", "<unk>")]
        if not validate_intervals(ww, duration):
            flags.append("invalid_word_intervals")
        if not validate_intervals(tiers["phones"], duration):
            flags.append("invalid_phone_intervals")
        expected = tokens(m["spoken_transcript"])
        actual = [w["label"].lower() for w in ww]
        if actual != expected:
            flags.append("word_sequence_mismatch")
        if flags:
            quality.append({"clip_id": cid, "accepted": False, "flags": flags, "expected": expected, "actual": actual})
            continue
        deviations = []
        for i, w in enumerate(ww):
            a = round(w["start_seconds"] * SR)
            b = round(w["end_seconds"] * SR)
            imported.append({"clip_id": cid, "word_index": i, "word": actual[i], "start_sample": a, "end_sample": b, "start_seconds": a / SR, "end_seconds": b / SR, "timing_method": "MFA automatic; model/version recorded separately", "human_verified": False})
            if i < len(old[cid]):
                deviations.extend((abs(a - old[cid][i]["start_sample"]) / SR, abs(b - old[cid][i]["end_sample"]) / SR))
        for i, p in enumerate(tiers["phones"]):
            phones.append({"clip_id": cid, "phone_index": i, **p})
        if deviations and max(deviations) > .1:
            flags.append("old_aligner_disagreement_gt_100ms")
        quality.append({"clip_id": cid, "accepted": True, "flags": flags, "max_old_aligner_disagreement_ms": round(max(deviations) * 1000, 2) if deviations else None, "median_old_aligner_disagreement_ms": round(statistics.median(deviations) * 1000, 2) if deviations else None, "needs_listening_review": bool(flags) or not m["spoken_transcript_human_verified"]})
    write_jsonl(out / "word_alignment_mfa.jsonl", imported)
    write_jsonl(out / "phone_alignment_mfa.jsonl", phones)
    write_jsonl(out / "quality_flags.jsonl", quality)
    accepted = sum(q["accepted"] for q in quality)
    write_json(out / "alignment_report.json", {"total_clips": len(source), "accepted_clips": accepted, "coverage_percent": round(100 * accepted / len(source), 2) if source else None, "flagged_clips": sum(bool(q["flags"]) for q in quality), "human_word_boundary_accuracy": None, "accuracy_note": "Coverage and disagreement with PocketSphinx are not boundary accuracy. A hand-labeled reference is required."})


def score_gold(out, gold_path, tolerance_ms=50, selection_path=None):
    predicted = rows(out / "word_alignment_mfa.jsonl")
    gold = rows(gold_path)
    if not gold:
        raise ValueError("Gold boundaries file is empty")
    for g in gold:
        if g.get("human_verified") is not True:
            raise ValueError("Every gold row must have human_verified=true after manual boundary review")
        if g.get("transcript_confirmed") is not True or g.get("verification_method") != "direct_listening":
            raise ValueError("Gold rows require transcript confirmation and direct listening")
        if not isinstance(g.get("reviewer_id"), str) or not g["reviewer_id"].strip():
            raise ValueError("Gold rows require a reviewer ID")
        if not all(isinstance(g.get(edge), (int, float)) for edge in ("start_seconds", "end_seconds")):
            raise ValueError("Every gold row needs numeric start_seconds and end_seconds")
        if g["start_seconds"] < 0 or g["end_seconds"] <= g["start_seconds"]:
            raise ValueError("Invalid gold boundary interval")
    p = {(x["clip_id"], x["word_index"]): x for x in predicted}
    keys = [(g["clip_id"], g["word_index"]) for g in gold]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate gold word rows")
    manifest_path = out / "input_manifest.jsonl"
    durations = {x["clip_id"]: x["samples"] / SR for x in rows(manifest_path)} if manifest_path.exists() else {}
    for g in gold:
        if g["clip_id"] in durations and g["end_seconds"] > durations[g["clip_id"]] + .01:
            raise ValueError("Gold boundary exceeds audio duration")
    by_gold_clip = defaultdict(list)
    for g in gold:
        by_gold_clip[g["clip_id"]].append(g)
    for clip_rows in by_gold_clip.values():
        ordered = sorted(clip_rows, key=lambda x: x["word_index"])
        if any(a["start_seconds"] > b["start_seconds"] or a["end_seconds"] > b["end_seconds"] for a, b in zip(ordered, ordered[1:])):
            raise ValueError("Gold word boundaries are not monotonic")
    selection = None
    if selection_path:
        selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
        wanted = {key for key in p if key[0] in set(selection["clip_ids"])}
        if set(keys) != wanted or len(gold) != selection["word_count"]:
            raise ValueError("Gold file must contain every word of the frozen selection, exactly once")
    errors = []
    missing = []
    for g in gold:
        key = (g["clip_id"], g["word_index"])
        if key not in p or p[key]["word"].lower() != g["word"].lower():
            missing.append(key)
            continue
        for edge in ("start_seconds", "end_seconds"):
            errors.append(abs(p[key][edge] - g[edge]) * 1000)
    if selection and missing:
        raise ValueError("Gold words differ from the selected MFA words; review the transcript before scoring")
    if not errors:
        raise ValueError("No matching predicted and gold word boundaries")
    errors.sort()
    report = {"selection_id": selection.get("selection_id") if selection else None, "sample_scope": "complete_frozen_selection" if selection else "ad_hoc_checked_subset", "gold_clip_count": len({g["clip_id"] for g in gold}), "gold_word_count": len(gold), "matched_word_count": len(errors) // 2, "missing_or_mismatched": missing, "boundary_count": len(errors), "median_absolute_error_ms": round(statistics.median(errors), 2), "p90_absolute_error_ms": round(errors[min(len(errors)-1, math.ceil(.9*len(errors))-1)], 2), "within_tolerance_percent": round(100*sum(x <= tolerance_ms for x in errors)/len(errors), 2), "tolerance_ms": tolerance_ms, "gold_basis": "user-supplied direct-listening boundaries; reviewer provenance recorded, but independently audit before publishing"}
    write_json(out / "gold_accuracy_report.json", report)
    return report


def make_template(root, out):
    manifest = rows(root / "metadata/manifest.jsonl")
    old = rows(root / "metadata/word_alignment.jsonl")
    selected = []
    speakers = sorted({m["speaker_id"] for m in manifest})
    for speaker in speakers:
        group = [m for m in manifest if m["speaker_id"] == speaker]
        for kind, level in (("none", "none"), ("pause", "minor"), ("quiet", "moderate"), ("rush", "severe"), ("drag", "severe")):
            match = next((m for m in group if m["flaw_type"] == kind and m["severity"] == level), None)
            if match:
                selected.append(match["clip_id"])
    remaining = {speaker: [m["clip_id"] for m in manifest if m["speaker_id"] == speaker and m["severity"] == "severe" and m["clip_id"] not in selected] for speaker in speakers}
    while len(selected) < 20 and any(remaining.values()):
        for speaker in speakers:
            if len(selected) >= 20:
                break
            if remaining[speaker]:
                selected.append(remaining[speaker].pop(0))
    chosen = set(selected)
    template = [{"clip_id": w["clip_id"], "word_index": w["word_index"], "word": w["word"], "start_seconds": None, "end_seconds": None, "human_verified": False, "transcript_confirmed": False, "reviewer_id": None, "verification_method": None} for w in old if w["clip_id"] in chosen]
    write_jsonl(out / "manual_gold_template.jsonl", template)
    write_json(out / "manual_gold_selection.json", {"clip_ids": selected, "clip_count": len(selected), "word_count": len(template), "instruction": "Listen and mark every word boundary manually, then set human_verified=true. Do not copy automatic start/end times as gold."})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("prepare", "import", "template", "score"))
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--aligned", type=Path)
    ap.add_argument("--gold", type=Path)
    ap.add_argument("--selection", type=Path)
    args = ap.parse_args()
    out = args.out or args.root / "alignment_mfa"
    if args.stage == "prepare":
        prepare(args.root, out)
    elif args.stage == "import":
        if not args.aligned:
            ap.error("--aligned is required for import")
        import_alignments(args.root, out, args.aligned)
    elif args.stage == "template":
        make_template(args.root, out)
    else:
        if not args.gold:
            ap.error("--gold is required for score")
        score_gold(out, args.gold, selection_path=args.selection)


if __name__ == "__main__":
    main()
