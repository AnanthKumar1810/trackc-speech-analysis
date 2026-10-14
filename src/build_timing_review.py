"""Freeze a small, speaker-balanced manual timing audit before scoring."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def readl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    out = ROOT / "alignment_mfa"
    manifest = readl(ROOT / "metadata/manifest.jsonl")
    times = readl(out / "word_alignment_mfa.jsonl")
    flags = {x["clip_id"]: x for x in readl(out / "quality_flags.jsonl")}
    by_speaker = defaultdict(list)
    for m in manifest:
        by_speaker[m["speaker_id"]].append(m)
    chosen = []
    assigned_flaw = {"barack_obama": "drag", "john_f_kennedy": "rush", "ronald_reagan": "pause"}
    for speaker, items in sorted(by_speaker.items()):
        baseline = sorted((m for m in items if m["flaw_type"] == "none"), key=lambda m: m["clip_id"])[0]
        edited = max((m for m in items if m["flaw_type"] == assigned_flaw[speaker]),
                     key=lambda m: (flags[m["clip_id"]].get("max_old_aligner_disagreement_ms") or 0, m["clip_id"]))
        chosen.extend((baseline, edited))
    selected = {m["clip_id"] for m in chosen}
    by_clip = defaultdict(list)
    for w in times:
        if w["clip_id"] in selected:
            by_clip[w["clip_id"]].append(w)
    clips = []
    template = []
    for m in chosen:
        cid = m["clip_id"]
        ww = sorted(by_clip[cid], key=lambda w: w["word_index"])
        if len(ww) != m["word_count"]:
            raise ValueError(f"MFA word count mismatch for {cid}")
        clips.append({"clip_id": cid, "speaker_id": m["speaker_id"], "split": m["split"],
                      "flaw_type": m["flaw_type"], "severity": m["severity"],
                      "audio_path": m["audio_path"], "duration_seconds": m["duration_seconds"],
                      "transcript": (ROOT / m["transcript_path"]).read_text(encoding="utf-8").strip(),
                      "max_old_aligner_disagreement_ms": flags[cid].get("max_old_aligner_disagreement_ms"),
                      "words": [{"word_index": w["word_index"], "word": w["word"],
                                 "mfa_start_seconds": w["start_seconds"], "mfa_end_seconds": w["end_seconds"]}
                                for w in ww]})
        template.extend({"clip_id": cid, "word_index": w["word_index"], "word": w["word"],
                         "start_seconds": None, "end_seconds": None, "human_verified": False,
                         "transcript_confirmed": False, "reviewer_id": None,
                         "verification_method": None} for w in ww)
    selection = {"selection_id": "timing_pilot_6_v1", "frozen_before_review": True,
                 "clip_ids": [m["clip_id"] for m in chosen], "clip_count": len(chosen),
                 "word_count": len(template), "sampling": "one baseline per archival speaker, plus each speaker's largest old-aligner disagreement within one assigned flaw (Obama drag, Kennedy rush, Reagan pause); selected automatically before manual checking",
                 "limits": "Purposeful stress sample; not representative of all 273 clips."}
    (out / "pilot_gold_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    (out / "pilot_gold_template.jsonl").write_text("".join(json.dumps(x) + "\n" for x in template), encoding="utf-8")
    (ROOT / "timing_review_data.js").write_text("window.TIMING_REVIEW_DATA = " + json.dumps({"selection": selection, "clips": clips}) + ";\n", encoding="utf-8")
    print(f"Prepared {len(clips)} clips and {len(template)} blank human-review rows")


if __name__ == "__main__": main()
