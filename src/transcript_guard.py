"""Conservative, local transcript preflight for same-script comparisons.

ASR is evidence for review, never proof of what was spoken.  The decoder is
deliberately independent of the expected text; forced alignment cannot serve
as a transcript check.
"""
import re
import threading
from difflib import SequenceMatcher

_model = None
_lock = threading.Lock()


def words(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower().replace("\u2019", "'"))


def compare(expected, recognized):
    target, heard = words(expected), words(recognized)
    if not target:
        raise ValueError("Expected transcript is empty")
    operations = []
    for tag, a, b, c, d in SequenceMatcher(None, target, heard, autojunk=False).get_opcodes():
        if tag != "equal":
            operations.append({"kind": tag, "expected": target[a:b], "recognized": heard[c:d]})
    return {
        "status": "asr_agrees_provisionally" if heard == target else "needs_transcript_review",
        "expected_words": len(target), "recognized_words": len(heard),
        "recognized_text": " ".join(heard), "differences": operations,
        "note": "ASR agreement is not human verification; disagreement may be an ASR error.",
    }


def recognize_pcm(pcm):
    """Decode 16-kHz mono int16 PCM with PocketSphinx's unconstrained LM."""
    import numpy as np
    from pocketsphinx import Decoder

    decoder = Decoder()
    decoder.start_utt()
    decoder.process_raw(np.asarray(pcm, dtype="<i2").tobytes(), full_utt=True)
    decoder.end_utt()
    hypothesis = decoder.hyp()
    return hypothesis.hypstr if hypothesis else ""


def whisper_transcribe(pcm):
    """Prefer a locally cached independent ASR model; never upload/download here."""
    global _model
    import numpy as np
    from faster_whisper import WhisperModel
    with _lock:
        if _model is None:
            _model = WhisperModel('base', device='cpu', compute_type='int8', local_files_only=True)
        samples = np.asarray(pcm, dtype=np.float32) / 32768.
        segments, _ = _model.transcribe(samples, language='en', beam_size=5, word_timestamps=True)
        segments = list(segments)
        return ' '.join(s.text.strip() for s in segments), [
            {'word': w.word.strip(), 'start_seconds': float(w.start), 'end_seconds': float(w.end)}
            for s in segments for w in (s.words or [])]


def check_pcm(pcm, expected):
    fallback = None
    try:
        text, times = whisper_transcribe(pcm)
        # Spoken small numbers may be rendered as digits by ASR.
        digits = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split()
        text = re.sub(r'\b\d+\b', lambda m: digits[int(m[0])] if int(m[0]) < len(digits) else m[0], text)
        result = compare(expected, text)
        result['recognizer'] = 'faster-whisper base, local CPU; independent of expected text'
        result['automatic_word_times'] = times
    except (ImportError, OSError, RuntimeError) as exc:
        fallback = str(exc)
        result = compare(expected, recognize_pcm(pcm))
        result['recognizer'] = 'PocketSphinx unconstrained English language model (fallback)'
    if fallback: result['fallback_reason'] = fallback
    return result
