"""Small format/metric checks for the optional MFA import stage."""
import json
import tempfile
import unittest
from pathlib import Path

from mfa_workflow import parse_textgrid, score_gold, validate_intervals


class WorkflowTests(unittest.TestCase):
    def test_textgrid_and_bounds(self):
        text = '''File type = "ooTextFile"
Object class = "TextGrid"
item [1]:
    class = "IntervalTier"
    name = "words"
    intervals [1]:
        xmin = 0.1
        xmax = 0.3
        text = "hello"
item [2]:
    class = "IntervalTier"
    name = "phones"
    intervals [1]:
        xmin = 0.1
        xmax = 0.3
        text = "HH"
'''
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "alignment_mfa") as d:
            path = Path(d) / "sample.TextGrid"
            path.write_text(text, encoding="utf-8")
            tiers = parse_textgrid(path)
        self.assertEqual(tiers["words"][0]["label"], "hello")
        self.assertEqual(tiers["phones"][0]["label"], "HH")
        self.assertTrue(validate_intervals(tiers["words"], 1))

    def test_gold_scoring(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "alignment_mfa") as d:
            out = Path(d)
            prediction = {"clip_id": "a", "word_index": 0, "word": "hello", "start_seconds": .1, "end_seconds": .3}
            gold = {**prediction, "start_seconds": .12, "end_seconds": .29, "human_verified": True, "transcript_confirmed": True, "reviewer_id": "test_reviewer", "verification_method": "direct_listening"}
            (out / "word_alignment_mfa.jsonl").write_text(json.dumps(prediction) + "\n", encoding="utf-8")
            gold_path = out / "gold.jsonl"
            gold_path.write_text(json.dumps(gold) + "\n", encoding="utf-8")
            result = score_gold(out, gold_path)
        self.assertEqual(result["matched_word_count"], 1)
        self.assertEqual(result["within_tolerance_percent"], 100)

    def test_fixed_selection_refuses_partial_review(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "alignment_mfa") as d:
            out = Path(d)
            predictions = [{"clip_id": "a", "word_index": i, "word": word, "start_seconds": i * .2, "end_seconds": i * .2 + .1} for i, word in enumerate(("hello", "world"))]
            (out / "word_alignment_mfa.jsonl").write_text("".join(json.dumps(x) + "\n" for x in predictions), encoding="utf-8")
            checked = {**predictions[0], "human_verified": True, "transcript_confirmed": True, "reviewer_id": "test_reviewer", "verification_method": "direct_listening"}
            gold = out / "gold.jsonl"
            gold.write_text(json.dumps(checked) + "\n", encoding="utf-8")
            selection = out / "selection.json"
            selection.write_text(json.dumps({"selection_id": "test", "clip_ids": ["a"], "word_count": 2}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "every word"):
                score_gold(out, gold, selection_path=selection)
            second = {**predictions[1], "human_verified": True, "transcript_confirmed": True, "reviewer_id": "test_reviewer", "verification_method": "direct_listening"}
            gold.write_text(json.dumps(checked) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
            result = score_gold(out, gold, selection_path=selection)
            self.assertEqual(result["sample_scope"], "complete_frozen_selection")
            self.assertEqual(result["matched_word_count"], 2)


if __name__ == "__main__":
    unittest.main()
