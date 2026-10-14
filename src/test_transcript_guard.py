import unittest
from transcript_guard import compare


class TranscriptGuardTests(unittest.TestCase):
    def test_missing_word_needs_review(self):
        result = compare("We will speak clearly", "we speak clearly")
        self.assertEqual(result["status"], "needs_transcript_review")
        self.assertEqual(result["differences"][0]["expected"], ["will"])

    def test_exact_asr_match_is_still_provisional(self):
        result = compare("We're ready.", "we're ready")
        self.assertEqual(result["status"], "asr_agrees_provisionally")
        self.assertIn("not human verification", result["note"])


if __name__ == "__main__": unittest.main()
