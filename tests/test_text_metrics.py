import unittest

from asr.metrics import character_error_rate, word_error_rate
from asr.text import Vocabulary, normalize_text


class TextAndMetricTests(unittest.TestCase):
    def test_normalization_and_ctc_decode(self) -> None:
        vocabulary = Vocabulary()
        self.assertEqual(normalize_text("Hello, WORLD!"), "hello world")
        encoded = vocabulary.encode("letter")
        self.assertEqual(
            vocabulary.decode(
                [*encoded[:3], vocabulary.blank_id, *encoded[3:]]
            ),
            "letter",
        )
        self.assertEqual(
            vocabulary.decode([vocabulary.char_to_id["a"], 0, vocabulary.char_to_id["a"]]),
            "aa",
        )

    def test_error_rates(self) -> None:
        self.assertAlmostEqual(word_error_rate(["the cat sat"], ["the bat sat"]), 1 / 3)
        self.assertAlmostEqual(
            character_error_rate(["cat"], ["bat"]),
            1 / 3,
        )

    def test_metric_inputs_must_match(self) -> None:
        with self.assertRaises(ValueError):
            word_error_rate(["one"], [])
        with self.assertRaises(ValueError):
            character_error_rate(["one"], [])


if __name__ == "__main__":
    unittest.main()
