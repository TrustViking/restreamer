from __future__ import annotations

import unittest

from app.core.confusables import detect_target_script, normalize_confusables


class TestConfusablesNormalization(unittest.TestCase):

    def test_lowercase_latin_to_cyrillic_includes_k_h_m_b_t(self) -> None:
        # Real-world bug: source contained 'Жайвороноk' with latin lowercase k
        # that prior code (uppercase-only table) failed to fold.
        sample: str = "Жайвороноk"
        normalized: str = normalize_confusables(sample, target_script="cyrillic")
        self.assertEqual(normalized, "Жайворонок")
        self.assertNotIn("k", normalized)
        self.assertIn("к", normalized)

    def test_lowercase_h_to_cyrillic(self) -> None:
        self.assertEqual(normalize_confusables("Аh", target_script="cyrillic"), "Ан")

    def test_lowercase_m_to_cyrillic(self) -> None:
        self.assertEqual(normalize_confusables("am", target_script="cyrillic"), "ам")

    def test_lowercase_b_to_cyrillic(self) -> None:
        self.assertEqual(normalize_confusables("bаба", target_script="cyrillic"), "ваба")

    def test_lowercase_t_to_cyrillic(self) -> None:
        self.assertEqual(normalize_confusables("tак", target_script="cyrillic"), "так")

    def test_uppercase_latin_to_cyrillic_regression(self) -> None:
        # Prior behavior: uppercase mappings worked. Make sure we kept them.
        self.assertEqual(normalize_confusables("Olena", target_script="cyrillic"), "Оlеnа")
        # Note: 'l' and 'n' have no homoglyph mapping by design.

    def test_cyrillic_to_latin_symmetric(self) -> None:
        # Reverse direction works for the same character set.
        self.assertEqual(normalize_confusables("кот", target_script="latin"), "kot")
        self.assertEqual(normalize_confusables("Карта", target_script="latin"), "Kapta")

    def test_unmapped_chars_pass_through(self) -> None:
        # Characters with no homoglyph pair must be untouched.
        self.assertEqual(normalize_confusables("dfgjlnqrsuvwz", target_script="cyrillic"), "dfgjlnqrsuvwz")

    def test_auto_detects_dominant_cyrillic(self) -> None:
        # Mostly cyrillic text with latin homoglyphs sprinkled in.
        sample: str = "Жайвороноk («Вікіпедія»)"
        normalized: str = normalize_confusables(sample, target_script="auto")
        self.assertIn("к", normalized)
        self.assertNotIn("k", normalized)

    def test_auto_detects_dominant_latin(self) -> None:
        # Mostly latin text with one cyrillic character - should fold to latin.
        sample: str = "Hellо world"  # 'о' here is cyrillic
        normalized: str = normalize_confusables(sample, target_script="auto")
        self.assertEqual(normalized, "Hello world")

    def test_invalid_target_script_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_confusables("test", target_script="klingon")

    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(normalize_confusables("", target_script="cyrillic"), "")
        self.assertEqual(normalize_confusables("", target_script="auto"), "")


class TestDetectTargetScript(unittest.TestCase):

    def test_dominant_cyrillic(self) -> None:
        self.assertEqual(detect_target_script("Привіт world"), "cyrillic")

    def test_dominant_latin(self) -> None:
        self.assertEqual(detect_target_script("Hello світ"), "latin")

    def test_tie_resolves_to_cyrillic(self) -> None:
        # 4 cyrillic vs 4 latin -> cyrillic (matches prior behavior)
        self.assertEqual(detect_target_script("ABCD абвг"), "cyrillic")

    def test_no_letters_returns_cyrillic(self) -> None:
        # Empty letter count -> cyrillic_count (0) >= latin_count (0) -> cyrillic
        self.assertEqual(detect_target_script("123 !@#"), "cyrillic")


if __name__ == "__main__":
    unittest.main()
