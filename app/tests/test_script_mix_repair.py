from __future__ import annotations

import unittest

from app.llm.merges.script_mix_repair import repair_script_mix_homoglyphs

# Build test tokens from explicit codepoints to avoid invisible homoglyph confusion.
# Latin: o = U+006F, k = U+006B
# Cyrillic: о = U+043E, к = U+043A
_ZHAIVORONOK_MIXED = "Жайворон" + "o" + "k"   # last two chars are Latin o, k
_ZHAIVORONOK_CYRILLIC = "Жайворон" + "о" + "к"  # fully Cyrillic


class TestRepairScriptMixHomoglyphs(unittest.TestCase):

    def test_zhaivoronok_uk(self) -> None:
        description = f"🔹 Тарас Іванов, Владислав {_ZHAIVORONOK_MIXED} («Вікіпедія»)"
        repaired, stats = repair_script_mix_homoglyphs(description=description, language="uk")

        self.assertEqual(stats.tokens_repaired, 1)
        self.assertEqual(stats.repaired_tokens_before, (_ZHAIVORONOK_MIXED,))
        self.assertEqual(stats.repaired_tokens_after, (_ZHAIVORONOK_CYRILLIC,))

        # Mixed token must be gone; fully Cyrillic must be present.
        self.assertNotIn(_ZHAIVORONOK_MIXED, repaired)
        self.assertIn(_ZHAIVORONOK_CYRILLIC, repaired)

        # Verify Unicode codepoints of the repaired token — guards against silent
        # visual confusion where two strings look identical but differ in script.
        repaired_token = stats.repaired_tokens_after[0]
        self.assertEqual(repaired_token[-1], "к", "last char must be Cyrillic к (U+043A)")
        self.assertEqual(repaired_token[-2], "о", "second-to-last must be Cyrillic о (U+043E)")

    def test_latin_only_word_not_repaired(self) -> None:
        description = "Hello twork world"
        repaired, stats = repair_script_mix_homoglyphs(description=description, language="uk")

        self.assertEqual(stats.tokens_repaired, 0)
        self.assertEqual(repaired, description)

    def test_pure_cyrillic_text_not_repaired(self) -> None:
        description = "Звичайний український текст"
        repaired, stats = repair_script_mix_homoglyphs(description=description, language="uk")

        self.assertEqual(stats.tokens_repaired, 0)
        self.assertEqual(repaired, description)

    def test_latin_with_non_homoglyph_char_not_repaired(self) -> None:
        # "Кириллица" is Cyrillic (9 chars) + "def" Latin (3 chars) → cyr > lat
        # but "d" and "f" are not in the homoglyph map → token must not be repaired.
        mixed = "Кириллица" + "def"   # d=U+0064, f=U+0066 not in map
        description = mixed
        repaired, stats = repair_script_mix_homoglyphs(description=description, language="uk")

        self.assertEqual(stats.tokens_repaired, 0)
        self.assertEqual(repaired, description)

    def test_english_language_passthrough(self) -> None:
        description = "Some english text with cyrillicа"
        repaired, stats = repair_script_mix_homoglyphs(description=description, language="en")

        self.assertEqual(stats.tokens_repaired, 0)
        self.assertEqual(repaired, description)

    def test_russian_i_and_c_repair(self) -> None:
        # "текст" is 5 Cyrillic chars; "ic" is 2 Latin chars (i, c) → cyr > lat.
        # For "ru": i → и (U+0438), c → с (U+0441).
        mixed_token = "текст" + "i" + "c"    # Latin i, c
        cyrillic_token = "текст" + "и" + "с"  # Cyrillic и, с
        description = mixed_token
        repaired, stats = repair_script_mix_homoglyphs(description=description, language="ru")

        self.assertEqual(stats.tokens_repaired, 1)
        self.assertEqual(stats.repaired_tokens_after, (cyrillic_token,))
        self.assertIn(cyrillic_token, repaired)

    def test_two_mixed_tokens_both_repaired(self) -> None:
        # "Жайворонok" and "Кoзел" (where o in Козел is Latin U+006F)
        mixed1 = _ZHAIVORONOK_MIXED
        mixed2 = "К" + "o" + "зел"    # К + Latin o + зел
        cyrillic2 = "К" + "о" + "зел"  # К + Cyrillic о + зел

        description = f"{mixed1} і {mixed2}"
        repaired, stats = repair_script_mix_homoglyphs(description=description, language="uk")

        self.assertEqual(stats.tokens_repaired, 2)
        self.assertIn(_ZHAIVORONOK_CYRILLIC, repaired)
        self.assertIn(cyrillic2, repaired)
        self.assertNotIn(mixed1, repaired)
        self.assertNotIn(mixed2, repaired)
