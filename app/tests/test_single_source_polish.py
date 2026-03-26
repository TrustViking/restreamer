from __future__ import annotations

import unittest

from app.publish.doc_helpers import _light_polish_single_source_description


class SingleSourcePolishTests(unittest.TestCase):
    def test_strips_cta_opener(self) -> None:
        text = "Приєднуйтесь до нашого ефіру!\n\nФакт один.\nФакт два."
        result = _light_polish_single_source_description(text)
        self.assertFalse(result.startswith("Приєднуйтесь"))
        self.assertIn("Факт один", result)

    def test_strips_service_tail(self) -> None:
        text = "Факт один.\nФакт два.\n\nSubscribe and share with friends!"
        result = _light_polish_single_source_description(text)
        self.assertNotIn("Subscribe", result)
        self.assertIn("Факт один", result)

    def test_preserves_clean_description(self) -> None:
        text = "Факт один.\nФакт два.\n\nФакт три."
        result = _light_polish_single_source_description(text)
        self.assertEqual(result, text)

    def test_returns_original_if_nothing_left(self) -> None:
        text = "Subscribe and share!"
        result = _light_polish_single_source_description(text)
        self.assertEqual(result, text)

    def test_strips_promotional_opener_en(self) -> None:
        text = "You will find the answers to these questions in the video.\n\nFact one.\nFact two."
        result = _light_polish_single_source_description(text)
        self.assertNotIn("You will find", result)
        self.assertIn("Fact one", result)

    def test_strips_promotional_opener_uk(self) -> None:
        text = "Ви знайдете відповіді на ці запитання у стрімі.\n\nФакт один.\nФакт два."
        result = _light_polish_single_source_description(text)
        self.assertNotIn("знайдете відповіді", result)
        self.assertIn("Факт один", result)

    def test_strips_promotional_opener_ru(self) -> None:
        text = "В этом видео вы узнаете подробности.\n\nФакт один.\nФакт два."
        result = _light_polish_single_source_description(text)
        self.assertNotIn("вы узнаете", result)
        self.assertIn("Факт один", result)

    def test_keeps_factual_opener_with_you(self) -> None:
        """'You' in factual context is not a promotional opener."""
        text = "You can see the damage clearly in satellite images from February 2026.\n\nDetails follow."
        result = _light_polish_single_source_description(text)
        self.assertIn("You can see", result)

    def test_promo_phrase_in_middle_paragraph_is_not_stripped(self) -> None:
        """Promotional phrases in non-first paragraphs must NOT trigger strip."""
        text = "Факт один — важна подія.\n\nYou will find the answers below.\n\nФакт три."
        result = _light_polish_single_source_description(text)
        self.assertIn("Факт один", result)
        self.assertIn("You will find", result)

    def test_strips_comment_cta_tail_ru(self) -> None:
        text = "Факт один.\nФакт два.\n\nНапишите в комментариях ваше мнение!"
        result = _light_polish_single_source_description(text)
        self.assertNotIn("Напишите в комментариях", result)
        self.assertIn("Факт один", result)

    def test_strips_comment_cta_tail_uk(self) -> None:
        text = "Факт один.\nФакт два.\n\nНапишіть у коментарях, що ви думаєте."
        result = _light_polish_single_source_description(text)
        self.assertNotIn("Напишіть у коментарях", result)
        self.assertIn("Факт один", result)

    def test_strips_comment_cta_tail_en(self) -> None:
        text = "Fact one.\nFact two.\n\nTell us in the comments what you think."
        result = _light_polish_single_source_description(text)
        self.assertNotIn("Tell us in the comments", result)
        self.assertIn("Fact one", result)

    def test_strips_multiple_service_tail_paragraphs(self) -> None:
        """Multiple trailing service paragraphs must all be stripped."""
        text = "Факт один.\nФакт два.\n\nSubscribe and share!\n\nJoin our community!"
        result = _light_polish_single_source_description(text)
        self.assertNotIn("Subscribe", result)
        self.assertNotIn("Join our", result)
        self.assertIn("Факт один", result)

    def test_keeps_factual_paragraph_mentioning_comment(self) -> None:
        """Long factual paragraph with word 'комментарий' must not be stripped."""
        text = (
            "Юрист дал развёрнутый комментарий о позиции защиты.\n\n"
            "Анализ доказательной базы показал серьёзные нарушения."
        )
        result = _light_polish_single_source_description(text)
        self.assertIn("комментарий", result)
        self.assertIn("Анализ", result)


if __name__ == "__main__":
    unittest.main()
