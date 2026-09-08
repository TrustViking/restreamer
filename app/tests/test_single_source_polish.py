from __future__ import annotations

import unittest

from app.publish.shared_helpers import light_polish_single_source_description


class SingleSourcePolishTests(unittest.TestCase):
    def test_strips_cta_opener(self) -> None:
        text = "Приєднуйтесь до нашого ефіру!\n\nФакт один.\nФакт два."
        result = light_polish_single_source_description(text)
        self.assertFalse(result.startswith("Приєднуйтесь"))
        self.assertIn("Факт один", result)

    def test_strips_service_tail(self) -> None:
        text = "Факт один.\nФакт два.\n\nSubscribe and share with friends!"
        result = light_polish_single_source_description(text)
        self.assertNotIn("Subscribe", result)
        self.assertIn("Факт один", result)

    def test_preserves_clean_description(self) -> None:
        text = "Факт один.\nФакт два.\n\nФакт три."
        result = light_polish_single_source_description(text)
        self.assertEqual(result, text)

    def test_returns_original_if_nothing_left(self) -> None:
        text = "Subscribe and share!"
        result = light_polish_single_source_description(text)
        self.assertEqual(result, text)

    def test_strips_promotional_opener_en(self) -> None:
        text = "You will find the answers to these questions in the video.\n\nFact one.\nFact two."
        result = light_polish_single_source_description(text)
        self.assertNotIn("You will find", result)
        self.assertIn("Fact one", result)

    def test_strips_promotional_opener_uk(self) -> None:
        text = "Ви знайдете відповіді на ці запитання у стрімі.\n\nФакт один.\nФакт два."
        result = light_polish_single_source_description(text)
        self.assertNotIn("знайдете відповіді", result)
        self.assertIn("Факт один", result)

    def test_strips_promotional_opener_ru(self) -> None:
        text = "В этом видео вы узнаете подробности.\n\nФакт один.\nФакт два."
        result = light_polish_single_source_description(text)
        self.assertNotIn("вы узнаете", result)
        self.assertIn("Факт один", result)

    def test_keeps_factual_opener_with_you(self) -> None:
        """'You' in factual context is not a promotional opener."""
        text = "You can see the damage clearly in satellite images from February 2026.\n\nDetails follow."
        result = light_polish_single_source_description(text)
        self.assertIn("You can see", result)

    def test_promo_phrase_in_middle_paragraph_is_not_stripped(self) -> None:
        """Promotional phrases in non-first paragraphs must NOT trigger strip."""
        text = "Факт один — важна подія.\n\nYou will find the answers below.\n\nФакт три."
        result = light_polish_single_source_description(text)
        self.assertIn("Факт один", result)
        self.assertIn("You will find", result)

    def test_strips_comment_cta_tail_ru(self) -> None:
        text = "Факт один.\nФакт два.\n\nНапишите в комментариях ваше мнение!"
        result = light_polish_single_source_description(text)
        self.assertNotIn("Напишите в комментариях", result)
        self.assertIn("Факт один", result)

    def test_strips_comment_cta_tail_uk(self) -> None:
        text = "Факт один.\nФакт два.\n\nНапишіть у коментарях, що ви думаєте."
        result = light_polish_single_source_description(text)
        self.assertNotIn("Напишіть у коментарях", result)
        self.assertIn("Факт один", result)

    def test_strips_comment_cta_tail_en(self) -> None:
        text = "Fact one.\nFact two.\n\nTell us in the comments what you think."
        result = light_polish_single_source_description(text)
        self.assertNotIn("Tell us in the comments", result)
        self.assertIn("Fact one", result)

    def test_strips_multiple_service_tail_paragraphs(self) -> None:
        """Multiple trailing service paragraphs must all be stripped."""
        text = "Факт один.\nФакт два.\n\nSubscribe and share!\n\nJoin our community!"
        result = light_polish_single_source_description(text)
        self.assertNotIn("Subscribe", result)
        self.assertNotIn("Join our", result)
        self.assertIn("Факт один", result)

    def test_keeps_factual_paragraph_mentioning_comment(self) -> None:
        """Long factual paragraph with word 'комментарий' must not be stripped."""
        text = (
            "Юрист дал развёрнутый комментарий о позиции защиты.\n\n"
            "Анализ доказательной базы показал серьёзные нарушения."
        )
        result = light_polish_single_source_description(text)
        self.assertIn("комментарий", result)
        self.assertIn("Анализ", result)

    def test_strips_hashtag_only_tail_block(self) -> None:
        text = "Описание видео.\n\n#тег1 #тег2 #тег3"
        result = light_polish_single_source_description(text)
        self.assertEqual("Описание видео.", result)

    def test_strips_multiline_cta_before_hashtags_tail(self) -> None:
        text = (
            "1,5 часа в неделю — и жизнь меняется.\n"
            "Учёные доказали: помогая другим, человек становится здоровее.\n\n"
            "✅ Подпишитесь на канал, чтобы не пропустить новые выпуски.\n"
            "👍 Поставьте лайк, если тема волонтёрства вам близка.\n"
            "💬 Напишите в комментариях: Как помощь другим изменила вашу жизнь?\n\n"
            "#психология #волонтёрство #самопомощь"
        )
        result = light_polish_single_source_description(text)
        self.assertEqual(
            "1,5 часа в неделю — и жизнь меняется.\n"
            "Учёные доказали: помогая другим, человек становится здоровее.",
            result,
        )

    def test_strips_multiline_cta_at_end_without_hashtags(self) -> None:
        text = (
            "1,5 часа в неделю — и жизнь меняется.\n"
            "Учёные доказали: помогая другим, человек становится здоровее.\n\n"
            "✅ Подпишитесь на канал, чтобы не пропустить новые выпуски.\n"
            "👍 Поставьте лайк, если тема волонтёрства вам близка.\n"
            "💬 Напишите в комментариях: Как помощь другим изменила вашу жизнь?"
        )
        result = light_polish_single_source_description(text)
        self.assertEqual(
            "1,5 часа в неделю — и жизнь меняется.\n"
            "Учёные доказали: помогая другим, человек становится здоровее.",
            result,
        )

    def test_keeps_factual_emoji_lines(self) -> None:
        text = (
            "Главная тема эфира — безопасность в кризисных ситуациях.\n\n"
            "✅ Решение принято на встрече экспертов.\n"
            "📢 Организаторы сообщили детали программы.\n"
            "🔥 Ключевые спикеры обсудят практические аспекты."
        )
        result = light_polish_single_source_description(text)
        self.assertEqual(text, result)

    def test_keeps_long_factual_paragraph_with_comments_word(self) -> None:
        text = (
            "Эксперты в комментариях подчеркнули важность темы. По их мнению, "
            "феномен заслуживает дополнительного изучения. Этот вывод подтверждается "
            "множеством исследований последних десятилетий."
        )
        result = light_polish_single_source_description(text)
        self.assertEqual(text, result)


if __name__ == "__main__":
    unittest.main()
