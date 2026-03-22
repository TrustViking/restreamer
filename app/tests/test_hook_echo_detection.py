from __future__ import annotations

import unittest

from app.llm.merges.merge_validation import _has_hook_echo_in_body
from app.llm.merges.merge_validation_helpers import (
    attempt_hook_echo_repair,
    has_hook_echo_in_body,
)


class HookEchoDetectionTests(unittest.TestCase):
    def test_exact_duplicate_hook_returns_true(self) -> None:
        description_text: str = (
            "Tonight we map how the sanctions vote changes transport risk windows for the next 48 hours.\n\n"
            "Tonight we map how the sanctions vote changes transport risk windows for the next 48 hours.\n\n"
            "🔹 New facts appear only here."
        )
        self.assertTrue(_has_hook_echo_in_body(description_text))

    def test_rephrased_hook_in_body_opener_returns_true(self) -> None:
        description_text: str = (
            "Tonight we map how the sanctions vote changes transport risk windows for the next 48 hours and why the aid corridor schedule shifts.\n\n"
            "The sanctions vote changes transport risk windows in the next 48 hours and shifts the aid corridor schedule, so we map those operational consequences in detail.\n\n"
            "🔹 Additional source facts start after this line."
        )
        self.assertTrue(_has_hook_echo_in_body(description_text))

    def test_different_hook_and_body_returns_false(self) -> None:
        description_text: str = (
            "Tonight we map how the sanctions vote changes transport risk windows for the next 48 hours.\n\n"
            "🔹 Geneva timeline updates with new WHO cargo batches and hospital routing details.\n\n"
            "🔹 Lviv repair crews report transformer queue reductions after midnight."
        )
        self.assertFalse(_has_hook_echo_in_body(description_text))

    def test_short_paragraphs_are_skipped(self) -> None:
        description_text: str = (
            "Short opener line.\n\n"
            "Short opener line.\n\n"
            "Another short line."
        )
        self.assertFalse(_has_hook_echo_in_body(description_text))

    def test_common_prefix_over_50_percent_returns_true(self) -> None:
        hook: str = "Tonight we map how the sanctions vote changes transport risk windows for the next 48 hours."
        # Body opener shares the first ~60% of hook characters, then diverges
        shared: str = hook[: int(len(hook) * 0.55)]
        body_opener: str = shared + " — additional context that makes this paragraph long enough to pass the 40-char gate."
        description_text: str = f"{hook}\n\n{body_opener}\n\n🔹 Some new bullet fact here."
        self.assertTrue(_has_hook_echo_in_body(description_text))

    def test_last_sentence_echo_in_body_opener_returns_true(self) -> None:
        """Case from 230326 RU: hook ends with thesis, body opens with same thesis."""
        description_text: str = (
            "«Він робив нам уколи і прикасався до нас» — 32 жертви Якуба Яхла "
            "вперше дають свідчення на камеру. "
            "⚖ Центральний конфлікт — безпека дітей і питання відповідальності.\n\n"
            "⚖ Центральний конфлікт — безпека дітей і питання відповідальності.\n"
            "🔹 Юридичне оновлення у справі.\n"
            "🔹 Свідчення дітей та міжнародний тиск.\n\n"
            "Дивіться ефір і діліться думками."
        )
        self.assertTrue(_has_hook_echo_in_body(description_text))

    def test_hook_thesis_not_repeated_returns_false(self) -> None:
        """Hook ends with thesis, body opens with different content — no echo."""
        description_text: str = (
            "«Він робив нам уколи і прикасався до нас» — 32 жертви Якуба Яхла "
            "вперше дають свідчення на камеру. "
            "⚖ Центральний конфлікт — безпека дітей і питання відповідальності.\n\n"
            "🔹 Юридичне оновлення: суд визнав докази неналежними.\n"
            "🔹 Свідчення дітей та міжнародний тиск.\n\n"
            "Дивіться ефір і діліться думками."
        )
        self.assertFalse(_has_hook_echo_in_body(description_text))


class HookEchoRepairTests(unittest.TestCase):
    _LONG_HOOK: str = (
        "Tonight we map how the sanctions vote changes transport risk windows for the next 48 hours "
        "and why the aid corridor schedule shifts across all active fronts."
    )

    def test_repair_removes_echo_paragraph_and_preserves_bullets(self) -> None:
        # Exact duplicate echo — prefix_ratio = 1.0 triggers detection
        hook: str = self._LONG_HOOK
        bullets: str = "🔹 Geneva timeline updated.\n🔹 Lviv repair crews active."
        description_text: str = f"{hook}\n\n{hook}\n\n{bullets}"
        repaired: str | None = attempt_hook_echo_repair(description_text)
        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertTrue(repaired.startswith(hook))
        self.assertIn("🔹 Geneva timeline updated.", repaired)
        # The echo (second paragraph = hook duplicate) must be gone
        paragraphs_after: list[str] = repaired.split("\n\n")
        self.assertEqual(len(paragraphs_after), 2)

    def test_repair_trims_echo_prefix_within_paragraph(self) -> None:
        # Paragraph 2 starts with the hook text then contains bullet lines
        hook: str = self._LONG_HOOK
        echo_with_bullets: str = f"{hook}\n🔹 New cargo batch update.\n🔹 Hospital routing adjusted."
        tail: str = "Subscribe for more updates."
        description_text: str = f"{hook}\n\n{echo_with_bullets}\n\n{tail}"
        repaired: str | None = attempt_hook_echo_repair(description_text)
        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertIn("🔹 New cargo batch update.", repaired)
        # The echoed prose prefix should be stripped
        self.assertNotIn(f"{hook}\n🔹", repaired)

    def test_repair_returns_none_when_no_bullets_after_echo(self) -> None:
        hook: str = self._LONG_HOOK
        prose: str = "This is another prose paragraph with no bullet markers at all in it anywhere."
        description_text: str = f"{hook}\n\n{hook}\n\n{prose}"
        repaired: str | None = attempt_hook_echo_repair(description_text)
        self.assertIsNone(repaired)

    def test_repair_returns_none_for_short_description(self) -> None:
        # Only 2 paragraphs → repair cannot proceed
        hook: str = self._LONG_HOOK
        description_text: str = f"{hook}\n\n{hook}"
        repaired: str | None = attempt_hook_echo_repair(description_text)
        self.assertIsNone(repaired)

    def test_repair_handles_compact_two_paragraph_echo(self) -> None:
        """2-paragraph compact description: hook + echo-prefix+bullets. Repair must succeed."""
        hook: str = self._LONG_HOOK
        echo_with_bullets: str = (
            f"{hook}\n"
            "🔹 Geneva timeline updated with new WHO cargo batches and routing details.\n"
            "🔹 Lviv repair crews report transformer queue reductions after midnight."
        )
        description_text: str = f"{hook}\n\n{echo_with_bullets}"
        repaired: str | None = attempt_hook_echo_repair(description_text)
        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertFalse(has_hook_echo_in_body(repaired))
        first_bullet: str = "🔹 Geneva timeline updated with new WHO cargo batches and routing details."
        self.assertEqual(repaired.count(first_bullet), 1)

    def test_repair_handles_bullet_fused_into_hook(self) -> None:
        """Reproduces 220326 pattern: hook ends with fused bullet, para 2 echoes then lists bullets."""
        hook_clean: str = (
            "«Он делал нам уколы и прикасался к нам» — свидетельства жертв Якуба Яхла становятся срочными."
        )
        fused_bullet: str = "🔹 Встреча с послом Лазаро Ньяланду обсуждает обвинения"
        hook_fused: str = f"{hook_clean} {fused_bullet}"
        echo_body: str = (
            f"{hook_clean}\n"
            f"{fused_bullet}\n"
            "🔹 По имеющимся данным, расследование расширяется на другие страны.\n"
            "📌 Сообщение со встречи дипломатического корпуса."
        )
        hashtags: str = "#Танзания #ЯкубЯхл"
        description_text: str = f"{hook_fused}\n\n{echo_body}\n\n{hashtags}"
        repaired: str | None = attempt_hook_echo_repair(description_text)
        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertFalse(has_hook_echo_in_body(repaired))
        # Fused bullet was identical to first body bullet — dedup must prevent double occurrence
        self.assertEqual(repaired.count("🔹 Встреча с послом Лазаро Ньяланду"), 1)
        # Other bullets must be preserved
        self.assertIn("🔹 По имеющимся данным", repaired)
        self.assertIn("📌 Сообщение со встречи", repaired)

    def test_repair_compact_two_para_returns_none_without_bullets(self) -> None:
        """2-paragraph echo with no bullet markers anywhere — repair must return None."""
        hook: str = self._LONG_HOOK
        prose_echo: str = (
            f"{hook} — this is a prose restatement of the hook without any bullet markers "
            "and without any new facts or additional content that would allow repair."
        )
        description_text: str = f"{hook}\n\n{prose_echo}"
        repaired: str | None = attempt_hook_echo_repair(description_text)
        self.assertIsNone(repaired)

    def test_repair_preserves_bullets_in_echo_paragraph_and_later_paragraphs(self) -> None:
        hook: str = self._LONG_HOOK
        echo_with_bullets: str = (
            f"{hook}\n"
            "🔹 Bullet from echo paragraph.\n"
            "🔹 Second bullet from echo paragraph."
        )
        later_paragraph: str = "🔹 Bullet from paragraph three.\n🔹 Another later fact."
        description_text: str = f"{hook}\n\n{echo_with_bullets}\n\n{later_paragraph}"
        repaired: str | None = attempt_hook_echo_repair(description_text)
        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertIn("🔹 Bullet from echo paragraph.", repaired)
        self.assertIn("🔹 Bullet from paragraph three.", repaired)


if __name__ == "__main__":
    unittest.main()
