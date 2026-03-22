from __future__ import annotations

import unittest

from app.llm.merges.merge_parser import parse_merge_response_or_raise


class OpenerCtaRawCheckTests(unittest.TestCase):
    def test_opener_cta_is_rejected_before_tail_recovery(self) -> None:
        with self.assertRaises(RuntimeError) as raised:
            parse_merge_response_or_raise(
                provider_name="openai",
                model_name="gpt-5.1",
                raw_text=(
                    '{"title":"CTA opener","description":"'
                    "Підпишіться та напишіть у коментарях, яку тему розібрати наступною. "
                    "Далі ми пояснюємо наслідки рішення, що впливає на логістику і терміни. "
                    "Окремо розкриваємо позиції сторін і практичні кроки для глядачів. "
                    "Наприкінці фіксуємо, що змінюється вже сьогодні."
                    '"}'
                ),
            )
        error_text: str = str(raised.exception)
        self.assertIn("cta_as_first_paragraph", error_text)
        self.assertNotIn("body paragraph count", error_text)


if __name__ == "__main__":
    unittest.main()
