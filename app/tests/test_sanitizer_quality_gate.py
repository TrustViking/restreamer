from __future__ import annotations

from app.publish.sanitizers.quality_gate import PublishQualityGate


class TestQualityGateDuplicateParagraphs:
    def test_no_duplicates(self) -> None:
        assert PublishQualityGate.has_duplicate_paragraphs(
            "First paragraph with enough words.\n\nSecond paragraph with different words."
        ) is False

    def test_exact_duplicate(self) -> None:
        assert PublishQualityGate.has_duplicate_paragraphs(
            "Same long text here with enough tokens for check.\n\nSame long text here with enough tokens for check."
        ) is True

    def test_short_paragraphs_not_flagged(self) -> None:
        assert PublishQualityGate.has_duplicate_paragraphs("Hi.\n\nHi.") is False


class TestQualityGateOpenerCta:
    def test_normal_opener(self) -> None:
        assert PublishQualityGate.has_opener_cta(
            "Climate change accelerates in Arctic regions.\n\n🔹 New data shows..."
        ) is False

    def test_cta_opener(self) -> None:
        assert PublishQualityGate.has_opener_cta(
            "Subscribe to our channel for updates!\n\n🔹 Today we discuss..."
        ) is True

    def test_comment_cta_opener_en(self) -> None:
        assert PublishQualityGate.has_opener_cta(
            "Leave a comment with what stood out most.\n\n🔹 Today we discuss..."
        ) is True

    def test_comment_cta_opener_uk(self) -> None:
        assert PublishQualityGate.has_opener_cta(
            "Напишіть у коментар ваші думки.\n\n🔹 Сьогодні розглянемо..."
        ) is True

    def test_comment_cta_opener_ru(self) -> None:
        assert PublishQualityGate.has_opener_cta(
            "Оставляйте комментарии по фактам.\n\n🔹 Сегодня разберем..."
        ) is True

    def test_empty(self) -> None:
        assert PublishQualityGate.has_opener_cta("") is False
