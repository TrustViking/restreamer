from __future__ import annotations

from types import SimpleNamespace

from app.publish.shared_helpers import (
    fallback_source_description_text,
    is_merge_payload_blocked,
    light_polish_single_source_description,
    no_description_text,
    normalize_urls_in_text,
    numbered_lines,
    numbered_original_titles,
)


def test_normalize_urls_cleans_youtube() -> None:
    text = "Watch https://www.youtube.com/watch?v=AygiyNcn9RM&t=1s here"
    result = normalize_urls_in_text(text)
    assert "https://youtu.be/AygiyNcn9RM" in result
    assert "&t=1s" not in result


def test_normalize_urls_preserves_non_youtube() -> None:
    text = "Visit https://example.com/page?q=test"
    result = normalize_urls_in_text(text)
    assert "https://example.com/page?q=test" in result


def test_numbered_lines_single() -> None:
    assert numbered_lines(["hello"]) == "hello"


def test_numbered_lines_multiple() -> None:
    result = numbered_lines(["a", "b"])
    assert "1) a" in result
    assert "2) b" in result


def test_numbered_lines_empty() -> None:
    assert numbered_lines([]) == "1) ..."


def test_no_description_text_default() -> None:
    assert no_description_text(None) == "no description"


def test_numbered_original_titles() -> None:
    videos = [
        SimpleNamespace(metadata=SimpleNamespace(title="Title A")),
        SimpleNamespace(metadata=SimpleNamespace(title="Title B")),
    ]
    result = numbered_original_titles(videos)
    assert "1) Title A" in result
    assert "2) Title B" in result


def test_is_merge_payload_blocked() -> None:
    payload = SimpleNamespace(
        has_publish_stage_duplicate=True,
        has_publish_stage_opener_cta=False,
    )
    assert is_merge_payload_blocked(payload) is True


def test_doc_and_telegram_use_identical_fallback_description_processing() -> None:
    from app.publish.doc_helpers import _fallback_source_description_text as doc_fallback
    from app.publish.telegram_renderer import _fallback_source_description_text as telegram_fallback

    video = SimpleNamespace(
        metadata=SimpleNamespace(
            description="Watch: https://www.youtube.com/watch?v=AygiyNcn9RM&t=1s",
        )
    )
    templates = None
    assert fallback_source_description_text(video, templates) == doc_fallback(video, templates)
    assert fallback_source_description_text(video, templates) == telegram_fallback(video, templates)
