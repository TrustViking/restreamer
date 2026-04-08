from __future__ import annotations

from app.resources.resource_loader import load_lines_resource, load_text_resource


class TestResourceLoader:
    def test_lexicon_cta_prefixes_loads(self) -> None:
        lines: tuple[str, ...] = load_lines_resource("lexicon_cta_prefixes.txt")
        assert len(lines) >= 10
        assert "Subscribe" in lines

    def test_lexicon_cta_hints_loads(self) -> None:
        lines: tuple[str, ...] = load_lines_resource("lexicon_cta_hints.txt")
        assert len(lines) >= 10
        assert "watch" in lines

    def test_lexicon_stopwords_loads(self) -> None:
        lines: tuple[str, ...] = load_lines_resource("lexicon_semantic_stopwords.txt")
        assert len(lines) >= 50

    def test_prompt_expanded_loads(self) -> None:
        text: str = load_text_resource("prompt_merge_contract_expanded.txt")
        assert "{expanded_bullet_min}" in text
        assert len(text) > 100

    def test_prompt_compact_loads(self) -> None:
        text: str = load_text_resource("prompt_merge_contract_compact.txt")
        assert "{compact_bullet_min}" in text


class TestPromptResources:
    def test_prompt_merge_title_description_loads(self) -> None:
        text: str = load_text_resource("prompt_merge_title_description.txt")
        assert "You are writing a YouTube stream title and description" in text
        assert "{language_name}" in text
        assert len(text) > 500

    def test_prompt_structural_rules_loads(self) -> None:
        text: str = load_text_resource("prompt_merge_structural_rules.txt")
        assert "STRUCTURAL RULES" in text
        assert "RULE 1" in text

    def test_prompt_startup_ping_loads(self) -> None:
        text: str = load_text_resource("prompt_startup_ping.txt")
        assert "health-check" in text
        assert "{model_name}" in text

    def test_prompt_retry_duplicate_paragraph_loads(self) -> None:
        text: str = load_text_resource("prompt_retry_duplicate_paragraph.txt")
        assert "CRITICAL" in text
        assert len(text) > 50

    def test_prompt_retry_overloaded_bullet_loads(self) -> None:
        text: str = load_text_resource("prompt_retry_overloaded_bullet.txt")
        assert "{overloaded_count}" in text

    def test_prompt_retry_paragraph_overflow_loads(self) -> None:
        text: str = load_text_resource("prompt_retry_paragraph_overflow.txt")
        assert "{actual_paragraphs}" in text

    def test_prompt_retry_cta_as_first_paragraph_loads(self) -> None:
        text: str = load_text_resource("prompt_retry_cta_as_first_paragraph.txt")
        assert "CTA" in text

    def test_all_prompt_files_are_nonempty(self) -> None:
        prompt_names: list[str] = [
            "prompt_merge_contract_compact.txt",
            "prompt_merge_contract_expanded.txt",
            "prompt_merge_contract_narrative.txt",
            "prompt_merge_title_description.txt",
            "prompt_merge_structural_rules.txt",
            "prompt_startup_ping.txt",
            "prompt_retry_insufficient_bullet_coverage.txt",
            "prompt_retry_duplicate_paragraph.txt",
            "prompt_retry_overloaded_bullet.txt",
            "prompt_retry_paragraph_overflow.txt",
            "prompt_retry_paragraph_underflow.txt",
            "prompt_retry_cta_as_first_paragraph.txt",
        ]
        for name in prompt_names:
            text: str = load_text_resource(name)
            assert len(text) > 10, f"Prompt file {name} is too short or empty"

    def test_all_lexicon_files_are_nonempty(self) -> None:
        lexicon_names: list[str] = [
            "lexicon_cta_prefixes.txt",
            "lexicon_cta_hints.txt",
            "lexicon_semantic_stopwords.txt",
            "lexicon_official_link_hints.txt",
        ]
        for name in lexicon_names:
            lines: tuple[str, ...] = load_lines_resource(name)
            assert len(lines) >= 5, f"Lexicon file {name} has too few entries"
