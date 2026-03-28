from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.llm.merges.merge_prompt import build_llm_merge_prompt_text
from app.llm.merges.merge_retry import _build_expanded_retry_profile

from app.tests.test_merge_contract_helpers import MergeContractServiceBase


class MergeContractPromptTests(MergeContractServiceBase):
    """Tests for prompt building concerns of merge contract."""

    def test_prompt_targets_youtube_title_and_description_only(self) -> None:
        prompt_text: str = build_llm_merge_prompt_text(
            language="en",
            videos=self._videos(),
            config=self._config(),
            no_description_text="no description",
        )
        self.assertIn("YouTube stream title and description", prompt_text)
        self.assertIn("99 characters", prompt_text)
        self.assertIn("title and description only", prompt_text)
        self.assertIn(
            "must still cover all merged source items and preserve key concrete facts from each source",
            prompt_text,
        )
        self.assertIn("Do not output generic slogans", prompt_text)
        self.assertIn("Use the compact merge contract for 1 to 2 source items.", prompt_text)
        self.assertIn("2 to 3 compact paragraphs", prompt_text)
        self.assertIn("Then write 4 to 7 short thesis bullet lines", prompt_text)
        self.assertIn("allowed marker", prompt_text)
        self.assertIn("Most bullets should start with 🔹", prompt_text)
        self.assertIn("no more than 3 accent markers", prompt_text)
        self.assertIn("Avoid asserting strong person titles", prompt_text)
        self.assertIn("If the sources touch different semantic domains, do not compress them into one sentence.", prompt_text)
        self.assertIn("These topics may stay in one final description, but present them as separate lines of discussion in separate sentences.", prompt_text)
        self.assertIn("Do not build one long cause-and-effect chain across all of those domains in a single sentence.", prompt_text)
        self.assertIn("Do not include any URLs in the output.", prompt_text)
        self.assertIn("Link blocks will be assembled later by the system.", prompt_text)
        self.assertIn("Do not use emoji in the title.", prompt_text)
        self.assertIn("optional one-line close", prompt_text.lower())
        self.assertNotIn("URL:", prompt_text)
        self.assertIn("Paragraph one.\n\nParagraph two.", prompt_text)

    def test_prompt_uses_expanded_contract_for_three_or_more_sources(self) -> None:
        with self.assertLogs(level="INFO") as captured:
            prompt_text: str = build_llm_merge_prompt_text(
                language="en",
                videos=self._videos_three_sources(),
                config=self._config(),
                no_description_text="no description",
            )

        joined_logs: str = "\n".join(captured.output)
        self.assertIn("Use the expanded merge contract for 3 or more source items.", prompt_text)
        self.assertIn("Write 4 to 6 short bullet lines total.", prompt_text)
        self.assertIn("2 to 3 thematic micro-blocks", prompt_text)
        self.assertIn(
            "Do not combine science or medicine, climate or environment, disasters or catastrophic hazards, psychology or cognition or behavior, and broad social or moral conclusions into one bullet",
            prompt_text,
        )
        self.assertIn("Thematic grouping is encouraged when useful", prompt_text)
        self.assertIn(
            "merge_prompt_contract_selected language=en source_count=3 contract_mode=expanded expected_bullet_range=4-6 expanded_structure_enabled=yes",
            joined_logs,
        )

    def test_compact_contract_is_read_from_templates_and_exposes_4_7_range(self) -> None:
        config: SimpleNamespace = self._config()
        config.templates.llm_merge_contracts_json = json.dumps(
            {
                "compact": (
                    "TEMPLATE COMPACT CONTRACT\n"
                    "Use compact bullet range {compact_bullet_range} for compact mode."
                ),
                "expanded": "Expanded template {expanded_bullet_min}-{expanded_bullet_max}",
                "narrative": "Narrative template",
            },
            ensure_ascii=False,
        )
        prompt_text: str = build_llm_merge_prompt_text(
            language="en",
            videos=self._videos(),
            config=config,
            no_description_text="no description",
        )
        self.assertIn("TEMPLATE COMPACT CONTRACT", prompt_text)
        self.assertIn("compact bullet range 4-7", prompt_text)

    def test_compact_prompt_ignores_expanded_retry_profile(self) -> None:
        retry_profile = _build_expanded_retry_profile(
            source_count=3,
            reject_signals=("insufficient_expanded_body",),
        )
        prompt_text: str = build_llm_merge_prompt_text(
            language="en",
            videos=self._videos(),
            config=self._config(),
            no_description_text="no description",
            expanded_retry_profile=retry_profile,
        )
        self.assertNotIn("EXPANDED RETRY FOCUS", prompt_text)
        self.assertIn("Use the compact merge contract for 1 to 2 source items.", prompt_text)

    def test_merge_prompt_uses_clean_full_source_text_without_urls_hashtags_or_truncation(self) -> None:
        videos = [
            SimpleNamespace(
                row_number=1,
                metadata=SimpleNamespace(
                    title="Source 1",
                    description=(
                        "Hook paragraph with concrete facts and named people. "
                        + ("A" * 2600)
                        + "\n\n"
                        "Main stream link https://youtu.be/aaaaaaaaaaa\n"
                        "Official links:\nhttps://example.org/details\n\n"
                        "Join and follow updates. #topic #update"
                    ),
                ),
                normalized_link="https://youtube.com/watch?v=sourcevideo01",
            ),
            SimpleNamespace(
                row_number=2,
                metadata=SimpleNamespace(
                    title="Source 2",
                    description="Second source keeps the semantic context intact without extra links.",
                ),
                normalized_link="https://youtube.com/watch?v=sourcevideo02",
            ),
        ]
        with self.assertLogs(level="INFO") as captured:
            prompt_text: str = build_llm_merge_prompt_text(
                language="en",
                videos=videos,
                config=self._config(),
                no_description_text="no description",
            )
        self.assertNotIn("https://youtu.be/aaaaaaaaaaa", prompt_text)
        self.assertNotIn("https://example.org/details", prompt_text)
        self.assertNotIn("#topic", prompt_text)
        self.assertNotIn("Official links:", prompt_text)
        self.assertNotIn("Join and follow updates.", prompt_text)
        self.assertIn("Hook paragraph with concrete facts and named people.", prompt_text)
        self.assertIn("Second source keeps the semantic context intact without extra links.", prompt_text)
        self.assertIn("Do not add a recommended materials block", prompt_text)
        self.assertIn("A" * 2400, prompt_text)
        self.assertNotIn("YOUTUBE CANDIDATES", prompt_text)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("hard_truncation=disabled", joined_logs)
        self.assertIn("merge_source_text_prepared language=en source_index=1", joined_logs)
        self.assertIn("urls_removed=", joined_logs)
        self.assertIn("hashtags_removed=", joined_logs)


if __name__ == "__main__":
    unittest.main()
