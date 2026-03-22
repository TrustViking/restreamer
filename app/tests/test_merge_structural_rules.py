from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.config.template_loader import load_templates_from_path
from app.llm.merges.merge_prompt import build_llm_merge_prompt_text
from app.llm.merges.merge_retry import _targeted_hook_echo_retry_profile


class MergeStructuralRulesPromptTests(unittest.TestCase):
    def _make_test_config(self, *, use_repo_templates: bool = False) -> SimpleNamespace:
        if use_repo_templates:
            return SimpleNamespace(
                templates=load_templates_from_path(Path("app/llm/prompts/templates.yaml"))
            )
        return self._config()

    def _make_test_videos(self, count: int) -> list[SimpleNamespace]:
        videos: list[SimpleNamespace] = []
        for index in range(1, count + 1):
            videos.append(
                self._video(
                    title=f"Title {index}",
                    description=(
                        f"Source {index} paragraph one with concrete facts and names.\n\n"
                        f"Source {index} paragraph two with extra details and timing."
                    ),
                )
            )
        return videos

    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
            templates=SimpleNamespace(
                llm_language_names_json='{"en":"English"}',
                llm_merge_title_description_prompt=(
                    "Write in {language_name}.\n"
                    "{merge_contract_block}\n\n"
                    "{youtube_candidates_block}\n\n"
                    "{sources_block}"
                ),
                llm_merge_structural_rules=(
                    "STRUCTURAL RULES — APPLY TO EVERY MERGE OUTPUT\n\n"
                    "RULE 1 — NO CTA AS OPENER:\n"
                    "The first paragraph MUST be the editorial hook.\n\n"
                    "RULE 2 — HOOK UNIQUENESS:\n"
                    "The hook paragraph appears exactly ONCE — as the first paragraph.\n\n"
                    "RULE 3 — BULLET COUNT DISCIPLINE:\n"
                    "The number of bullets must stay within the range specified by the contract.\n\n"
                    "RULE 4 — PARAGRAPH 2 STARTS WITH A BULLET:\n"
                    "Immediately after the hook, the next content must be the bullet block.\n\n"
                    "EXAMPLE — CORRECT structure (uk):\n"
                    "Hook paragraph here.\n\n"
                    "EXAMPLE — WRONG structure (3 violations):\n"
                    "CTA as first paragraph."
                ),
                llm_merge_contracts_json=json.dumps(
                    {
                        "compact": "COMPACT CONTRACT {compact_bullet_range}",
                        "expanded": "EXPANDED CONTRACT {expanded_bullet_min}-{expanded_bullet_max}",
                        "narrative": "NARRATIVE CONTRACT",
                    },
                    ensure_ascii=False,
                ),
                llm_merge_retry_reinforcements_json=json.dumps({}, ensure_ascii=False),
            ),
        )

    def _video(self, *, title: str, description: str) -> SimpleNamespace:
        return SimpleNamespace(
            metadata=SimpleNamespace(
                title=title,
                description=description,
            ),
            normalized_link="https://youtube.com/watch?v=aaaaaaaaaaa",
        )

    def _assert_structural_rules_present(self, prompt_text: str) -> None:
        self.assertIn("STRUCTURAL RULES", prompt_text)
        self.assertIn("RULE 1 — NO CTA AS OPENER", prompt_text)
        self.assertIn("RULE 2 — HOOK UNIQUENESS", prompt_text)
        self.assertIn("RULE 3 — BULLET COUNT DISCIPLINE", prompt_text)
        self.assertIn("RULE 4 — PARAGRAPH 2 STARTS WITH A BULLET", prompt_text)
        self.assertIn("EXAMPLE — CORRECT structure", prompt_text)
        self.assertIn("EXAMPLE — WRONG structure", prompt_text)

    def test_structural_rules_are_appended_for_compact_mode(self) -> None:
        videos: list[SimpleNamespace] = [
            self._video(title="One", description="low overlap lowercase text one"),
            self._video(title="Two", description="low overlap lowercase text two"),
        ]
        prompt_text: str = build_llm_merge_prompt_text(
            language="en",
            videos=videos,
            config=self._config(),
            no_description_text="no description",
        )
        self.assertIn("COMPACT CONTRACT 4-7", prompt_text)
        self._assert_structural_rules_present(prompt_text)

    def test_structural_rules_are_appended_for_expanded_mode(self) -> None:
        videos: list[SimpleNamespace] = [
            self._video(title="One", description="source text one"),
            self._video(title="Two", description="source text two"),
            self._video(title="Three", description="source text three"),
        ]
        prompt_text: str = build_llm_merge_prompt_text(
            language="en",
            videos=videos,
            config=self._config(),
            no_description_text="no description",
        )
        self.assertIn("EXPANDED CONTRACT 4-6", prompt_text)
        self._assert_structural_rules_present(prompt_text)

    def test_structural_rules_are_appended_for_narrative_mode(self) -> None:
        narrative_description: str = (
            "John Smith met Mary Jones while Alex Brown, Nina White, and Oleg Ivanov "
            "reported from the same event."
        )
        videos: list[SimpleNamespace] = [
            self._video(title="One", description=narrative_description),
            self._video(title="Two", description=narrative_description),
        ]
        with patch("app.llm.merges.merge_prompt._sources_share_single_event", return_value=True):
            prompt_text = build_llm_merge_prompt_text(
                language="en",
                videos=videos,
                config=self._config(),
                no_description_text="no description",
            )
        self.assertIn("NARRATIVE CONTRACT", prompt_text)
        self._assert_structural_rules_present(prompt_text)

    def test_prompt_contains_hook_sharpness_examples(self) -> None:
        """Verify that the merged prompt includes ranked hook examples."""
        prompt_text: str = build_llm_merge_prompt_text(
            language="uk",
            videos=self._make_test_videos(3),
            config=self._make_test_config(use_repo_templates=True),
            no_description_text="(no description)",
        )
        self.assertIn("HOOK SHARPNESS", prompt_text)
        self.assertIn("STRONG", prompt_text)
        self.assertIn("WEAK", prompt_text)

    def test_hook_echo_uses_dedicated_retry_profile(self) -> None:
        """hook_echo_in_body must route to its own retry profile, not to generic duplicate."""
        profile = _targeted_hook_echo_retry_profile()
        self.assertEqual(profile.reject_signals, ("hook_echo_in_body",))
        self.assertIn("no_hook_echo", profile.focus_tags)
        self.assertTrue(profile.enabled)
        combined: str = " ".join(profile.reinforcement_lines)
        self.assertIn("hook", combined.lower())

    def test_prompt_contains_multilingual_hook_sharpness_examples(self) -> None:
        """Verify that MEDIUM/WEAK examples exist for all three languages."""
        prompt_text: str = build_llm_merge_prompt_text(
            language="ru",
            videos=self._make_test_videos(3),
            config=self._make_test_config(use_repo_templates=True),
            no_description_text="(no description)",
        )
        self.assertIn("MEDIUM (ru)", prompt_text)
        self.assertIn("WEAK (ru)", prompt_text)
        self.assertIn("MEDIUM (en)", prompt_text)
        self.assertIn("WEAK (en)", prompt_text)


if __name__ == "__main__":
    unittest.main()
