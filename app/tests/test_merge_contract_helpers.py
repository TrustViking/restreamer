from __future__ import annotations

import json
import unittest
from types import SimpleNamespace


class MergeContractServiceBase(unittest.TestCase):
    """Shared helpers for merge contract service tests."""

    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
            llm=SimpleNamespace(
                provider="openai",
                model="gpt-5.1",
                timeout_sec=30.0,
                max_output_tokens=1000,
                pre_delay_sec=0.0,
                source_desc_max_chars=500,
                run_if_single_source=False,
            ),
            templates=SimpleNamespace(
                llm_language_names_json='{"en":"English"}',
                llm_language_names={"en": "English"},
                llm_merge_title_description_prompt="""
Write a YouTube stream title and description in {language_name}.
Generate a new final title, not a copy of any single source title.
Mentally extract key points from each source, preserve all non-trivial source-specific points,
combine overlaps, compress repetition, and produce one coherent final description.
Write a strong native YouTube title no longer than 99 characters.
Do not enumerate sources as 1) 2) 3).
Do not output generic slogans or abstract editorial phrasing.
Do not use emoji in the title.
{merge_contract_block}
Avoid asserting strong person titles or role labels unless they are clearly necessary and well-supported by the sources.
Optional official links block is allowed before close line with 1 to 3 non-YouTube links from sources.
An optional one-line close should be practical CTA + 2 to 5 hashtags.
Return strict JSON with title and description only.

{youtube_candidates_block}

{sources_block}
""".strip(),
                llm_merge_structural_rules=(
                    "MERGE STRUCTURAL RULES\n"
                    "RULE 1: Start with a standalone hook paragraph before any bullets.\n"
                    "RULE 2: Keep visual paragraph boundaries explicit with one blank line between structural blocks.\n"
                    "RULE 3: Keep CTA and hashtags only in the final tail position, never as opener lines.\n"
                    "RULE 4: Do not repeat or paraphrase the hook thesis in the next adjacent line or paragraph.\n"
                    "EXAMPLE A (bad): CTA line opens the description and the real hook starts later.\n"
                    "EXAMPLE A (good): Hook opens first, CTA appears only at the end.\n"
                    "EXAMPLE B (bad): Two adjacent lines restate the same thesis with minor wording changes.\n"
                    "EXAMPLE B (good): The second line introduces new facts instead of repeating the opener."
                ),
                llm_merge_contracts_json=json.dumps(
                    {
                        "compact": (
                            "Use the compact merge contract for 1 to 2 source items.\n"
                            "Write one cohesive stream description in 2 to 3 compact paragraphs.\n"
                            "Paragraph 1 (hook): write 1 to 2 sentences grounded in the main tension, risk, or key conflict.\n"
                            "Keep the hook editorial and readable, but never clickbait.\n"
                            "Paragraph 2 (theses block): open with one short editorial statement that names the central tension, key question, or main conflict — not a lead-in phrase like 'In this stream you will see'.\n"
                            "Then write {compact_bullet_min} to {compact_bullet_max} short thesis bullet lines (target range {compact_bullet_range}).\n"
                            "Each bullet line must start with exactly one allowed marker: 🔹 📌 🎤 🎥 ⚖ 🌐 ✅.\n"
                            "Most bullets should start with 🔹.\n"
                            "Accent markers are rare and optional; use no more than 3 accent markers per theses block.\n"
                            "Keep marker usage controlled and readable; do not use dash-only bullets as the sole style.\n"
                            "Do not present the agenda as SOURCE 1 / SOURCE 2.\n"
                            "Keep agenda points specific and factual, not generic placeholders.\n"
                            "The description must still cover all merged source items and preserve key concrete facts from each source."
                        ),
                        "expanded": (
                            "Use the expanded merge contract for 3 or more source items.\n"
                            "Write one cohesive stream description in 3 to 4 compact paragraphs.\n"
                            "Paragraph 1 (hook): write 1 to 2 sentences grounded in the main tension, risk, or key conflict.\n"
                            "Keep the hook editorial and readable, but never clickbait.\n"
                            "After the hook, use a more open agenda structure instead of one overloaded thesis block.\n"
                            "Write {expanded_bullet_min} to {expanded_bullet_max} short bullet lines total.\n"
                            "You may organize the bullets into 2 to 3 thematic micro-blocks when that improves clarity. Separate each thematic micro-block from the next with a blank line.\n"
                            "IMPORTANT: Bullet lines within the same thematic micro-block must be separated by single newlines (\\n), NOT by blank lines (\\n\\n). A blank line starts a new paragraph. The entire bullet section should be at most 2 visual paragraphs.\n"
                            "Each bullet line must start with exactly one allowed marker: 🔹 📌 🎤 🎥 ⚖ 🌐 ✅.\n"
                            "Most bullets should start with 🔹.\n"
                            "Accent markers are rare and optional; use no more than 3 accent markers per description.\n"
                            "Keep marker usage controlled and readable; do not use dash-only bullets as the sole style.\n"
                            "Do not present the agenda as SOURCE 1 / SOURCE 2 / SOURCE 3.\n"
                            "Keep agenda points specific and factual, not generic placeholders.\n"
                            "Do not let the opening hook consume most of the useful summary space.\n"
                            "A polished opening is never a substitute for a concrete multi-angle summary.\n"
                            "Treat the post-hook body as the main payload and let it carry most of the concrete information.\n"
                            "Make most bullets fact-bearing: anchor them with names, places, institutions, numbers, timings, events, or operational consequences whenever the sources provide them.\n"
                            "Give the body at least two clearly substantive agenda lanes after the hook instead of one thin run of near-duplicate bullets.\n"
                            "Across the agenda, preserve distinguishable source details such as names, places, numbers, events, or clearly separate thematic nodes whenever the sources provide them.\n"
                            "Across 3 or more sources, spread the bullets across multiple source lines or topic nodes so the summary does not collapse into one generic lane.\n"
                            "If the merged sources span different domains, separate them across different bullets or short thematic blocks instead of compressing them into one universal bullet.\n"
                            "Do not combine science or medicine, climate or environment, disasters or catastrophic hazards, psychology or cognition or behavior, and broad social or moral conclusions into one bullet or one cause-and-effect chain unless the sources explicitly require that connection.\n"
                            "Thematic grouping is encouraged when useful: research, hazards, human behavior, practical risk, public meaning, or response can be separated into different bullets or micro-blocks.\n"
                            "The description must still cover all merged source items and preserve key concrete facts from each source.\n"
                            "{speaker_anchor_line}"
                        ),
                        "narrative": (
                            "This stream covers a single unified event or case. "
                            "Write the description as connected prose, not a bullet list. "
                            "Hook paragraph first, then 2-3 prose paragraphs. No bullets."
                        ),
                    },
                    ensure_ascii=False,
                ),
                llm_merge_retry_reinforcements_json=json.dumps({}, ensure_ascii=False),
            ),
        )

    def _videos(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Title 1",
                    description="Paragraph one.\n\nParagraph two.",
                ),
                normalized_link="https://youtube.com/watch?v=aaaaaaaaaaa",
            ),
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Title 2",
                    description="Paragraph three.\n\nParagraph four.",
                ),
                normalized_link="https://youtube.com/watch?v=bbbbbbbbbbb",
            ),
        ]

    def _videos_three_sources(self) -> list[SimpleNamespace]:
        return [
            *self._videos(),
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Title 3",
                    description="Paragraph five.\n\nParagraph six.",
                ),
                normalized_link="https://youtube.com/watch?v=ccccccccccc",
            ),
        ]

    def _expanded_validation_videos(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Brussels sanctions vote briefing",
                    description=(
                        "In Brussels, Anna Kovalenko tracks the March 18 sanctions vote, "
                        "budget amendments, and customs delays after the commission session."
                    ),
                ),
                normalized_link="https://youtube.com/watch?v=expandedsource01",
            ),
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Kharkiv rail and drone update",
                    description=(
                        "In Kharkiv, Oleh Martynenko reports 17 drone strikes, rail hub outages, "
                        "and evacuation routes for Saltivka districts."
                    ),
                ),
                normalized_link="https://youtube.com/watch?v=expandedsource02",
            ),
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Geneva relief corridor desk",
                    description=(
                        "From Geneva, Marta Leone outlines the aid corridor timetable, WHO cargo "
                        "counts, and donor pledges for Odesa and Mykolaiv hospitals."
                    ),
                ),
                normalized_link="https://youtube.com/watch?v=expandedsource03",
            ),
        ]

    def _expanded_validation_videos_four_sources(self) -> list[SimpleNamespace]:
        return [
            *self._expanded_validation_videos(),
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Lviv grid repair logistics",
                    description=(
                        "In Lviv, Iryna Melnyk details transformer shortages, repair crews, "
                        "and the grid restoration queue after regional substation damage."
                    ),
                ),
                normalized_link="https://youtube.com/watch?v=expandedsource04",
            ),
        ]

    def _make_merge_response(
        self,
        *,
        title: str,
        description: str,
    ) -> SimpleNamespace:
        structured_payload: dict[str, str] = {
            "title": title,
            "description": description,
        }
        raw_text: str = json.dumps(structured_payload, ensure_ascii=False)
        return SimpleNamespace(
            raw_text=raw_text,
            structured_payload=structured_payload,
        )

    def _expanded_description_with_body_paragraphs(
        self,
        *,
        body_paragraphs: int,
    ) -> str:
        if body_paragraphs < 2:
            raise ValueError("body_paragraphs must be at least 2")
        hook_paragraph: str = (
            "Tonight we track how the Brussels vote, Kharkiv transport shocks, and Geneva aid timing now intersect: "
            "each lane carries concrete operational consequences for viewers following this agenda."
        )
        bullet_paragraphs: list[str] = [
            (
                "In this stream you'll see:\n"
                "🔹 Brussels sanctions vote and budget amendments after the March 18 commission session\n"
                "🔹 Anna Kovalenko maps coalition counts and customs pressure before the chamber debate"
            ),
            "🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts.",
            "🔹 Oleh Martynenko details evacuation routes and depot repair sequencing on the eastern line.",
            "🔹 Geneva aid corridor timetable, WHO cargo counts, and donor pledges for Odesa hospitals.",
            "🔹 Marta Leone explains how Mykolaiv deliveries depend on the next donor release window.",
            "🔹 Lviv transformer shipments and repair crew rotations now define the overnight recovery queue.",
            "🔹 Baltic cargo reroutes are changing fuel timing and insurance windows for regional logistics.",
            "🔹 Emergency procurement updates now tie Brussels financing signals to corridor-level medical deliveries.",
        ]
        required_bullet_paragraphs: int = body_paragraphs - 1
        if required_bullet_paragraphs > len(bullet_paragraphs):
            raise ValueError("requested body_paragraphs exceeds test fixture capacity")
        selected_bullet_paragraphs: list[str] = bullet_paragraphs[:required_bullet_paragraphs]
        return "\n\n".join([hook_paragraph, *selected_bullet_paragraphs])


if __name__ == "__main__":
    unittest.main()
