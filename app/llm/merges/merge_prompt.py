from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.config.settings import AppConfig
from app.core.text_utils import normalize_multiline_text
from app.resources.resource_loader import load_text_resource
from app.core.models import PlannedVideo
from app.llm.merges.merge_constants import SEMANTIC_TOKEN_PATTERN, URL_PATTERN
from app.llm.merges.merge_retry import (
    ExpandedRetryProfile,
    _format_template_placeholders,
    _merge_contract_block_with_retry,
    _template_field_text,
    _template_json_object,
)
from app.llm.merges.merge_text_utils import (
    _extract_description_paragraphs_raw,
    _extract_named_entities,
    _is_official_links_heading_line,
    _looks_like_service_tail_paragraph,
)

LOGGER = _get_logger_impl(__name__)

@dataclass(frozen=True)
class PreparedMergeSourceDescription:
    text: str
    raw_chars: int
    cleaned_chars: int
    urls_removed: int
    hashtags_removed: int
    service_paragraphs_dropped: int

@dataclass(frozen=True)
class MergeContractMode:
    mode_label: str
    source_count: int
    bullet_range_label: Optional[str]
    bullet_range_min: int
    bullet_range_max: int
    expanded_structure_enabled: bool
    contract_block: str
    max_body_paragraphs: int

def _language_name_for_merge_prompt(language: str, llm_language_names_json: str) -> str:
    default_names: dict[str, str] = {
        "uk": "Ukrainian",
        "en": "English",
        "ru": "Russian",
        "other": "the original language of sources",
    }
    try:
        import json

        payload = json.loads(llm_language_names_json)
        if isinstance(payload, dict):
            return str(payload.get(language, payload.get("other", default_names["other"])))
    except Exception:
        pass
    return default_names.get(language, default_names["other"])

def _normalize_source_description_text(text: str) -> str:
    normalized: str = normalize_multiline_text(text)
    return re.sub(r"\n{3,}", "\n\n", normalized)

def _strip_source_urls_from_text(text: str) -> tuple[str, int]:
    removed_urls: int = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal removed_urls
        removed_urls += 1
        return ""

    cleaned_text: str = URL_PATTERN.sub(_replace, str(text or ""))
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)
    return (cleaned_text.strip(" ,;:-"), removed_urls)

def _strip_source_hashtags_from_text(text: str) -> tuple[str, int]:
    hashtag_pattern: re.Pattern[str] = re.compile(r"(?<!\w)#[^\s#]+", flags=re.UNICODE)
    removed_hashtags: int = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal removed_hashtags
        removed_hashtags += 1
        return ""

    cleaned_text: str = hashtag_pattern.sub(_replace, str(text or ""))
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)
    return (cleaned_text.strip(" ,;:-"), removed_hashtags)

def _clean_source_description_for_llm(text: str) -> PreparedMergeSourceDescription:
    normalized_text: str = _normalize_source_description_text(text)
    raw_chars: int = len(normalized_text)
    if not normalized_text:
        return PreparedMergeSourceDescription(
            text="",
            raw_chars=0,
            cleaned_chars=0,
            urls_removed=0,
            hashtags_removed=0,
            service_paragraphs_dropped=0,
        )

    cleaned_paragraphs: List[str] = []
    urls_removed: int = 0
    hashtags_removed: int = 0
    service_paragraphs_dropped: int = 0

    for paragraph in _extract_description_paragraphs_raw(normalized_text):
        cleaned_lines: List[str] = []
        for raw_line in str(paragraph or "").split("\n"):
            line: str = str(raw_line or "").strip()
            if not line:
                continue
            cleaned_line, line_urls_removed = _strip_source_urls_from_text(line)
            cleaned_line, line_hashtags_removed = _strip_source_hashtags_from_text(
                cleaned_line
            )
            urls_removed += line_urls_removed
            hashtags_removed += line_hashtags_removed
            cleaned_line = re.sub(r"\s{2,}", " ", cleaned_line).strip(" ,;:-")
            if not cleaned_line or _is_official_links_heading_line(cleaned_line):
                continue
            cleaned_lines.append(cleaned_line)

        cleaned_paragraph: str = "\n".join(cleaned_lines).strip()
        if not cleaned_paragraph:
            service_paragraphs_dropped += 1
            continue
        cleaned_paragraphs.append(cleaned_paragraph)

    while cleaned_paragraphs and _looks_like_service_tail_paragraph(cleaned_paragraphs[-1]):
        cleaned_paragraphs.pop()
        service_paragraphs_dropped += 1

    cleaned_text: str = "\n\n".join(
        paragraph for paragraph in cleaned_paragraphs if paragraph.strip()
    ).strip()
    return PreparedMergeSourceDescription(
        text=cleaned_text,
        raw_chars=raw_chars,
        cleaned_chars=len(cleaned_text),
        urls_removed=urls_removed,
        hashtags_removed=hashtags_removed,
        service_paragraphs_dropped=service_paragraphs_dropped,
    )

def _source_texts_for_merge_quality(videos: Sequence[PlannedVideo]) -> tuple[str, ...]:
    return tuple(
        (
            f"{video.metadata.title.strip()}\n"
            f"{_clean_source_description_for_llm(video.metadata.description.strip()).text}"
        ).strip()
        for video in videos
    )

def build_llm_merge_prompt_text(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: AppConfig,
    no_description_text: str,
    expanded_retry_profile: Optional[ExpandedRetryProfile] = None,
) -> str:
    if len(videos) < 2:
        raise ValueError("Expected at least 2 videos for merged generation.")
    language_name: str = _language_name_for_merge_prompt(
        language,
        config.templates.llm_language_names_json,
    )
    source_blocks: List[str] = []
    cleaned_source_texts: List[str] = []
    raw_source_chars_total: int = 0
    cleaned_source_chars_total: int = 0
    for index, video in enumerate(videos, start=1):
        prepared_description: PreparedMergeSourceDescription = _clean_source_description_for_llm(
            video.metadata.description.strip() or no_description_text
        )
        description_for_prompt: str = prepared_description.text or no_description_text
        cleaned_source_texts.append(description_for_prompt)
        raw_source_chars_total += prepared_description.raw_chars
        cleaned_source_chars_total += len(description_for_prompt)
        LOGGER.info(
            "merge_source_text_prepared language=%s source_index=%d row=%s raw_chars=%d cleaned_chars=%d urls_removed=%d hashtags_removed=%d service_paragraphs_dropped=%d hard_truncation=disabled",
            language,
            index,
            getattr(video, "row_number", "unknown"),
            prepared_description.raw_chars,
            len(description_for_prompt),
            prepared_description.urls_removed,
            prepared_description.hashtags_removed,
            prepared_description.service_paragraphs_dropped,
        )
        source_blocks.append(
            "\n".join(
                [
                    f"SOURCE {index}",
                    f"TITLE: {video.metadata.title.strip()}",
                    f"DESCRIPTION: {description_for_prompt}",
                ]
            )
        )
    LOGGER.info(
        "merge_prompt_sources_ready language=%s source_count=%d raw_source_chars_total=%d cleaned_source_chars_total=%d hard_truncation=disabled",
        language,
        len(videos),
        raw_source_chars_total,
        cleaned_source_chars_total,
    )
    contract_mode: MergeContractMode = _select_merge_contract_mode(
        source_count=len(videos),
        source_texts=cleaned_source_texts,
        templates=config.templates,
    )
    LOGGER.info(
        "merge_prompt_contract_selected language=%s source_count=%d contract_mode=%s expected_bullet_range=%s expanded_structure_enabled=%s narrative_trigger=%s",
        language,
        contract_mode.source_count,
        contract_mode.mode_label,
        contract_mode.bullet_range_label,
        "yes" if contract_mode.expanded_structure_enabled else "no",
        "yes" if contract_mode.mode_label == "narrative" else "no",
    )
    contract_block: str = _merge_contract_block_with_retry(
        contract_mode=contract_mode,
        expanded_retry_profile=expanded_retry_profile,
        templates=config.templates,
    )
    link_policy_block: str = (
        "SYSTEM LINK POLICY\n"
        "Do not include any URLs in the output.\n"
        "Do not add a recommended materials block or an official links block.\n"
        "Link blocks will be assembled later by the system."
    )
    cross_domain_sentence_policy_block: str = (
        "CROSS-DOMAIN SENTENCE POLICY\n"
        "If the sources touch different semantic domains, do not compress them into one sentence.\n"
        "Especially do not merge medicine or biology or neurobiology, climate or weather or ecology, disasters or geophysics or natural hazards, psychology or thinking or behavior, and social or moral or civilizational conclusions into one sentence.\n"
        "These topics may stay in one final description, but present them as separate lines of discussion in separate sentences.\n"
        "Do not build one long cause-and-effect chain across all of those domains in a single sentence.\n"
        "Several calm sentences are better than one overloaded super-sentence."
    )
    template_prompt: str = str(config.templates.llm_merge_title_description_prompt or "").strip()
    if template_prompt:
        formatted_template: str = template_prompt.format(
            language_name=language_name,
            sources_block="\n\n".join(source_blocks),
            youtube_candidates_block="",
            merge_contract_block=contract_block,
        ).strip()
        if "{merge_contract_block}" not in template_prompt:
            formatted_template = (
                f"{formatted_template}\n\n{contract_block}"
            ).strip()
        return f"{formatted_template}\n\n{cross_domain_sentence_policy_block}\n\n{link_policy_block}".strip()
    return (
        "You are writing a YouTube stream title and description.\n"
        f"Write output only in {language_name}.\n"
        "Use only facts explicitly present in the source descriptions.\n"
        "Treat the sources as one complete stream, not as a list of separate videos.\n"
        "Generate a new final title, not a copy of any single source title.\n"
        "Do not use emoji in the title.\n"
        "Mentally extract key points from each source, preserve all non-trivial source-specific points,\n"
        "combine overlaps, compress repetition, and produce one coherent final description.\n"
        "Write a strong native YouTube title no longer than 99 characters.\n"
        f"{contract_block}\n"
        "The description must cover all source inputs that were merged.\n"
        "Do not drop a source-specific fact, event, or angle without clear overlap-based reason.\n"
        "Preserve important recognizable names from sources when relevant; never invent names.\n"
        "Avoid asserting strong person titles or role labels unless they are clearly necessary and well-supported by the sources.\n"
        f"{cross_domain_sentence_policy_block}\n"
        f"{link_policy_block}\n"
        "An optional one-line closing sentence should be a light practical CTA with 2 to 5 hashtags.\n"
        "Do not enumerate sources as 1) 2) 3).\n"
        "Do not write a dry digest, protocol, or generic CTA block.\n"
        "Do not output generic slogans, abstract editorial text, or propagandistic phrasing.\n"
        "Do not replace concrete facts with broad statements like 'an important conversation about everything'.\n"
        'Output only one strict JSON object with exactly these keys: title, description.\n\n'
        f"{'\n\n'.join(source_blocks)}"
    ).strip()

def _sources_share_single_event(source_texts: List[str]) -> bool:
    def _extract_named_entities(text: str) -> set[str]:
        words: List[str] = re.split(r'\s+', text)
        entities: set[str] = set()
        current_sequence: List[str] = []
        for word in words:
            cleaned: str = re.sub(r'[^\w]', '', word)
            if cleaned and cleaned[0].isalpha() and cleaned[0].isupper():
                current_sequence.append(cleaned)
            else:
                if len(current_sequence) >= 2:
                    entities.add(' '.join(current_sequence))
                current_sequence = []
        if len(current_sequence) >= 2:
            entities.add(' '.join(current_sequence))
        return entities

    source_entity_sets: List[set[str]] = [_extract_named_entities(text) for text in source_texts]
    all_entities: set[str] = set().union(*source_entity_sets) if source_entity_sets else set()
    overlap_count: int = sum(
        1 for entity in all_entities
        if sum(1 for entity_set in source_entity_sets if entity in entity_set) >= 2
    )
    return overlap_count >= 5

def _build_speaker_anchor_line(*, source_count: int, source_texts: List[str]) -> str:
    if source_count < 4:
        return ""
    speaker_names: List[str] = _extract_source_speaker_names(source_texts)
    if not speaker_names:
        return ""
    min_names: int = min(len(speaker_names), source_count - 1)
    names_joined: str = ", ".join(speaker_names)
    return (
        f"For {source_count} sources surface at least {min_names} distinct named speakers or participants from the list below, covering as many sources as possible. "
        "Do not drop any speaker from this list without clear overlap reason. "
        f"Known names from sources: {names_joined}."
    )

def _extract_source_speaker_names(source_texts: List[str]) -> List[str]:
    all_entities: List[str] = []
    for source_text in source_texts:
        source_entities: set[str] = _extract_named_entities(source_text)
        all_entities.extend(source_entities)
    unique_entities: set[str] = set(all_entities)
    sorted_entities: List[str] = sorted(
        unique_entities,
        key=lambda entity: (-len(entity), entity),
    )
    return sorted_entities[:8]

def _default_expanded_contract_template() -> str:
    return load_text_resource("prompt_merge_contract_expanded.txt")

def _default_narrative_contract_template() -> str:
    return load_text_resource("prompt_merge_contract_narrative.txt")

def _default_compact_contract_template() -> str:
    return load_text_resource("prompt_merge_contract_compact.txt")

def _merge_contract_templates_from_templates(templates: Optional[object]) -> dict[str, str]:
    contract_templates_payload: dict[str, object] = _template_json_object(
        templates,
        "llm_merge_contracts",
    )
    normalized_templates: dict[str, str] = {}
    for template_key in ("compact", "expanded", "narrative"):
        template_text: str = str(contract_templates_payload.get(template_key, "") or "").strip()
        if template_text:
            normalized_templates[template_key] = template_text
    return normalized_templates

def _select_merge_contract_mode(
    *,
    source_count: int,
    source_texts: List[str],
    templates: Optional[object] = None,
) -> MergeContractMode:
    contract_templates_by_mode: dict[str, str] = _merge_contract_templates_from_templates(
        templates
    )
    compact_bullet_min: int = 4
    compact_bullet_max: int = 7
    compact_bullet_range: str = f"{compact_bullet_min}-{compact_bullet_max}"
    if source_count >= 3:
        if source_count <= 3:
            expanded_bullet_min: int = 4
            expanded_bullet_max: int = 6
        elif source_count <= 5:
            expanded_bullet_min = 5
            expanded_bullet_max = 7
        else:
            expanded_bullet_min = 6
            expanded_bullet_max = 9
        expanded_bullet_range_label: str = f"{expanded_bullet_min}-{expanded_bullet_max}"
        speaker_anchor_line: str = _build_speaker_anchor_line(
            source_count=source_count,
            source_texts=source_texts,
        )
        expanded_template: str = contract_templates_by_mode.get(
            "expanded",
            _default_expanded_contract_template(),
        )
        contract_block: str = _format_template_placeholders(
            expanded_template,
            {
                "source_count": source_count,
                "expanded_bullet_min": expanded_bullet_min,
                "expanded_bullet_max": expanded_bullet_max,
                "expanded_bullet_range": expanded_bullet_range_label,
                "compact_bullet_min": compact_bullet_min,
                "compact_bullet_max": compact_bullet_max,
                "compact_bullet_range": compact_bullet_range,
                "speaker_anchor_line": speaker_anchor_line,
            },
        )
        if speaker_anchor_line and "{speaker_anchor_line}" not in expanded_template:
            contract_block = f"{contract_block}\n{speaker_anchor_line}".strip()
        return MergeContractMode(
            mode_label="expanded",
            source_count=source_count,
            bullet_range_label=expanded_bullet_range_label,
            bullet_range_min=expanded_bullet_min,
            bullet_range_max=expanded_bullet_max,
            expanded_structure_enabled=True,
            max_body_paragraphs=7,
            contract_block=contract_block,
        )
    if source_count >= 2 and _sources_share_single_event(source_texts):
        narrative_template: str = contract_templates_by_mode.get(
            "narrative",
            _default_narrative_contract_template(),
        )
        contract_block = _format_template_placeholders(
            narrative_template,
            {
                "source_count": source_count,
                "compact_bullet_min": compact_bullet_min,
                "compact_bullet_max": compact_bullet_max,
                "compact_bullet_range": compact_bullet_range,
            },
        )
        return MergeContractMode(
            mode_label="narrative",
            source_count=source_count,
            bullet_range_label=None,
            bullet_range_min=0,
            bullet_range_max=0,
            expanded_structure_enabled=False,
            max_body_paragraphs=5,
            contract_block=contract_block,
        )
    compact_template: str = contract_templates_by_mode.get(
        "compact",
        _default_compact_contract_template(),
    )
    compact_contract_block: str = _format_template_placeholders(
        compact_template,
        {
            "source_count": source_count,
            "compact_bullet_min": compact_bullet_min,
            "compact_bullet_max": compact_bullet_max,
            "compact_bullet_range": compact_bullet_range,
        },
    )
    return MergeContractMode(
        mode_label="compact",
        source_count=source_count,
        bullet_range_label=compact_bullet_range,
        bullet_range_min=compact_bullet_min,
        bullet_range_max=compact_bullet_max,
        expanded_structure_enabled=False,
        max_body_paragraphs=4,
        contract_block=compact_contract_block,
    )

def _single_source_translate_prompt(
    *,
    source_language: str,
    target_language: str,
    source_description: str,
) -> str:
    template: str = load_text_resource("prompt_single_source_translate.txt")
    return template.format(
        source_language=source_language,
        target_language=target_language,
        source_description=source_description.strip(),
    ).strip()
