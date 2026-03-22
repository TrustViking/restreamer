from __future__ import annotations

import re
from typing import List, Optional

from app.llm.merges.merge_constants import ALLOWED_BULLET_MARKERS, SEMANTIC_TOKEN_PATTERN

_BAD_HOOK_PATTERNS: tuple[str, ...] = (
    "наш канал",
    "наш некомерційний",
    "наш неприбутковий",
    "наш неприбутков",
    "не просуває",
    "не пропагує",
    "не продвигает",
    "не пропагандирует",
    "our channel",
    "our nonprofit",
    "if you want more",
    "if you'd like more",
    "хочете продовження",
    "хотите продолжения",
    "write in the comments",
    "напишіть у коментарях",
    "напишите в комментариях",
    "поширюйте",
    "поділіться",
    "приєднуйтесь",
    "stay tuned",
    "поделитесь",
    "смотрите полный",
    "watch the full",
    "follow the full",
    "если вы смотрели стрим",
    "если вы смотрели стрим, напишите",
    # Meta-editorial openings: preamble instead of a factual hook.
    "матеріал подано",
    "матеріал представлено",
    "матеріал підготовлено",
    "матеріал розміщено",
    "цей матеріал є частиною",
    "цей матеріал подано",
    "матеріал публікується",
    "материал подан",
    "материал представлен",
    "материал подготовлен",
    "материал публикуется",
    "этот материал является частью",
    "данный материал",
    "this material is presented",
    "this material is part of",
    "this content is presented",
    "this video is part of",
    "presented as part of",
    "in the context of",
    "within the context of",
    "as part of an ongoing",
    "в контексті",
    "в рамках обговорення",
    "в рамках обсуждения",
    "в рамках розслідування",
    "в рамках расследования",
    "в рамках документального",
)


def extract_description_paragraphs_raw(text: str) -> List[str]:
    normalized_text: str = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized_text:
        return []
    return [part.strip() for part in re.split(r"\n\s*\n", normalized_text) if part.strip()]


def _normalize_similarity_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _semantic_token_set(text: str) -> set[str]:
    return {
        token.lower()
        for token in SEMANTIC_TOKEN_PATTERN.findall(str(text or ""))
        if token.strip()
    }


def looks_like_bad_hook_paragraph(text: str) -> bool:
    normalized_text: str = re.sub(r"\s+", " ", str(text or "").strip()).lower()
    if not normalized_text:
        return False
    return any(pattern in normalized_text for pattern in _BAD_HOOK_PATTERNS)


def has_duplicate_paragraphs(description_text: str) -> bool:
    paragraphs: List[str] = extract_description_paragraphs_raw(description_text)
    seen_normalized_paragraphs: set[str] = set()
    token_sets: List[set[str]] = []
    for paragraph in paragraphs:
        normalized_paragraph: str = _normalize_similarity_text(paragraph)
        token_list: List[str] = [token for token in normalized_paragraph.split(" ") if token]
        if len(token_list) < 6:
            continue
        if normalized_paragraph in seen_normalized_paragraphs:
            return True
        seen_normalized_paragraphs.add(normalized_paragraph)
        current_token_set: set[str] = set(token_list)
        for previous_token_set in token_sets:
            union_size: int = len(current_token_set | previous_token_set)
            if union_size == 0:
                continue
            intersection_size: int = len(current_token_set & previous_token_set)
            jaccard_similarity: float = intersection_size / union_size
            if jaccard_similarity >= 0.72:
                return True
        token_sets.append(current_token_set)
    return False


def _common_prefix_ratio(first_text: str, second_text: str) -> float:
    shorter_length: int = min(len(first_text), len(second_text))
    if shorter_length <= 0:
        return 0.0
    common_prefix_length: int = 0
    for first_char, second_char in zip(first_text, second_text):
        if first_char != second_char:
            break
        common_prefix_length += 1
    return common_prefix_length / shorter_length


def has_hook_echo_in_body(description_text: str) -> bool:
    """Detect when hook thesis is repeated at the start of the bullet block."""
    paragraphs: List[str] = extract_description_paragraphs_raw(description_text)
    if len(paragraphs) < 2:
        return False
    hook_text: str = paragraphs[0].strip()
    body_opener_line: str = paragraphs[1].split("\n")[0].strip()
    if len(hook_text) < 40 or len(body_opener_line) < 40:
        return False
    shorter_length: int = min(len(hook_text), len(body_opener_line))
    common_prefix_length: int = 0
    for hook_char, body_char in zip(hook_text, body_opener_line):
        if hook_char != body_char:
            break
        common_prefix_length += 1
    prefix_ratio: float = common_prefix_length / shorter_length
    if prefix_ratio > 0.50:
        return True
    hook_tokens: set[str] = _semantic_token_set(hook_text)
    body_tokens: set[str] = _semantic_token_set(body_opener_line)
    union_size: int = len(hook_tokens | body_tokens)
    if union_size == 0:
        return False
    hook_sentences: List[str] = [
        sentence.strip()
        for sentence in re.split(r"[.!?]\s+|\n", hook_text)
        if sentence.strip()
    ]
    if hook_sentences:
        last_hook_sentence: str = hook_sentences[-1]
        if len(last_hook_sentence) >= 30 and len(body_opener_line) >= 30:
            last_sentence_tokens: set[str] = _semantic_token_set(last_hook_sentence)
            body_opener_tokens_for_sentence: set[str] = _semantic_token_set(body_opener_line)
            sentence_union_size: int = len(
                last_sentence_tokens | body_opener_tokens_for_sentence
            )
            if sentence_union_size > 0:
                sentence_jaccard: float = len(
                    last_sentence_tokens & body_opener_tokens_for_sentence
                ) / sentence_union_size
                if sentence_jaccard >= 0.55:
                    return True
    jaccard_similarity: float = len(hook_tokens & body_tokens) / union_size
    return jaccard_similarity >= 0.60


def _line_starts_with_bullet(line_text: str) -> bool:
    stripped_line: str = line_text.strip()
    return any(stripped_line.startswith(marker) for marker in ALLOWED_BULLET_MARKERS)


def _split_hook_trailing_bullet(hook_paragraph: str) -> tuple[str, Optional[str]]:
    """Detect and split a bullet marker that was fused into the hook paragraph.

    Mode A: multi-line hook — the first subsequent line starting with a bullet
    marker is treated as the fused bullet; everything before it is the clean hook.

    Mode B: single-line hook — searches for the last sentence-ending punctuation
    ('. ', '! ', '? ') that appears immediately before a bullet marker (only
    whitespace between them), at a position > 40 characters into the line.
    Uses rfind so the split point is as close as possible to the fused bullet.

    Returns (clean_hook, extracted_bullet).  extracted_bullet is None when no
    fused bullet is found.
    """
    hook_lines: List[str] = hook_paragraph.split("\n")

    # Mode A: multi-line hook
    if len(hook_lines) > 1:
        for line_index in range(1, len(hook_lines)):
            line_text: str = hook_lines[line_index]
            if _line_starts_with_bullet(line_text):
                clean_hook_multiline: str = "\n".join(hook_lines[:line_index]).strip()
                extracted_bullet_multiline: str = line_text.strip()
                return clean_hook_multiline, extracted_bullet_multiline
        return hook_paragraph, None

    # Mode B: single-line hook
    single_line: str = hook_paragraph.strip()
    earliest_bullet_position: Optional[int] = None
    for marker in ALLOWED_BULLET_MARKERS:
        marker_position: int = single_line.find(marker)
        if marker_position > 0:
            if earliest_bullet_position is None or marker_position < earliest_bullet_position:
                earliest_bullet_position = marker_position

    if earliest_bullet_position is None:
        return hook_paragraph, None

    prefix_text: str = single_line[:earliest_bullet_position]
    best_cut_position: Optional[int] = None

    for punctuation_sequence in (". ", "! ", "? "):
        candidate_position: int = prefix_text.rfind(punctuation_sequence)
        if candidate_position < 40:
            continue
        # Only accept if nothing but whitespace separates the punctuation end from the bullet
        between_text: str = single_line[
            candidate_position + len(punctuation_sequence):earliest_bullet_position
        ]
        if between_text.strip() != "":
            continue
        # +1 to include the punctuation character itself (exclude the trailing space)
        cut_after: int = candidate_position + 1
        if best_cut_position is None or cut_after > best_cut_position:
            best_cut_position = cut_after

    if best_cut_position is None:
        return hook_paragraph, None

    clean_hook_single: str = single_line[:best_cut_position].strip()
    extracted_bullet_single: str = single_line[earliest_bullet_position:].strip()

    if not _line_starts_with_bullet(extracted_bullet_single):
        return hook_paragraph, None

    return clean_hook_single, extracted_bullet_single


def attempt_hook_echo_repair(description_text: str) -> Optional[str]:
    """Attempt to surgically repair a description where paragraph 2 echoes the hook.

    Structural repair only: strips the echoed prose prefix from paragraph 2 and
    preserves all existing bullet/body text verbatim from the first bullet onward.
    Returns the repaired description string, or None if repair is not possible.

    Two-pass strategy:
    - First pass: strip the echo prefix from paragraph 2.
    - Second pass (if first pass leaves the hook echo intact): split any bullet
      that was fused into the hook line, with deduplication against the body.
    """
    paragraphs: List[str] = extract_description_paragraphs_raw(description_text)
    if len(paragraphs) < 2:
        return None
    hook_paragraph: str = paragraphs[0]
    echo_paragraph: str = paragraphs[1]
    remaining_paragraphs: List[str] = paragraphs[2:]

    # Check if paragraph 2 actually echoes the hook
    check_text: str = f"{hook_paragraph}\n\n{echo_paragraph}\n\nPlaceholder paragraph for length."
    if not has_hook_echo_in_body(check_text):
        return None

    # Try to find a bullet line within paragraph 2 and trim to it
    echo_lines: List[str] = echo_paragraph.split("\n")
    first_bullet_index: Optional[int] = None
    for line_index, line_text in enumerate(echo_lines):
        if _line_starts_with_bullet(line_text):
            first_bullet_index = line_index
            break

    repaired_description: str
    if first_bullet_index is not None:
        bullet_lines: List[str] = echo_lines[first_bullet_index:]
        trimmed_echo_block: str = "\n".join(bullet_lines)
        parts: List[str] = [hook_paragraph, trimmed_echo_block] + remaining_paragraphs
        repaired_description = "\n\n".join(parts)
    else:
        # No bullet in paragraph 2 — try using paragraph 3 as the bullet block
        if not remaining_paragraphs:
            return None
        next_paragraph: str = remaining_paragraphs[0]
        if not _line_starts_with_bullet(next_paragraph):
            return None
        parts = [hook_paragraph] + remaining_paragraphs
        repaired_description = "\n\n".join(parts)

    # First-pass verification
    if not has_hook_echo_in_body(repaired_description):
        return repaired_description

    # Second pass: the hook still contains a fused bullet — try to split it out
    repaired_paragraphs: List[str] = extract_description_paragraphs_raw(repaired_description)
    if len(repaired_paragraphs) < 2:
        return None
    repaired_hook: str = repaired_paragraphs[0]
    repaired_body_block: str = repaired_paragraphs[1]
    remaining_after_body: List[str] = repaired_paragraphs[2:]

    clean_hook_result: str
    extracted_bullet_result: Optional[str]
    clean_hook_result, extracted_bullet_result = _split_hook_trailing_bullet(repaired_hook)

    if extracted_bullet_result is None:
        return None

    # Dedup check: if extracted bullet is identical to the first body bullet, do not prepend
    body_bullet_lines: List[str] = [
        line.strip()
        for line in repaired_body_block.split("\n")
        if _line_starts_with_bullet(line)
    ]

    new_body_block: str
    if body_bullet_lines:
        first_body_bullet_normalized: str = re.sub(r"\s+", " ", body_bullet_lines[0])
        extracted_normalized: str = re.sub(r"\s+", " ", extracted_bullet_result.strip())
        if first_body_bullet_normalized == extracted_normalized:
            new_body_block = repaired_body_block
        else:
            new_body_block = extracted_bullet_result + "\n" + repaired_body_block
    else:
        new_body_block = extracted_bullet_result + "\n" + repaired_body_block

    second_pass_parts: List[str] = [clean_hook_result, new_body_block] + remaining_after_body
    second_pass_description: str = "\n\n".join(second_pass_parts)

    if has_hook_echo_in_body(second_pass_description):
        return None
    return second_pass_description
