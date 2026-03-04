from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _render_template(template: str, values: Dict[str, Any]) -> str:
    try:
        return template.format(**values)
    except KeyError as error:
        raise RuntimeError(f"Template render failed, missing key: {error}") from error


def _transliterate_cyrillic_to_latin(value: str) -> str:
    mapping: Dict[str, str] = {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "h",
        "ґ": "g",
        "д": "d",
        "е": "e",
        "є": "ie",
        "ж": "zh",
        "з": "z",
        "и": "y",
        "і": "i",
        "ї": "yi",
        "й": "i",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "kh",
        "ц": "ts",
        "ч": "ch",
        "ш": "sh",
        "щ": "shch",
        "ь": "",
        "ю": "iu",
        "я": "ia",
        "ё": "yo",
        "э": "e",
        "ъ": "",
    }
    output: List[str] = []
    for char in value.lower():
        if char in mapping:
            output.append(mapping[char])
            continue
        if ("a" <= char <= "z") or ("0" <= char <= "9"):
            output.append(char)
            continue
        if char.isspace() or char in {"-", "_"}:
            output.append("_")
            continue
        output.append("_")
    return "".join(output)


def build_safe_entity_name(video_title: str) -> str:
    transliterated: str = _transliterate_cyrillic_to_latin(video_title)
    normalized_title: str = re.sub(r"\s+", "_", transliterated.strip().lower())
    clean_title: str = re.sub(r"[^a-z0-9_]+", "_", normalized_title)
    clean_title = re.sub(r"_+", "_", clean_title).strip("_")
    if not clean_title:
        clean_title = "video"
    return clean_title[:120]


def _display_language_code(language: str) -> str:
    if language == "uk":
        return "ua"
    return language


def build_drive_preview_path_segments(
    *,
    template: str,
    language: str,
    date_key: str,
) -> List[str]:
    raw_template: str = str(template or "").strip()
    if not raw_template:
        return []
    rendered_path: str = raw_template.format(
        streamertg="streamertg",
        preview="preview",
        language=_display_language_code(language),
        date=date_key,
    )
    return [
        segment.strip()
        for segment in re.split(r"[\\/]+", rendered_path)
        if segment.strip()
    ]


class NamePathBuilder:
    def __init__(
        self,
        local_image_dir_template: str,
        local_doc_dir_template: Optional[str],
        preview_name_template: str,
        doc_title_template: str,
        language_codes_json: str,
        max_filename_stem: int,
    ) -> None:
        self._local_image_dir_template: str = local_image_dir_template
        self._local_doc_dir_template: Optional[str] = (
            str(local_doc_dir_template or "").strip() or None
        )
        self._preview_name_template: str = preview_name_template
        self._doc_title_template: str = doc_title_template
        self._max_filename_stem: int = max(16, int(max_filename_stem))
        self._language_codes: Dict[str, str] = {}
        try:
            payload: Any = json.loads(language_codes_json)
            if isinstance(payload, dict):
                for key, value in payload.items():
                    self._language_codes[str(key)] = str(value).upper()
        except Exception as error:
            raise RuntimeError(
                f"Invalid template files.language_codes: {error}"
            ) from error

    def build_doc_title(
        self,
        date_key: str,
        created_at: datetime,
        processing_mode: str,
    ) -> str:
        creation_stamp: str = created_at.strftime("%H%M_%d%m%y")
        return _render_template(
            self._doc_title_template,
            {
                "date": date_key,
                "creation_stamp": creation_stamp,
                "processing_mode": processing_mode,
            },
        )

    def build_docx_path(self, date_key: str, doc_title: str) -> Optional[Path]:
        if self._local_doc_dir_template is None:
            return None
        base_dir: str = self._local_doc_dir_template.format(date=date_key)
        safe_stem: str = re.sub(r"[^A-Za-z0-9._-]+", "_", str(doc_title or "").strip())
        safe_stem = re.sub(r"_+", "_", safe_stem).strip("._-")
        safe_stem = safe_stem[: self._max_filename_stem].rstrip("._-") or "document"
        return Path(base_dir) / f"{safe_stem}.docx"

    def build_image_path(
        self,
        language: str,
        date_key: str,
        language_position: int,
        title: str,
        extension: str,
    ) -> Path:
        display_language: str = _display_language_code(language)
        base_dir: str = self._local_image_dir_template.format(
            language=display_language,
            date=date_key,
        )
        file_language_code: str = self._language_codes.get(language, "OT")
        safe_title: str = build_safe_entity_name(title)
        raw_stem: str = _render_template(
            self._preview_name_template,
            {
                "index": language_position,
                "language": file_language_code,
                "title": safe_title,
                "date": date_key,
                "display_language": display_language.upper(),
            },
        )
        safe_stem: str = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_stem)
        safe_stem = re.sub(r"_+", "_", safe_stem).strip("._-")
        safe_stem = safe_stem[: self._max_filename_stem].rstrip("._-") or "video"
        return Path(base_dir) / f"{safe_stem}{extension}"
