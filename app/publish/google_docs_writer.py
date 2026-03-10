from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.config.settings import AppTemplates
from app.core.models import (
    LanguageMergeAttempt,
    MergedLanguageContent,
    PlannedVideo,
    VideoMetadata,
)
from app.google import GoogleDocsClient
from app.publish.doc_helpers import (
    _build_language_table_rows,
    _no_description_text,
    _thumbnail_candidates,
)


LOGGER = _get_logger_impl(__name__)


@dataclass(frozen=True)
class PreviewInsertPlan:
    row_index: int
    row_number: int
    language: str
    paragraph_insert_index: int
    plus_one_used: bool
    effective_offset: int
    link_index: int
    image_index: int


class GoogleDocsReportWriter:
    """
    Создает таблицу в документе и заполняет ее.
    Вставляет изображение в ячейку preview через insertInlineImage
    (нужен публичный URL изображения).
    """

    def __init__(self, docs_client: GoogleDocsClient, templates: AppTemplates) -> None:
        self._docs_client: GoogleDocsClient = docs_client
        self._templates: AppTemplates = templates

    def write_daily_document(
        self,
        document_id: str,
        header_text: str,
        language_groups: Dict[str, List[PlannedVideo]],
        merged_content_by_language: Optional[Dict[str, MergedLanguageContent]] = None,
        merge_audit_by_language: Optional[Dict[str, LanguageMergeAttempt]] = None,
    ) -> None:
        self._insert_header_text(
            document_id=document_id,
            text=f"{header_text}\n\n",
        )
        guard_index: int = (
            self._docs_client.get_document_end_index(document_id=document_id) - 1
        )
        self._docs_client.insert_text_at_index(
            document_id=document_id,
            index=guard_index,
            text="\n",
        )
        LOGGER.info(
            "Inserted blank paragraph before first table (doc formatting guard) index=%d.",
            guard_index,
        )

        non_empty_languages: List[str] = [
            language
            for language in ("uk", "en", "ru", "other")
            if language_groups.get(language)
        ]

        for index, language in enumerate(non_empty_languages):
            self.write_language_table(
                document_id=document_id,
                language=language,
                videos=language_groups[language],
                merged_content=(
                    (merged_content_by_language or {}).get(language)
                    if merged_content_by_language
                    else None
                ),
                merge_attempt=(
                    (merge_audit_by_language or {}).get(language)
                    if merge_audit_by_language
                    else None
                ),
            )
            if index < len(non_empty_languages) - 1:
                self._docs_client.batch_update(
                    document_id=document_id,
                    requests_payload=[
                        {"insertPageBreak": {"endOfSegmentLocation": {}}}
                    ],
                )

    def _insert_styled_text(self, document_id: str, text: str, bold: bool) -> None:
        doc: Dict[str, Any] = self._docs_client.get_document(document_id=document_id)
        content: List[Dict[str, Any]] = cast(
            List[Dict[str, Any]], doc["body"]["content"]
        )
        start_index: int = int(content[-1]["endIndex"]) - 1
        self._docs_client.batch_update(
            document_id=document_id,
            requests_payload=[
                {"insertText": {"location": {"index": start_index}, "text": text}},
                {
                    "updateTextStyle": {
                        "range": {
                            "startIndex": start_index,
                            "endIndex": start_index + len(text),
                        },
                        "textStyle": {
                            "weightedFontFamily": {"fontFamily": "Arial"},
                            "fontSize": {"magnitude": 13, "unit": "PT"},
                            "bold": bold,
                        },
                        "fields": "weightedFontFamily,fontSize,bold",
                    }
                },
            ],
        )

    def _insert_header_text(self, document_id: str, text: str) -> None:
        doc: Dict[str, Any] = self._docs_client.get_document(document_id=document_id)
        content: List[Dict[str, Any]] = cast(
            List[Dict[str, Any]], doc["body"]["content"]
        )
        start_index: int = int(content[-1]["endIndex"]) - 1

        requests_payload: List[Dict[str, Any]] = [
            {"insertText": {"location": {"index": start_index}, "text": text}},
            {
                "updateTextStyle": {
                    "range": {
                        "startIndex": start_index,
                        "endIndex": start_index + len(text),
                    },
                    "textStyle": {
                        "weightedFontFamily": {"fontFamily": "Arial"},
                        "fontSize": {"magnitude": 13, "unit": "PT"},
                        "bold": False,
                    },
                    "fields": "weightedFontFamily,fontSize,bold",
                }
            },
        ]

        bold_line_prefixes: Tuple[str, ...] = ()
        try:
            raw_prefixes: Any = json.loads(
                self._templates.google_doc_bold_line_prefixes_json
            )
            if isinstance(raw_prefixes, list):
                bold_line_prefixes = tuple(str(item) for item in raw_prefixes)
        except Exception:
            bold_line_prefixes = ()
        cursor: int = 0
        for line in text.splitlines(keepends=True):
            line_start: int = start_index + cursor
            line_end: int = line_start + len(line)
            if any(line.startswith(prefix) for prefix in bold_line_prefixes):
                requests_payload.append(
                    {
                        "updateTextStyle": {
                            "range": {
                                "startIndex": line_start,
                                "endIndex": line_end,
                            },
                            "textStyle": {"bold": True},
                            "fields": "bold",
                        }
                    }
                )
            cursor += len(line)

        self._docs_client.batch_update(
            document_id=document_id,
            requests_payload=requests_payload,
        )

    def write_header_only(
        self,
        document_id: str,
        header_text: str,
    ) -> None:
        self._insert_header_text(document_id=document_id, text=header_text)

    def insert_page_break(self, document_id: str) -> None:
        self._docs_client.batch_update(
            document_id=document_id,
            requests_payload=[{"insertPageBreak": {"endOfSegmentLocation": {}}}],
        )

    def write_language_table(
        self,
        document_id: str,
        language: str,
        videos: List[PlannedVideo],
        merged_content: Optional[MergedLanguageContent] = None,
        merge_attempt: Optional[LanguageMergeAttempt] = None,
        time_display: Optional[str] = None,
        table_insert_index: Optional[int] = None,
    ) -> None:
        row_values: List[Tuple[str, bool]] = _build_language_table_rows(
            language=language,
            videos=videos,
            merged_content=merged_content,
            merge_attempt=merge_attempt,
            time_display=time_display,
            templates=self._templates,
        )
        rows: int = len(row_values)
        columns: int = 1
        if table_insert_index is None:
            self._docs_client.insert_table_at_end(
                document_id=document_id,
                rows=rows,
                columns=columns,
            )
        else:
            self._docs_client.insert_table_at_index(
                document_id=document_id,
                rows=rows,
                columns=columns,
                index=table_insert_index,
            )
        doc: Dict[str, Any] = self._docs_client.get_document(document_id=document_id)
        cell_start_indices: List[int] = _find_last_table_cell_paragraph_start_indices(
            doc=doc,
            rows=rows,
            columns=columns,
        )

        def _cell_start_index(row_index: int, use_plus_one: bool) -> int:
            idx: int = _cell_index(row=row_index, col=0, columns=columns)
            return cell_start_indices[idx] + (1 if use_plus_one else 0)

        def _build_requests(use_plus_one: bool) -> List[Dict[str, Any]]:
            requests_payload: List[Dict[str, Any]] = []
            for row_index in range(rows - 1, -1, -1):
                text, is_bold = row_values[row_index]
                start_index: int = _cell_start_index(
                    row_index=row_index,
                    use_plus_one=use_plus_one,
                )
                text_to_insert: str = f"{text}\n"
                requests_payload.append(
                    {
                        "insertText": {
                            "location": {"index": start_index},
                            "text": text_to_insert,
                        }
                    }
                )
                requests_payload.append(
                    {
                        "updateTextStyle": {
                            "range": {
                                "startIndex": start_index,
                                "endIndex": start_index + len(text_to_insert),
                            },
                            "textStyle": {
                                "weightedFontFamily": {"fontFamily": "Arial"},
                                "fontSize": {"magnitude": 13, "unit": "PT"},
                                "bold": is_bold,
                            },
                            "fields": "weightedFontFamily,fontSize,bold",
                        }
                    }
                )
            return requests_payload

        try:
            self._docs_client.batch_update(
                document_id=document_id,
                requests_payload=_build_requests(use_plus_one=False),
            )
        except Exception:
            self._docs_client.batch_update(
                document_id=document_id,
                requests_payload=_build_requests(use_plus_one=True),
            )

        self._apply_language_table_visual_style(
            document_id=document_id,
            rows=rows,
            columns=columns,
            row_values=row_values,
        )
        self._apply_language_table_content_style(
            document_id=document_id,
            rows=rows,
            columns=columns,
            row_values=row_values,
        )

        preview_label_row_index: int = 5
        preview_first_item_row_index: int = preview_label_row_index + 1

        def _refresh_row_start_index(row_index: int) -> int:
            updated_doc: Dict[str, Any] = self._docs_client.get_document(
                document_id=document_id
            )
            updated_indices: List[int] = _find_last_table_cell_paragraph_start_indices(
                doc=updated_doc,
                rows=rows,
                columns=columns,
            )
            idx: int = _cell_index(row=row_index, col=0, columns=columns)
            return updated_indices[idx] + 1

        def _build_preview_insert_plan(
            *,
            row_index: int,
            row_number: int,
            language: str,
            link_text: str,
            plus_one_used: bool,
        ) -> PreviewInsertPlan:
            paragraph_insert_index: int = _refresh_row_start_index(row_index)
            effective_offset: int = 1 if plus_one_used else 0
            link_index: int = paragraph_insert_index + effective_offset
            image_index: int = link_index + len(link_text)
            return PreviewInsertPlan(
                row_index=row_index,
                row_number=row_number,
                language=language,
                paragraph_insert_index=paragraph_insert_index,
                plus_one_used=plus_one_used,
                effective_offset=effective_offset,
                link_index=link_index,
                image_index=image_index,
            )

        indexed_videos: List[Tuple[int, PlannedVideo]] = list(
            enumerate(videos, start=1)
        )

        def _preview_link_text(video: PlannedVideo) -> str:
            candidate_link: str = (
                str(video.normalized_link or "").strip()
                or str(video.original_link or "").strip()
                or str(video.metadata.url or "").strip()
            )
            return f"{candidate_link}\n" if candidate_link else ""

        for preview_index, video in reversed(indexed_videos):
            row_index: int = preview_first_item_row_index + (preview_index - 1)
            row_start_index: int = _refresh_row_start_index(row_index)
            link_text: str = _preview_link_text(video)
            if not link_text:
                LOGGER.warning(
                    "Preview link insert skipped for row %d in language %s: empty link payload.",
                    video.row_number,
                    language,
                )
                continue
            successful_preview_plan: Optional[PreviewInsertPlan] = None
            link_insert_error: str = ""
            for use_plus_one in (False, True):
                try:
                    preview_insert_plan: PreviewInsertPlan = _build_preview_insert_plan(
                        row_index=row_index,
                        row_number=video.row_number,
                        language=language,
                        link_text=link_text,
                        plus_one_used=use_plus_one,
                    )
                    self._docs_client.batch_update(
                        document_id=document_id,
                        requests_payload=[
                            {
                                "insertText": {
                                    "location": {"index": preview_insert_plan.link_index},
                                    "text": link_text,
                                }
                            }
                        ],
                    )
                    successful_preview_plan = preview_insert_plan
                    LOGGER.info(
                        "preview_insert row=%d lang=%s stage=link link_plus_one_used=%s effective_offset=%d paragraph_insert_index=%d link_index=%d",
                        preview_insert_plan.row_number,
                        preview_insert_plan.language,
                        "yes" if preview_insert_plan.plus_one_used else "no",
                        preview_insert_plan.effective_offset,
                        preview_insert_plan.paragraph_insert_index,
                        preview_insert_plan.link_index,
                    )
                    if use_plus_one:
                        LOGGER.info(
                            "Preview link insert recovered with plus_one=True for row %d in language %s.",
                            video.row_number,
                            language,
                        )
                    break
                except Exception as error:
                    link_insert_error = str(error)
            if successful_preview_plan is None:
                LOGGER.warning(
                    "Preview link insert failed for row %d in language %s. row_index=%d start_index=%d link_length=%d reason=%s",
                    video.row_number,
                    language,
                    row_index,
                    row_start_index,
                    len(link_text),
                    link_insert_error or "unknown",
                )
                continue

            inserted: bool = False
            for image_uri in _thumbnail_candidates(video):
                try:
                    self._docs_client.batch_update(
                        document_id=document_id,
                        requests_payload=[
                            {
                                "insertInlineImage": {
                                    "location": {"index": successful_preview_plan.image_index},
                                    "uri": image_uri,
                                    "objectSize": {
                                        "height": {"magnitude": 120, "unit": "PT"},
                                        "width": {"magnitude": 210, "unit": "PT"},
                                    },
                                }
                            },
                        ],
                    )
                    LOGGER.info(
                        "preview_insert row=%d lang=%s stage=image link_plus_one_used=%s effective_offset=%d image_index=%d link_and_image_mode=aligned",
                        successful_preview_plan.row_number,
                        successful_preview_plan.language,
                        "yes" if successful_preview_plan.plus_one_used else "no",
                        successful_preview_plan.effective_offset,
                        successful_preview_plan.image_index,
                    )
                    inserted = True
                    break
                except Exception:
                    continue
            if not inserted:
                LOGGER.warning(
                    "Preview image insert skipped for row %d in language %s.",
                    video.row_number,
                    language,
                )

    def _apply_language_table_visual_style(
        self,
        document_id: str,
        rows: int,
        columns: int,
        row_values: List[Tuple[str, bool]],
    ) -> None:
        # Calm palette for header rows in each language table.
        row_to_rgb: Dict[int, Tuple[float, float, float]] = {
            0: (0.78, 0.84, 0.94),  # language row: muted blue
            1: (0.85, 0.91, 0.83),  # title row: soft green
            3: (0.81, 0.86, 0.78),  # description row: sage
            5: (0.87, 0.83, 0.76),  # preview row: warm sand
        }

        doc: Dict[str, Any] = self._docs_client.get_document(document_id=document_id)
        table_start_index: int = _find_last_table_start_index(doc=doc)
        cell_start_indices: List[int] = _find_last_table_cell_paragraph_start_indices(
            doc=doc,
            rows=rows,
            columns=columns,
        )

        requests_payload: List[Dict[str, Any]] = []
        for row_index, (red, green, blue) in row_to_rgb.items():
            if row_index >= rows:
                continue

            requests_payload.append(
                {
                    "updateTableCellStyle": {
                        "tableRange": {
                            "tableCellLocation": {
                                "tableStartLocation": {"index": table_start_index},
                                "rowIndex": row_index,
                                "columnIndex": 0,
                            },
                            "rowSpan": 1,
                            "columnSpan": 1,
                        },
                        "tableCellStyle": {
                            "backgroundColor": {
                                "color": {
                                    "rgbColor": {
                                        "red": red,
                                        "green": green,
                                        "blue": blue,
                                    }
                                }
                            }
                        },
                        "fields": "backgroundColor",
                    }
                }
            )

            cell_idx: int = _cell_index(row=row_index, col=0, columns=columns)
            paragraph_start: int = cell_start_indices[cell_idx] + 1
            paragraph_end: int = paragraph_start + len(row_values[row_index][0]) + 1
            requests_payload.append(
                {
                    "updateParagraphStyle": {
                        "range": {
                            "startIndex": paragraph_start,
                            "endIndex": paragraph_end,
                        },
                        "paragraphStyle": {"alignment": "CENTER"},
                        "fields": "alignment",
                    }
                }
            )

        if requests_payload:
            self._docs_client.batch_update(
                document_id=document_id,
                requests_payload=requests_payload,
            )

    def _apply_language_table_content_style(
        self,
        document_id: str,
        rows: int,
        columns: int,
        row_values: List[Tuple[str, bool]],
    ) -> None:
        # Content rows: titles and descriptions.
        title_text_row: int = 2
        description_text_row: int = 4
        if rows <= description_text_row:
            return

        doc: Dict[str, Any] = self._docs_client.get_document(document_id=document_id)
        cell_start_indices: List[int] = _find_last_table_cell_paragraph_start_indices(
            doc=doc,
            rows=rows,
            columns=columns,
        )

        def _row_range(row_index: int) -> Tuple[int, int]:
            cell_idx: int = _cell_index(row=row_index, col=0, columns=columns)
            # Text can be inserted either at paragraph boundary or +1 fallback.
            # Start from paragraph boundary to avoid losing the first character style.
            start_index: int = cell_start_indices[cell_idx]
            end_index: int = start_index + len(row_values[row_index][0]) + 1
            return start_index, end_index

        title_start, title_end = _row_range(title_text_row)
        desc_start, desc_end = _row_range(description_text_row)

        requests_payload: List[Dict[str, Any]] = [
            {
                "updateTextStyle": {
                    "range": {"startIndex": title_start, "endIndex": title_end},
                    "textStyle": {"bold": True},
                    "fields": "bold",
                }
            },
            {
                "updateTextStyle": {
                    "range": {"startIndex": desc_start, "endIndex": desc_end},
                    "textStyle": {"bold": False},
                    "fields": "bold",
                }
            },
            {
                "updateParagraphStyle": {
                    "range": {"startIndex": title_start, "endIndex": title_end},
                    "paragraphStyle": {"alignment": "JUSTIFIED"},
                    "fields": "alignment",
                }
            },
            {
                "updateParagraphStyle": {
                    "range": {"startIndex": desc_start, "endIndex": desc_end},
                    "paragraphStyle": {"alignment": "START"},
                    "fields": "alignment",
                }
            },
        ]
        self._docs_client.batch_update(
            document_id=document_id,
            requests_payload=requests_payload,
        )

    def write_video_table(
        self,
        document_id: str,
        video: VideoMetadata,
        docs_thumbnail_url: str,
        fallback_external_thumbnail_url: str,
    ) -> None:
        rows: int = 4
        columns: int = 1

        self._docs_client.insert_table_at_end(
            document_id=document_id,
            rows=rows,
            columns=columns,
        )

        doc: Dict[str, Any] = self._docs_client.get_document(document_id=document_id)

        cell_start_indices: List[int] = _find_last_table_cell_paragraph_start_indices(
            doc=doc,
            rows=rows,
            columns=columns,
        )
        LOGGER.debug(
            "Google table raw paragraph start indices: %s",
            cell_start_indices,
        )

        def _cell_start_index(
            row_index: int, col_index: int, use_plus_one: bool
        ) -> int:
            idx: int = _cell_index(row=row_index, col=col_index, columns=columns)
            if use_plus_one:
                # Вставляем внутрь paragraph ячейки, а не в его структурную границу.
                return cell_start_indices[idx] + 1
            return cell_start_indices[idx]

        def _build_text_requests_payload(use_plus_one: bool) -> List[Dict[str, Any]]:
            effective_cell_indices: List[int] = [
                _cell_start_index(
                    row_index=row_idx,
                    col_index=0,
                    use_plus_one=use_plus_one,
                )
                for row_idx in range(rows)
            ]
            LOGGER.debug(
                "Google table effective insert indices (row0..row%d), plus_one=%s: %s",
                rows - 1,
                use_plus_one,
                effective_cell_indices,
            )

            requests_payload: List[Dict[str, Any]] = []
            description_text: str = video.description.strip() or _no_description_text(
                self._templates
            )

            def _build_format_request(
                start_index: int, end_index: int
            ) -> Dict[str, Any]:
                return {
                    "updateTextStyle": {
                        "range": {
                            "startIndex": start_index,
                            "endIndex": end_index,
                        },
                        "textStyle": {
                            "weightedFontFamily": {"fontFamily": "Arial"},
                            "fontSize": {"magnitude": 13, "unit": "PT"},
                        },
                        "fields": "weightedFontFamily,fontSize",
                    }
                }

            # Важно: requests идут снизу вверх, чтобы сдвиг индексов не ломал следующие вставки.
            url_start_index: int = _cell_start_index(
                row_index=2,
                col_index=0,
                use_plus_one=use_plus_one,
            )
            url_text: str = f"{video.url}\n"
            requests_payload.append(
                {
                    "insertText": {
                        "location": {"index": url_start_index},
                        "text": url_text,
                    }
                }
            )
            requests_payload.append(
                _build_format_request(
                    start_index=url_start_index,
                    end_index=url_start_index + len(url_text),
                )
            )

            description_start_index: int = _cell_start_index(
                row_index=1,
                col_index=0,
                use_plus_one=use_plus_one,
            )
            description_full_text: str = f"{description_text}\n"
            requests_payload.append(
                {
                    "insertText": {
                        "location": {"index": description_start_index},
                        "text": description_full_text,
                    }
                }
            )
            requests_payload.append(
                _build_format_request(
                    start_index=description_start_index,
                    end_index=description_start_index + len(description_full_text),
                )
            )

            title_start_index: int = _cell_start_index(
                row_index=0,
                col_index=0,
                use_plus_one=use_plus_one,
            )
            title_text: str = f"{video.title}\n"
            requests_payload.append(
                {
                    "insertText": {
                        "location": {"index": title_start_index},
                        "text": title_text,
                    }
                }
            )
            requests_payload.append(
                _build_format_request(
                    start_index=title_start_index,
                    end_index=title_start_index + len(title_text),
                )
            )
            LOGGER.debug(
                "Google Docs text batchUpdate requests order: url(row2), description(row1), title(row0)"
            )
            return requests_payload

        try:
            self._docs_client.batch_update(
                document_id=document_id,
                requests_payload=_build_text_requests_payload(use_plus_one=False),
            )
            LOGGER.info("Google Docs table text fill succeeded with plus_one=False.")
        except Exception as base_error:
            LOGGER.warning(
                "Google Docs table text fill failed with plus_one=False. Retrying with plus_one=True. Error: %s",
                base_error,
            )
            self._docs_client.batch_update(
                document_id=document_id,
                requests_payload=_build_text_requests_payload(use_plus_one=True),
            )
            LOGGER.info(
                "Google Docs table text fill succeeded with plus_one=True (fallback)."
            )

        # Важно: после вставки текста структура документа изменилась.
        # Перед вставкой картинки/ссылки пересчитываем индексы ячеек заново.
        doc = self._docs_client.get_document(document_id=document_id)
        cell_start_indices = _find_last_table_cell_paragraph_start_indices(
            doc=doc,
            rows=rows,
            columns=columns,
        )
        LOGGER.debug(
            "Google table refreshed paragraph start indices before image step: %s",
            cell_start_indices,
        )

        def _build_image_request(use_plus_one: bool, image_uri: str) -> Dict[str, Any]:
            image_index: int = _cell_start_index(
                row_index=3, col_index=0, use_plus_one=use_plus_one
            )
            return {
                "insertInlineImage": {
                    "location": {"index": image_index},
                    "uri": image_uri,
                    "objectSize": {
                        "height": {"magnitude": 180, "unit": "PT"},
                        "width": {"magnitude": 320, "unit": "PT"},
                    },
                }
            }

        def _build_image_link_text_request(use_plus_one: bool) -> Dict[str, Any]:
            image_index: int = _cell_start_index(
                row_index=3, col_index=0, use_plus_one=use_plus_one
            )
            return {
                "insertText": {
                    "location": {"index": image_index},
                    "text": f"Thumbnail: {fallback_external_thumbnail_url}\n",
                }
            }

        def _is_bounds_error(error: Exception) -> bool:
            return "inside the bounds of an existing paragraph" in str(error).lower()

        try:
            self._docs_client.batch_update(
                document_id=document_id,
                requests_payload=[_build_image_request(False, docs_thumbnail_url)],
            )
            LOGGER.info("Google Docs image insert succeeded with plus_one=False.")
            return
        except Exception as image_error:
            LOGGER.warning(
                "Google Docs image insert failed with plus_one=False. Error: %s",
                image_error,
            )
            if _is_bounds_error(image_error):
                try:
                    self._docs_client.batch_update(
                        document_id=document_id,
                        requests_payload=[
                            _build_image_request(True, docs_thumbnail_url)
                        ],
                    )
                    LOGGER.info(
                        "Google Docs image insert succeeded with plus_one=True (fallback)."
                    )
                    return
                except Exception as plus_one_error:
                    LOGGER.warning(
                        "Google Docs image insert failed with plus_one=True. Error: %s",
                        plus_one_error,
                    )

        LOGGER.warning(
            "Google Docs image insert failed; inserting external thumbnail link instead."
        )
        doc = self._docs_client.get_document(document_id=document_id)
        cell_start_indices = _find_last_table_cell_paragraph_start_indices(
            doc=doc,
            rows=rows,
            columns=columns,
        )
        LOGGER.debug(
            "Google table refreshed paragraph start indices before image-link fallback: %s",
            cell_start_indices,
        )
        try:
            self._docs_client.batch_update(
                document_id=document_id,
                requests_payload=[_build_image_link_text_request(False)],
            )
        except Exception:
            self._docs_client.batch_update(
                document_id=document_id,
                requests_payload=[_build_image_link_text_request(True)],
            )
        LOGGER.info("Google Docs image row filled with external thumbnail link.")



def _cell_index(row: int, col: int, columns: int) -> int:
    return row * columns + col



def _find_last_table_cell_paragraph_start_indices(
    doc: Dict[str, Any], rows: int, columns: int
) -> List[int]:
    """
    Возвращает startIndex абзаца для каждой ячейки (по строкам) последней таблицы.
    Предполагается, что в каждой ячейке есть хотя бы один абзац.
    """
    body: Dict[str, Any] = doc.get("body", {})
    content: List[Dict[str, Any]] = body.get("content", [])
    tables: List[Dict[str, Any]] = [
        item["table"] for item in content if "table" in item
    ]
    if not tables:
        raise RuntimeError("Не нашел таблицу в документе после insertTable.")
    table: Dict[str, Any] = tables[-1]

    table_rows: List[Dict[str, Any]] = table.get("tableRows", [])
    if len(table_rows) < rows:
        raise RuntimeError("Таблица имеет меньше строк, чем ожидается.")
    if any(len(r.get("tableCells", [])) < columns for r in table_rows[:rows]):
        raise RuntimeError("Таблица имеет меньше колонок, чем ожидается.")

    result: List[int] = []
    for row_index in range(rows):
        cells: List[Dict[str, Any]] = table_rows[row_index]["tableCells"]
        for col_index in range(columns):
            cell_content: List[Dict[str, Any]] = cells[col_index].get("content", [])
            paragraph: Optional[Dict[str, Any]] = None
            for element in cell_content:
                if "paragraph" in element:
                    paragraph = element["paragraph"]
                    break
            if paragraph is None:
                raise RuntimeError(
                    "Не нашел paragraph в ячейке таблицы (ожидался всегда)."
                )
            # Элементы абзаца лежат в cell_content; нужен startIndex
            # структурного контейнера, а не вложенного paragraph.
            # Поэтому ищем startIndex в том же элементе, где есть "paragraph".
            paragraph_container: Optional[Dict[str, Any]] = None
            for element in cell_content:
                if "paragraph" in element:
                    paragraph_container = element
                    break
            if paragraph_container is None or "startIndex" not in paragraph_container:
                raise RuntimeError(
                    "Не удалось определить startIndex paragraph контейнера в ячейке."
                )
            start_index: int = int(paragraph_container["startIndex"])

            # start_index: int = (
            #     int(paragraph_container["startIndex"]) + 1
            # )  # +1, чтобы вставка шла внутрь абзаца.
            result.append(start_index)

    if len(result) != rows * columns:
        raise RuntimeError("Внутренняя ошибка: неверное количество индексов ячеек.")
    return result


def _find_last_table_start_index(doc: Dict[str, Any]) -> int:
    body: Dict[str, Any] = doc.get("body", {})
    content: List[Dict[str, Any]] = body.get("content", [])
    table_items: List[Dict[str, Any]] = [item for item in content if "table" in item]
    if not table_items:
        raise RuntimeError("Не найдена таблица в документе.")
    start_index_raw: Optional[Any] = table_items[-1].get("startIndex")
    if start_index_raw is None:
        raise RuntimeError("Не найден startIndex последней таблицы.")
    return int(start_index_raw)


