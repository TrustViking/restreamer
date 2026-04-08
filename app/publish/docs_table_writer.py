from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING, Tuple

from app.bootstrap.logging_config import get_logger
from app.config.settings import AppTemplates
from app.core.text_utils import utf16_len
from app.core.models import (
    LanguageMergeAttempt,
    MergedLanguageContent,
    PlannedVideo,
    VideoMetadata,
)
from app.google import GoogleDocsClient
from app.publish.shared_helpers import no_description_text as _no_description_text

from .docs_request_builder import DocsRequestBuilder
from .docs_table_helpers import (
    TableSnapshot,
    _build_last_table_snapshot,
    _cell_index,
    _find_last_table_cell_paragraph_start_indices,
    _find_last_table_start_index,
    _is_google_docs_bounds_error,
)


if TYPE_CHECKING:
    from .docs_preview_inserter import DocsPreviewInserter


LOGGER = get_logger(__name__)


class LanguageTableRowsBuilder(Protocol):
    def __call__(
        self,
        *,
        language: str,
        videos: List[PlannedVideo],
        merged_content: Optional[MergedLanguageContent],
        merge_attempt: Optional[LanguageMergeAttempt],
        time_display: Optional[str],
        templates: Optional[AppTemplates],
        artifact_status: str,
    ) -> List[Tuple[str, bool]]:
        ...

class DocsTableWriter:
    TITLE_TEXT_ROW_INDEX: int = 2
    DESCRIPTION_TEXT_ROW_INDEX: int = 4

    VIDEO_TABLE_ROWS: int = 4
    VIDEO_TABLE_COLUMNS: int = 1

    VISUAL_STYLE_ROW_TO_RGB: Dict[int, Tuple[float, float, float]] = {
        0: (0.78, 0.84, 0.94),
        1: (0.85, 0.91, 0.83),
        3: (0.81, 0.86, 0.78),
        5: (0.87, 0.83, 0.76),
    }

    def __init__(
        self,
        docs_client: GoogleDocsClient,
        request_builder: DocsRequestBuilder,
        preview_inserter: DocsPreviewInserter,
    ) -> None:
        self._docs_client: GoogleDocsClient = docs_client
        self._request_builder: DocsRequestBuilder = request_builder
        self._preview_inserter: DocsPreviewInserter = preview_inserter

    def write_language_table(
        self,
        document_id: str,
        language: str,
        videos: List[PlannedVideo],
        merged_content: Optional[MergedLanguageContent] = None,
        merge_attempt: Optional[LanguageMergeAttempt] = None,
        time_display: Optional[str] = None,
        table_insert_index: Optional[int] = None,
        artifact_status: str = "none",
        *,
        templates: AppTemplates,
        build_language_table_rows: LanguageTableRowsBuilder,
    ) -> None:
        row_values: List[Tuple[str, bool]] = build_language_table_rows(
            language=language,
            videos=videos,
            merged_content=merged_content,
            merge_attempt=merge_attempt,
            time_display=time_display,
            templates=templates,
            artifact_status=artifact_status,
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

        table_snapshot: TableSnapshot = self._get_last_table_snapshot(
            document_id=document_id,
            rows=rows,
            columns=columns,
        )
        self._fill_language_table_rows(
            document_id=document_id,
            language=language,
            rows=rows,
            columns=columns,
            row_values=row_values,
            table_snapshot=table_snapshot,
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
        self._preview_inserter.insert_previews(
            document_id=document_id,
            language=language,
            videos=videos,
            rows=rows,
            columns=columns,
        )

    def write_video_table(
        self,
        document_id: str,
        video: VideoMetadata,
        docs_thumbnail_url: str,
        fallback_external_thumbnail_url: str,
        *,
        templates: AppTemplates,
    ) -> None:
        rows: int = self.VIDEO_TABLE_ROWS
        columns: int = self.VIDEO_TABLE_COLUMNS

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

        def _effective_indices(use_plus_one: bool) -> List[int]:
            return [
                cell_start_indices[_cell_index(row=row_idx, col=0, columns=columns)]
                + (1 if use_plus_one else 0)
                for row_idx in range(rows)
            ]

        try:
            effective_cell_indices: List[int] = _effective_indices(use_plus_one=False)
            LOGGER.debug(
                "Google table effective insert indices (row0..row%d), plus_one=%s: %s",
                rows - 1,
                False,
                effective_cell_indices,
            )
            description_text: str = video.description.strip() or _no_description_text(
                templates
            )
            requests_payload: List[Dict[str, Any]] = (
                self._request_builder.build_video_text_requests_payload(
                    rows=rows,
                    columns=columns,
                    cell_start_indices=cell_start_indices,
                    video=video,
                    description_text=description_text,
                    use_plus_one=False,
                )
            )
            LOGGER.debug(
                "Google Docs text batchUpdate requests order: url(row2), description(row1), title(row0)"
            )
            self._docs_client.batch_update(
                document_id=document_id,
                requests_payload=requests_payload,
            )
            LOGGER.info("Google Docs table text fill succeeded with plus_one=False.")
        except Exception as base_error:
            LOGGER.warning(
                "Google Docs table text fill failed with plus_one=False. Retrying with plus_one=True. Error: %s",
                base_error,
            )
            effective_cell_indices = _effective_indices(use_plus_one=True)
            LOGGER.debug(
                "Google table effective insert indices (row0..row%d), plus_one=%s: %s",
                rows - 1,
                True,
                effective_cell_indices,
            )
            description_text = video.description.strip() or _no_description_text(templates)
            requests_payload = self._request_builder.build_video_text_requests_payload(
                rows=rows,
                columns=columns,
                cell_start_indices=cell_start_indices,
                video=video,
                description_text=description_text,
                use_plus_one=True,
            )
            LOGGER.debug(
                "Google Docs text batchUpdate requests order: url(row2), description(row1), title(row0)"
            )
            self._docs_client.batch_update(
                document_id=document_id,
                requests_payload=requests_payload,
            )
            LOGGER.info(
                "Google Docs table text fill succeeded with plus_one=True (fallback)."
            )

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

        try:
            self._docs_client.batch_update(
                document_id=document_id,
                requests_payload=[
                    self._request_builder.build_video_image_request(
                        row_index=DocsRequestBuilder.VIDEO_TABLE_IMAGE_ROW_INDEX,
                        col_index=0,
                        columns=columns,
                        cell_start_indices=cell_start_indices,
                        use_plus_one=False,
                        image_uri=docs_thumbnail_url,
                    )
                ],
            )
            LOGGER.info("Google Docs image insert succeeded with plus_one=False.")
            return
        except Exception as image_error:
            LOGGER.warning(
                "Google Docs image insert failed with plus_one=False. Error: %s",
                image_error,
            )
            if _is_google_docs_bounds_error(image_error):
                try:
                    self._docs_client.batch_update(
                        document_id=document_id,
                        requests_payload=[
                            self._request_builder.build_video_image_request(
                                row_index=DocsRequestBuilder.VIDEO_TABLE_IMAGE_ROW_INDEX,
                                col_index=0,
                                columns=columns,
                                cell_start_indices=cell_start_indices,
                                use_plus_one=True,
                                image_uri=docs_thumbnail_url,
                            )
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
                requests_payload=[
                    self._request_builder.build_video_image_link_text_request(
                        row_index=DocsRequestBuilder.VIDEO_TABLE_IMAGE_ROW_INDEX,
                        col_index=0,
                        columns=columns,
                        cell_start_indices=cell_start_indices,
                        use_plus_one=False,
                        fallback_external_thumbnail_url=fallback_external_thumbnail_url,
                    )
                ],
            )
        except Exception:
            self._docs_client.batch_update(
                document_id=document_id,
                requests_payload=[
                    self._request_builder.build_video_image_link_text_request(
                        row_index=DocsRequestBuilder.VIDEO_TABLE_IMAGE_ROW_INDEX,
                        col_index=0,
                        columns=columns,
                        cell_start_indices=cell_start_indices,
                        use_plus_one=True,
                        fallback_external_thumbnail_url=fallback_external_thumbnail_url,
                    )
                ],
            )
        LOGGER.info("Google Docs image row filled with external thumbnail link.")

    def _get_last_table_snapshot(
        self,
        *,
        document_id: str,
        rows: int,
        columns: int,
    ) -> TableSnapshot:
        doc: Dict[str, Any] = self._docs_client.get_document(document_id=document_id)
        return _build_last_table_snapshot(doc=doc, rows=rows, columns=columns)

    def _fill_language_table_rows(
        self,
        *,
        document_id: str,
        language: str,
        rows: int,
        columns: int,
        row_values: List[Tuple[str, bool]],
        table_snapshot: TableSnapshot,
    ) -> None:
        rows_filled: int = 0
        active_snapshot: TableSnapshot = table_snapshot
        for row_index in range(rows - 1, -1, -1):
            text, is_bold = row_values[row_index]
            active_snapshot = self._insert_language_table_row_text(
                document_id=document_id,
                language=language,
                row_index=row_index,
                rows=rows,
                columns=columns,
                text=text,
                is_bold=is_bold,
                rows_filled_before=rows_filled,
                table_snapshot=active_snapshot,
            )
            rows_filled += 1

    def _insert_language_table_row_text(
        self,
        *,
        document_id: str,
        language: str,
        row_index: int,
        rows: int,
        columns: int,
        text: str,
        is_bold: bool,
        rows_filled_before: int,
        table_snapshot: TableSnapshot,
    ) -> TableSnapshot:
        text_to_insert: str = f"{text}\n"
        attempts: Tuple[Tuple[str, bool, bool, bool], ...] = (
            ("initial", False, False, False),
            ("retry_recomputed", False, True, True),
            ("retry_recomputed_offset", True, True, True),
        )
        current_snapshot: TableSnapshot = table_snapshot
        last_error: Optional[Exception] = None

        for attempt_label, use_plus_one, refetch_doc, recompute_indices in attempts:
            if refetch_doc:
                current_snapshot = self._get_last_table_snapshot(
                    document_id=document_id,
                    rows=rows,
                    columns=columns,
                )
            cell_idx: int = _cell_index(row=row_index, col=0, columns=columns)
            start_index: int = current_snapshot.cell_start_indices[cell_idx]
            effective_index: int = start_index + (1 if use_plus_one else 0)
            try:
                LOGGER.debug(
                    "language_table_fill doc_id=%s lang=%s table_index=%d row_index=%d column_index=%d attempt=%s insert_index=%d paragraph_start_index=%d refetch_doc=%s recompute_cell_indices=%s offset_workaround=%s rows_filled_before=%d",
                    document_id,
                    language,
                    current_snapshot.table_number,
                    row_index,
                    0,
                    attempt_label,
                    effective_index,
                    start_index,
                    "yes" if refetch_doc else "no",
                    "yes" if recompute_indices else "no",
                    "yes" if use_plus_one else "no",
                    rows_filled_before,
                )
                self._docs_client.batch_update(
                    document_id=document_id,
                    requests_payload=self._request_builder.build_insert_and_style_requests(
                        index=effective_index,
                        text=text_to_insert,
                        bold=is_bold,
                    ),
                )
                LOGGER.debug(
                    "language_table_fill_succeeded doc_id=%s lang=%s table_index=%d row_index=%d column_index=%d attempt=%s insert_index=%d rows_filled_after=%d",
                    document_id,
                    language,
                    current_snapshot.table_number,
                    row_index,
                    0,
                    attempt_label,
                    effective_index,
                    rows_filled_before + 1,
                )
                return self._get_last_table_snapshot(
                    document_id=document_id,
                    rows=rows,
                    columns=columns,
                )
            except Exception as error:
                last_error = error
                bounds_error: bool = _is_google_docs_bounds_error(error)
                LOGGER.warning(
                    "language_table_fill_failed doc_id=%s lang=%s table_index=%d row_index=%d column_index=%d attempt=%s insert_index=%d paragraph_start_index=%d refetch_doc=%s recompute_cell_indices=%s offset_workaround=%s rows_filled_before=%d bounds_error=%s error=%s",
                    document_id,
                    language,
                    current_snapshot.table_number,
                    row_index,
                    0,
                    attempt_label,
                    effective_index,
                    start_index,
                    "yes" if refetch_doc else "no",
                    "yes" if recompute_indices else "no",
                    "yes" if use_plus_one else "no",
                    rows_filled_before,
                    "yes" if bounds_error else "no",
                    error,
                )
                if not bounds_error:
                    raise

        raise RuntimeError(
            "Google Docs language table fill failed after recompute retry. "
            f"document_id={document_id} language={language} row_index={row_index} "
            f"rows_filled_before={rows_filled_before} reason={last_error}"
        ) from last_error

    def _apply_language_table_visual_style(
        self,
        document_id: str,
        rows: int,
        columns: int,
        row_values: List[Tuple[str, bool]],
    ) -> None:
        doc: Dict[str, Any] = self._docs_client.get_document(document_id=document_id)
        table_start_index: int = _find_last_table_start_index(doc=doc)
        cell_start_indices: List[int] = _find_last_table_cell_paragraph_start_indices(
            doc=doc,
            rows=rows,
            columns=columns,
        )

        requests_payload: List[Dict[str, Any]] = []
        for row_index, rgb in self.VISUAL_STYLE_ROW_TO_RGB.items():
            if row_index >= rows:
                continue

            requests_payload.append(
                self._request_builder.build_cell_background_request(
                    table_start_index=table_start_index,
                    row=row_index,
                    col=0,
                    rgb=rgb,
                )
            )

            cell_idx: int = _cell_index(row=row_index, col=0, columns=columns)
            paragraph_start: int = cell_start_indices[cell_idx] + 1
            paragraph_end: int = paragraph_start + utf16_len(row_values[row_index][0]) + 1
            requests_payload.append(
                self._request_builder.build_paragraph_style_request(
                    start=paragraph_start,
                    end=paragraph_end,
                    alignment="CENTER",
                )
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
        title_text_row: int = self.TITLE_TEXT_ROW_INDEX
        description_text_row: int = self.DESCRIPTION_TEXT_ROW_INDEX
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
            start_index: int = cell_start_indices[cell_idx] + 1
            end_index: int = start_index + utf16_len(row_values[row_index][0]) + 1
            return start_index, end_index

        title_start, title_end = _row_range(title_text_row)
        desc_start, desc_end = _row_range(description_text_row)

        requests_payload: List[Dict[str, Any]] = [
            self._request_builder.build_paragraph_style_request(
                start=title_start,
                end=title_end,
                alignment="JUSTIFIED",
            ),
            self._request_builder.build_paragraph_style_request(
                start=desc_start,
                end=desc_end,
                alignment="START",
            ),
        ]
        self._docs_client.batch_update(
            document_id=document_id,
            requests_payload=requests_payload,
        )
