from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.bootstrap.logging_config import get_logger
from app.core.text_utils import utf16_len
from app.core.models import PlannedVideo
from app.google import GoogleDocsClient
from app.publish.doc_helpers import _thumbnail_candidates

from .docs_request_builder import DocsRequestBuilder
from .docs_table_helpers import _cell_index, _find_last_table_cell_paragraph_start_indices


LOGGER = get_logger(__name__)


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


class DocsPreviewInserter:
    PREVIEW_LABEL_ROW_INDEX: int = 5
    PREVIEW_FIRST_ITEM_ROW_INDEX: int = PREVIEW_LABEL_ROW_INDEX + 1

    PREVIEW_IMAGE_HEIGHT_PT: float = 120
    PREVIEW_IMAGE_WIDTH_PT: float = 210

    def __init__(
        self,
        docs_client: GoogleDocsClient,
        request_builder: DocsRequestBuilder,
    ) -> None:
        self._docs_client: GoogleDocsClient = docs_client
        self._request_builder: DocsRequestBuilder = request_builder

    def insert_previews(
        self,
        document_id: str,
        language: str,
        videos: List[PlannedVideo],
        rows: int,
        columns: int,
    ) -> None:
        indexed_videos: List[Tuple[int, PlannedVideo]] = list(
            enumerate(videos, start=1)
        )

        for preview_index, video in reversed(indexed_videos):
            row_index: int = self.PREVIEW_FIRST_ITEM_ROW_INDEX + (preview_index - 1)
            row_start_index: int = self._refresh_row_start_index(
                document_id=document_id,
                row_index=row_index,
                rows=rows,
                columns=columns,
            )
            link_text: str = self._preview_link_text(video)
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
                    preview_insert_plan: PreviewInsertPlan = self._build_preview_insert_plan(
                        document_id=document_id,
                        row_index=row_index,
                        row_number=video.row_number,
                        language=language,
                        rows=rows,
                        columns=columns,
                        link_text=link_text,
                        plus_one_used=use_plus_one,
                    )
                    self._docs_client.batch_update(
                        document_id=document_id,
                        requests_payload=[
                            self._request_builder.build_insert_text_request(
                                index=preview_insert_plan.link_index,
                                text=link_text,
                            )
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
                            self._request_builder.build_inline_image_request(
                                index=successful_preview_plan.image_index,
                                uri=image_uri,
                                height_pt=self.PREVIEW_IMAGE_HEIGHT_PT,
                                width_pt=self.PREVIEW_IMAGE_WIDTH_PT,
                            )
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

    def _refresh_row_start_index(
        self,
        *,
        document_id: str,
        row_index: int,
        rows: int,
        columns: int,
    ) -> int:
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
        self,
        *,
        document_id: str,
        row_index: int,
        row_number: int,
        language: str,
        rows: int,
        columns: int,
        link_text: str,
        plus_one_used: bool,
    ) -> PreviewInsertPlan:
        paragraph_insert_index: int = self._refresh_row_start_index(
            document_id=document_id,
            row_index=row_index,
            rows=rows,
            columns=columns,
        )
        effective_offset: int = 1 if plus_one_used else 0
        link_index: int = paragraph_insert_index + effective_offset
        image_index: int = link_index + utf16_len(link_text)
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

    @staticmethod
    def _preview_link_text(video: PlannedVideo) -> str:
        candidate_link: str = (
            str(video.normalized_link or "").strip()
            or str(video.original_link or "").strip()
            or str(video.metadata.url or "").strip()
        )
        return f"{candidate_link}\n" if candidate_link else ""
