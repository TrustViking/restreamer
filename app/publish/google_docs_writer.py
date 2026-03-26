from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, cast

from app.bootstrap.logging_config import get_logger
from app.config.settings import AppTemplates
from app.core.models import (
    LanguageMergeAttempt,
    MergedLanguageContent,
    PlannedVideo,
    VideoMetadata,
)
from app.google import GoogleDocsClient
from app.publish.doc_helpers import _build_language_table_rows

from .docs_preview_inserter import DocsPreviewInserter
from .docs_request_builder import DocsRequestBuilder
from .docs_table_writer import DocsTableWriter


LOGGER = get_logger(__name__)


class GoogleDocsReportWriter:
    def __init__(self, docs_client: GoogleDocsClient, templates: AppTemplates) -> None:
        self._docs_client: GoogleDocsClient = docs_client
        self._templates: AppTemplates = templates

        request_builder = DocsRequestBuilder()
        preview_inserter = DocsPreviewInserter(
            docs_client=docs_client,
            request_builder=request_builder,
        )
        self._table_writer = DocsTableWriter(
            docs_client=docs_client,
            request_builder=request_builder,
            preview_inserter=preview_inserter,
        )
        self._request_builder = request_builder

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
                    requests_payload=[self._request_builder.build_page_break_request()],
                )

    def _insert_styled_text(self, document_id: str, text: str, bold: bool) -> None:
        doc: Dict[str, Any] = self._docs_client.get_document(document_id=document_id)
        content: List[Dict[str, Any]] = cast(
            List[Dict[str, Any]], doc["body"]["content"]
        )
        start_index: int = int(content[-1]["endIndex"]) - 1
        self._docs_client.batch_update(
            document_id=document_id,
            requests_payload=self._request_builder.build_insert_and_style_requests(
                index=start_index,
                text=text,
                bold=bold,
            ),
        )

    def _insert_header_text(self, document_id: str, text: str) -> None:
        doc: Dict[str, Any] = self._docs_client.get_document(document_id=document_id)
        content: List[Dict[str, Any]] = cast(
            List[Dict[str, Any]], doc["body"]["content"]
        )
        start_index: int = int(content[-1]["endIndex"]) - 1

        requests_payload: List[Dict[str, Any]] = self._request_builder.build_insert_and_style_requests(
            index=start_index,
            text=text,
            bold=False,
        )

        bold_line_prefixes: Tuple[str, ...] = ()
        raw_prefixes: object = getattr(
            self._templates,
            "google_doc_bold_line_prefixes",
            [],
        )
        if isinstance(raw_prefixes, list):
            bold_line_prefixes = tuple(str(item) for item in raw_prefixes)

        cursor: int = 0
        for line in text.splitlines(keepends=True):
            line_start: int = start_index + cursor
            line_end: int = line_start + len(line)
            if any(line.startswith(prefix) for prefix in bold_line_prefixes):
                requests_payload.append(
                    self._request_builder.build_text_style_request(
                        start=line_start,
                        end=line_end,
                        bold=True,
                    )
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
            requests_payload=[self._request_builder.build_page_break_request()],
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
        artifact_status: str = "none",
    ) -> None:
        self._table_writer.write_language_table(
            document_id=document_id,
            language=language,
            videos=videos,
            merged_content=merged_content,
            merge_attempt=merge_attempt,
            time_display=time_display,
            table_insert_index=table_insert_index,
            artifact_status=artifact_status,
            templates=self._templates,
            build_language_table_rows=_build_language_table_rows,
        )

    def write_video_table(
        self,
        document_id: str,
        video: VideoMetadata,
        docs_thumbnail_url: str,
        fallback_external_thumbnail_url: str,
    ) -> None:
        self._table_writer.write_video_table(
            document_id=document_id,
            video=video,
            docs_thumbnail_url=docs_thumbnail_url,
            fallback_external_thumbnail_url=fallback_external_thumbnail_url,
            templates=self._templates,
        )
