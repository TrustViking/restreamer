from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from app.config.settings import AppConfig, AppTemplates
from app.core.branching import BRANCH_MERGE
from app.core.error_summary import summarize_error
from app.core.models import (
    BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE,
    LanguageMergeAttempt,
    MergedLanguageContent,
    PlannedVideo,
)
from app.google import GoogleDocsClient, GoogleDriveClient
from app.llm.models.model_identity import resolve_effective_llm_model
from app.observability.content_contract import (
    analyze_content_contract,
    build_contract_transition,
    build_pre_sanitation_text,
    infer_tail_source,
    infer_title_mode,
    log_content_contract,
    log_publish_block_summary,
    log_tail_analysis,
)
from app.observability.runtime_analytics import record_contract_result
from app.paths.name_builder import NamePathBuilder
from app.paths.output_naming import (
    collect_used_runtime_models,
    render_doc_title_models_segment,
)
from app.planning import format_time_key_for_display
from app.publish.google_docs_writer import GoogleDocsReportWriter
from app.publish.doc_helpers import _build_descriptions_summary, _build_titles_summary
from app.publish.post_llm_sanitation import (
    resolve_block_generation_mode,
    sanitize_post_llm_title,
)

from .slot_processing import SlotProcessResult, resolve_merge_artifact_status


@dataclass(frozen=True)
class DailyDocumentPublishResult:
    doc_title: str
    doc_url: str
    header_context: Dict[str, str]
    slot_keys: List[str]
    local_docx_exported: bool = False
    local_export_path: Optional[Path] = None
    google_doc_created: bool = False


def _language_heading(language: str, templates: AppTemplates) -> str:
    payload: object = getattr(templates, "google_doc_language_headings", {})
    if isinstance(payload, dict):
        return str(payload.get(language, payload.get("other", "OTHER")))
    return "OTHER"


def _render_template(template: str, values: Dict[str, Any]) -> str:
    try:
        return template.format(**values)
    except KeyError as error:
        raise RuntimeError(f"Template render failed, missing key: {error}") from error


def _build_doc_header_text(
    *,
    header_context: Dict[str, str],
    templates: AppTemplates,
) -> str:
    context: Dict[str, str] = dict(header_context)
    context.setdefault("language_time_titles", "")
    return _render_template(templates.google_doc_header, context)


def _document_processing_mode_label(*, processing_mode: str, branch_label: str) -> str:
    normalized_branch_label: str = str(branch_label or "").strip().lower()
    if normalized_branch_label.startswith("audit/"):
        return normalized_branch_label.replace("/", "_")
    return normalized_branch_label or str(processing_mode or "").strip()


def _artifact_heading_label(
    *,
    heading: str,
    merged_content: Optional[MergedLanguageContent],
    merge_attempt: Optional[LanguageMergeAttempt],
    artifact_status: str,
) -> str:
    block_generation_mode: str = resolve_block_generation_mode(
        merge_attempt=merge_attempt,
        merged_content=merged_content,
    )
    if block_generation_mode == BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE:
        return f"{heading} ⚠ [merge failed — source list]"
    return heading


def _log_merge_block_publish_truth(
    *,
    logger: logging.Logger,
    branch_label: str,
    date_key: str,
    slot_key: str,
    language: str,
    source_videos: List[PlannedVideo],
    merged_content: Optional[MergedLanguageContent],
    merge_attempt: Optional[LanguageMergeAttempt],
) -> None:
    if merged_content is None and merge_attempt is None:
        return
    logger.info(
        "merge_block_publish_truth branch=%s date_key=%s slot_key=%s language=%s block_generation_mode=%s artifact_marker_applied=%s title_source=%s hook_source=%s hashtags_source=%s body_source=%s source_video_count=%d",
        branch_label,
        date_key,
        slot_key,
        language,
        resolve_block_generation_mode(
            merge_attempt=merge_attempt,
            merged_content=merged_content,
        ),
        (
            "yes"
            if resolve_block_generation_mode(
                merge_attempt=merge_attempt,
                merged_content=merged_content,
            )
            == BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE
            else "no"
        ),
        (
            str(
                (merge_attempt.title_source if merge_attempt is not None else None)
                or (merged_content.title_source if merged_content is not None else None)
                or "unknown"
            )
        ),
        (
            str(
                (merge_attempt.hook_source if merge_attempt is not None else None)
                or (merged_content.hook_source if merged_content is not None else None)
                or "unknown"
            )
        ),
        (
            str(
                (merge_attempt.hashtags_source if merge_attempt is not None else None)
                or (merged_content.hashtags_source if merged_content is not None else None)
                or "unknown"
            )
        ),
        (
            str(
                (merge_attempt.body_source if merge_attempt is not None else None)
                or (merged_content.body_source if merged_content is not None else None)
                or "unknown"
            )
        ),
        len(source_videos),
    )


def _export_document_to_local_docx(
    *,
    logger: logging.Logger,
    drive_client: GoogleDriveClient,
    name_builder: NamePathBuilder,
    document_id: str,
    date_key: str,
    doc_title: str,
) -> Optional[Path]:
    local_doc_path: Optional[Path] = name_builder.build_docx_path(
        date_key=date_key,
        doc_title=doc_title,
    )
    if local_doc_path is None:
        return None
    try:
        docx_bytes: bytes = drive_client.export_google_doc_as_docx(document_id)
        local_doc_path.parent.mkdir(parents=True, exist_ok=True)
        local_doc_path.write_bytes(docx_bytes)
        logger.info('doc_local_exported path="%s"', local_doc_path)
        return local_doc_path
    except Exception as error:
        logger.warning(
            'doc_local_export_failed reason="%s"',
            summarize_error(error),
        )
        return None


def publish_daily_document(
    *,
    logger: logging.Logger,
    config: AppConfig,
    docs_client: GoogleDocsClient,
    drive_client: GoogleDriveClient,
    report_writer: GoogleDocsReportWriter,
    name_builder: NamePathBuilder,
    slot_results: List[SlotProcessResult],
    date_key: str,
    dry_run: bool,
    processing_mode: str,
    kiev_tz: ZoneInfo,
    branch_label: str,
) -> DailyDocumentPublishResult:
    header_blocks: List[str] = []
    for slot in slot_results:
        time_display: str = format_time_key_for_display(slot.slot_time_key)
        for language in ("uk", "en", "ru", "other"):
            slot_language_items: List[PlannedVideo] = slot.language_groups[language]
            if not slot_language_items:
                continue
            merged_content: Optional[MergedLanguageContent] = (
                slot.merged_content_by_language.get(language)
            )
            merge_attempt: Optional[LanguageMergeAttempt] = (
                slot.merge_audit_by_language.get(language)
            )
            merged_title: str = ""
            if processing_mode == "nomerge":
                merged_title = _build_titles_summary(videos=slot_language_items).strip()
            else:
                if merged_content is not None and merged_content.title.strip():
                    merged_title = sanitize_post_llm_title(merged_content.title.strip())
                else:
                    merged_title = _build_titles_summary(
                        videos=slot_language_items,
                        merged_content=merged_content,
                        merge_attempt=merge_attempt,
                    ).strip()
            if not merged_title:
                continue
            heading_text: str = _artifact_heading_label(
                heading=f"{_language_heading(language, config.templates)} - {time_display}",
                merged_content=merged_content,
                merge_attempt=merge_attempt,
                artifact_status=slot.merge_artifact_status,
            )
            header_blocks.append(
                f"{heading_text}\n{merged_title}"
            )
    language_time_titles: str = "\n\n".join(header_blocks).strip()
    used_runtime_models = collect_used_runtime_models(
        slot_results=slot_results,
        configured_model=resolve_effective_llm_model(config),
    )
    real_merge_blocks: int = sum(slot.real_merge_blocks for slot in slot_results)
    merge_candidate_blocks: int = sum(slot.merge_candidate_blocks for slot in slot_results)
    fallback_merge_blocks: int = sum(slot.fallback_merge_blocks for slot in slot_results)
    merge_artifact_status: str = resolve_merge_artifact_status(
        merge_candidate_blocks=merge_candidate_blocks,
        real_merge_blocks=real_merge_blocks,
        fallback_merge_blocks=fallback_merge_blocks,
    )
    fallback_merge_targets: List[str] = [
        target
        for slot in slot_results
        for target in slot.fallback_merge_targets
    ]
    doc_title: str = name_builder.build_doc_title(
        date_key=date_key,
        created_at=datetime.now(kiev_tz),
        processing_mode=_document_processing_mode_label(
            processing_mode=processing_mode,
            branch_label=branch_label,
        ),
        llm_models_segment=(
            render_doc_title_models_segment(used_runtime_models=used_runtime_models)
            if branch_label == BRANCH_MERGE
            else ""
        ),
    )
    first_header_context: Dict[str, str] = dict(slot_results[0].header_context)
    first_header_context["language_time_titles"] = language_time_titles
    local_export_path: Optional[Path] = name_builder.build_docx_path(
        date_key=date_key,
        doc_title=doc_title,
    )
    logger.info(
        '[%s] doc_publish_start date_key=%s slot_count=%d dry_run=%s local_export_enabled=%s local_export_path="%s" merge_artifact_status=%s merge_candidate_blocks=%d fallback_merge_blocks=%d fallback_targets=%s effective_model=%s',
        branch_label,
        date_key,
        len(slot_results),
        dry_run,
        local_export_path is not None,
        str(local_export_path) if local_export_path is not None else "",
        merge_artifact_status,
        merge_candidate_blocks,
        fallback_merge_blocks,
        ",".join(fallback_merge_targets) or "none",
        used_runtime_models.configured_model or "unknown",
    )

    doc_url: str = "DRY_RUN_DOC_URL"
    google_doc_created: bool = False
    exported_docx_path: Optional[Path] = None
    if not dry_run:
        try:
            document_id: str = docs_client.create_document(title=doc_title)
            google_doc_created = True
        except Exception as error:
            raise RuntimeError(
                "Google Docs create_document failed. "
                f"date={date_key} title={doc_title!r} "
                f"reason={summarize_error(error)}"
            ) from error
        if config.google.doc_share_mode != "private":
            role_by_mode: Dict[str, str] = {
                "anyone_reader": "reader",
                "anyone_commenter": "commenter",
                "anyone_writer": "writer",
            }
            role: str = role_by_mode[config.google.doc_share_mode]
            try:
                drive_client.set_anyone_permission(file_id=document_id, role=role)
            except Exception as error:
                raise RuntimeError(
                    "Google Drive set_anyone_permission failed. "
                    f"document_id={document_id} "
                    f"share_mode={config.google.doc_share_mode} role={role} "
                    f"reason={summarize_error(error)}"
                ) from error
        if config.google.drive_folder_id:
            try:
                drive_client.move_file_to_folder(
                    file_id=document_id,
                    folder_id=config.google.drive_folder_id,
                )
            except Exception as error:
                raise RuntimeError(
                    "Google Drive move_file_to_folder failed. "
                    f"document_id={document_id} "
                    f"folder_id={config.google.drive_folder_id} "
                    f"reason={summarize_error(error)}"
                ) from error
        try:
            report_writer.write_header_only(
                document_id=document_id,
                header_text=_build_doc_header_text(
                    header_context=first_header_context,
                    templates=config.templates,
                ),
            )
            guard_index: int = docs_client.get_document_end_index(document_id=document_id) - 1
            docs_client.insert_text_at_index(
                document_id=document_id,
                index=guard_index,
                text="\n",
            )
            logger.info(
                "docs_formatting_guard date_key=%s inserted_blank_paragraph_before_first_table index=%d",
                date_key,
                guard_index,
            )
            first_table: bool = True
            for slot in slot_results:
                time_display = format_time_key_for_display(slot.slot_time_key)
                for language in ("uk", "en", "ru", "other"):
                    if not slot.language_groups[language]:
                        continue
                    merged_content = slot.merged_content_by_language.get(language)
                    merge_attempt = slot.merge_audit_by_language.get(language)
                    _log_merge_block_publish_truth(
                        logger=logger,
                        branch_label=branch_label,
                        date_key=date_key,
                        slot_key=slot.slot_key,
                        language=language,
                        source_videos=slot.language_groups[language],
                        merged_content=merged_content,
                        merge_attempt=merge_attempt,
                    )
                    if not first_table:
                        report_writer.insert_page_break(document_id=document_id)
                    report_writer.write_language_table(
                        document_id=document_id,
                        language=language,
                        videos=slot.language_groups[language],
                        merged_content=merged_content,
                        merge_attempt=merge_attempt,
                        time_display=time_display,
                        artifact_status=slot.merge_artifact_status,
                    )
                    first_table = False
        except Exception as error:
            raise RuntimeError(
                "Google Docs write_daily_document failed. "
                f"document_id={document_id} date={date_key} "
                f"reason={summarize_error(error)}"
            ) from error
        exported_docx_path = _export_document_to_local_docx(
            logger=logger,
            drive_client=drive_client,
            name_builder=name_builder,
            document_id=document_id,
            date_key=date_key,
            doc_title=doc_title,
        )
        doc_url = f"https://docs.google.com/document/d/{document_id}/edit"

    logger.info(
        "[%s] date=%s: Google Doc created: %s (slots=%d)",
        branch_label,
        date_key,
        doc_url,
        len(slot_results),
    )
    slot_keys: List[str] = [slot.slot_key for slot in slot_results]
    slots_list_text: str = f"[{','.join(slot_keys)}]"
    logger.info(
        "[%s] date_key=%s doc_url=%s slots=%s doc_grouping=date telegram_send_mode=per_date",
        branch_label,
        date_key,
        doc_url,
        slots_list_text,
    )
    logger.info(
        "[%s] doc_publish_finish date_key=%s slot_count=%d doc_url=%s",
        branch_label,
        date_key,
        len(slot_results),
        doc_url,
    )
    logger.info(
        '[%s] doc_publish_forensic date_key=%s local_docx_exported=%s local_export_path="%s" google_doc_created=%s google_doc_url=%s target_folder_id=%s',
        branch_label,
        date_key,
        "yes" if exported_docx_path is not None else "no",
        str(exported_docx_path) if exported_docx_path is not None else "",
        "yes" if google_doc_created else "no",
        doc_url,
        str(config.google.drive_folder_id or ""),
    )
    return DailyDocumentPublishResult(
        doc_title=doc_title,
        doc_url=doc_url,
        header_context=first_header_context,
        slot_keys=slot_keys,
        local_docx_exported=exported_docx_path is not None,
        local_export_path=exported_docx_path,
        google_doc_created=google_doc_created,
    )

