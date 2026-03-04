from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Tuple

from app.config.settings import AppConfig
from app.core.constants import LOGGER_NAME_ENV_VAR
from app.core.env_flags import llm_allow_in_dry_run_from_env
from app.core.error_summary import summarize_error
from app.observability.runtime_analytics import (
    log_error_event,
    log_warning_informational,
    log_warning_operational,
)
from app.pipeline.runtime_services import BatchServices
from app.telegram.bot_client import TelegramBotClient


LoggerNameMetaResolver = Callable[[], Tuple[str, str, bool]]


def log_section(*, logger: logging.Logger, title: str) -> None:
    logger.info("")
    logger.info("=== %s ===", title)


def _resolve_llm_merge_enabled(
    *,
    logger: logging.Logger,
    config: AppConfig,
    dry_run: bool,
    resolved_audit_mode: str,
    run_id: str,
) -> bool:
    primary_attempts: int = 2
    fallback_enabled: bool = True
    log_section(logger=logger, title="LLM API")
    logger.info(
        "run_id=%s Processing mode selected: audit audit_mode=%s",
        run_id,
        resolved_audit_mode,
    )
    if resolved_audit_mode == "unite":
        logger.info("LLM selection: audit branch execution=nomerge,merge")
    else:
        logger.info("LLM selection: audit branch execution=%s", resolved_audit_mode)
    logger.info(
        "run_id=%s LLM provider selected: %s",
        run_id,
        config.llm_provider,
    )
    logger.info(
        "OpenAI merge policy model.primary=%s model.fallback=%s primary_attempts=%d fallback_enabled=%s timeout_sec=%.1f max_output_tokens=%d source_desc_max_chars=%d pre_delay_sec=%.1f",
        config.openai_model_primary,
        config.openai_model_fallback,
        primary_attempts,
        "yes" if fallback_enabled else "no",
        config.openai_timeout_sec,
        config.openai_max_output_tokens,
        config.llm_source_desc_max_chars,
        config.openai_pre_delay_sec,
    )
    gpt_api_key_present: bool = bool(os.getenv("GPT_API_KEY", "").strip())
    llm_merge_enabled: bool = False
    merge_mode_enabled: bool = resolved_audit_mode in {"merge", "unite"}
    llm_allow_in_dry_run: bool = llm_allow_in_dry_run_from_env()
    if merge_mode_enabled:
        llm_merge_enabled = gpt_api_key_present
        if dry_run and not llm_allow_in_dry_run:
            llm_merge_enabled = False
            logger.info(
                "LLM selection: disabled in dry-run by STG_LLM_ALLOW_IN_DRY_RUN=0."
            )
    if llm_merge_enabled:
        logger.info(
            "LLM selection: provider=%s enabled for merge stage.",
            config.llm_provider,
        )
    else:
        if not merge_mode_enabled:
            logger.info("llm_merge=skipped reason=audit_mode=nomerge")
        elif dry_run and not llm_allow_in_dry_run:
            logger.info(
                "LLM selection: dry-run mode -> merge disabled, append fallback will be used."
            )
        else:
            log_warning_operational(
                logger,
                "LLM selection: disabled (missing provider prerequisites). Continue without merge stage."
            )
    return llm_merge_enabled


def run_startup_health_checks(
    *,
    logger: logging.Logger,
    config: AppConfig,
    services: BatchServices,
    telegram_client: TelegramBotClient,
    resolve_logger_name_meta: LoggerNameMetaResolver,
    dry_run: bool,
    resolved_audit_mode: str,
    run_id: str,
) -> bool:
    startup_errors: List[str] = []
    logger.info("Startup identity check: begin.")
    logger.info("Logger name in use: %s", logger.name)
    logger_name_resolved, logger_name_source, logger_env_present = (
        resolve_logger_name_meta()
    )
    logger.info(
        "Logger env probe: var=%s env_present=%s source=%s resolved=%s",
        LOGGER_NAME_ENV_VAR,
        logger_env_present,
        logger_name_source,
        logger_name_resolved,
    )

    log_section(logger=logger, title="Environment")
    try:
        google_auth_mode: str = services.factory.get_auth_mode()
        logger.info("Google auth mode: %s", google_auth_mode)
        if google_auth_mode == "oauth":
            oauth_credentials_path, oauth_token_path = services.factory.get_oauth_paths()
            logger.info(
                "Google OAuth paths: credentials=%s token=%s",
                oauth_credentials_path,
                oauth_token_path,
            )
            logger.info(
                "Google auth hint: default GOOGLE_AUTH_MODE=oauth; set GOOGLE_AUTH_MODE=service_account to force service account mode."
            )
    except Exception as error:
        issue: str = f"Google auth mode check failed: {summarize_error(error)}"
        startup_errors.append(issue)
        log_error_event(logger, "%s", issue, reason_code="google_auth_mode_check_failed")

    try:
        google_runtime_principal: str = services.factory.get_runtime_principal_email()
        logger.info(
            "Google runtime account (script): %s",
            google_runtime_principal,
        )
    except Exception as error:
        issue = f"Google runtime account lookup failed: {summarize_error(error)}"
        startup_errors.append(issue)
        log_error_event(logger, "%s", issue, reason_code="google_runtime_account_lookup_failed")

    try:
        google_project_id, google_project_name = services.factory.get_google_project_info(
            strict=True
        )
        logger.info(
            "Google project resolved via API: name=%s id=%s",
            google_project_name,
            google_project_id,
        )
    except Exception as error:
        log_warning_informational(
            logger,
            "Google project API lookup skipped: %s",
            summarize_error(error),
            reason_code="google_project_id_unavailable",
        )

    log_section(logger=logger, title="Google APIs")
    try:
        sheets_owner_info: str = services.drive_client.get_file_owner_info(
            file_id=config.google_sheets_id
        )
        logger.info(
            "Google owner account for Sheets file: %s",
            sheets_owner_info,
        )
    except Exception as error:
        issue = f"Google Sheets owner lookup failed: {summarize_error(error)}"
        startup_errors.append(issue)
        log_error_event(logger, "%s", issue, reason_code="google_sheets_owner_lookup_failed")

    try:
        sheets_id_resolved, sheets_title = services.sheets_client.ping_access(
            spreadsheet_id=config.google_sheets_id
        )
        logger.info(
            "Google Sheets API OK. Server response: spreadsheet_id=%s, title=%s",
            sheets_id_resolved,
            sheets_title,
        )
    except Exception as error:
        issue = f"Google Sheets connection failed: {summarize_error(error)}"
        startup_errors.append(issue)
        log_error_event(logger, "%s", issue, reason_code="google_sheets_connection_failed")

    try:
        drive_user_name, drive_user_email = services.drive_client.ping_access()
        logger.info(
            "Google Drive API OK. Server response: user=%s <%s>",
            drive_user_name,
            drive_user_email,
        )
    except Exception as error:
        issue = f"Google Drive connection failed: {summarize_error(error)}"
        startup_errors.append(issue)
        log_error_event(logger, "%s", issue, reason_code="google_drive_connection_failed")

    try:
        docs_probe_result: str = services.docs_client.ping_access()
        logger.info("Google Docs API OK. Server response: %s", docs_probe_result)
    except Exception as error:
        issue = f"Google Docs connection failed: {summarize_error(error)}"
        startup_errors.append(issue)
        log_error_event(logger, "%s", issue, reason_code="google_docs_connection_failed")

    llm_merge_enabled: bool = _resolve_llm_merge_enabled(
        logger=logger,
        config=config,
        dry_run=dry_run,
        resolved_audit_mode=resolved_audit_mode,
        run_id=run_id,
    )

    log_section(logger=logger, title="Telegram API")
    try:
        if not config.telegram_enabled:
            raise RuntimeError("telegram.enabled=false")
        telegram_me: Dict[str, Any] = telegram_client.get_me()
        telegram_bot_name: str = str(
            telegram_me.get("username")
            or telegram_me.get("first_name")
            or telegram_me.get("id")
            or "unknown"
        ).strip()
        logger.info("Telegram bot identity: %s", telegram_bot_name)
        logger.info(
            "Telegram API OK. Server response: bot=%s, id=%s",
            telegram_bot_name,
            str(telegram_me.get("id", "unknown")),
        )
    except Exception as error:
        issue = f"Telegram connection failed: {summarize_error(error)}"
        startup_errors.append(issue)
        log_error_event(logger, "%s", issue, reason_code="telegram_connection_failed")

    if startup_errors:
        log_warning_operational(
            logger,
            "Startup health-check decision: CONTINUE_WITH_WARNINGS. Failed checks: %d",
            len(startup_errors),
            reason_code="startup_health_check_failed",
        )
        for issue in startup_errors:
            log_warning_operational(
                logger,
                "Startup check failure detail: %s",
                issue,
                reason_code="startup_health_check_failed",
            )
    else:
        logger.info("Startup health-check decision: CONTINUE.")
    return llm_merge_enabled
