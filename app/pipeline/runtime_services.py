from __future__ import annotations

from dataclasses import dataclass

from app.config.settings import AppConfig
from app.google import (
    GoogleDocsClient,
    GoogleDriveClient,
    GoogleServicesFactory,
    GoogleSheetsClient,
)
from app.publish.google_docs_writer import GoogleDocsReportWriter


@dataclass(frozen=True)
class BatchServices:
    factory: GoogleServicesFactory
    sheets_client: GoogleSheetsClient
    docs_client: GoogleDocsClient
    drive_client: GoogleDriveClient
    report_writer: GoogleDocsReportWriter


def build_runtime_services(*, config: AppConfig) -> BatchServices:
    services_factory: GoogleServicesFactory = GoogleServicesFactory(
        auth_mode=config.google.auth_mode,
        service_account_path=config.google.service_account_path,
    )
    sheets_client: GoogleSheetsClient = GoogleSheetsClient(
        sheets_service=services_factory.create_sheets_service()
    )
    docs_client: GoogleDocsClient = GoogleDocsClient(
        docs_service=services_factory.create_docs_service()
    )
    drive_client: GoogleDriveClient = GoogleDriveClient(
        drive_service=services_factory.create_drive_service()
    )
    report_writer: GoogleDocsReportWriter = GoogleDocsReportWriter(
        docs_client=docs_client,
        templates=config.templates,
    )
    return BatchServices(
        factory=services_factory,
        sheets_client=sheets_client,
        docs_client=docs_client,
        drive_client=drive_client,
        report_writer=report_writer,
    )
