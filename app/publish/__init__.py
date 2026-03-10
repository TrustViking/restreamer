from __future__ import annotations

__all__ = ["GoogleDocsReportWriter"]


def __getattr__(name: str) -> object:
    if name == "GoogleDocsReportWriter":
        from .google_docs_writer import GoogleDocsReportWriter

        return GoogleDocsReportWriter
    raise AttributeError(name)
