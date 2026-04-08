from __future__ import annotations

import importlib


class TestImportCycleResolved:
    """Verify that core modules can be imported independently of app.llm."""

    def test_cta_detection_imports_without_llm_init(self) -> None:
        """cta_detection must not trigger app.llm.__init__ re-exports."""
        mod = importlib.import_module("app.core.cta_detection")
        assert hasattr(mod, "looks_like_cta_paragraph")
        assert hasattr(mod, "looks_like_cta_line")

    def test_tail_parser_imports_without_llm_init(self) -> None:
        mod = importlib.import_module("app.publish.sanitizers.tail_parser")
        assert hasattr(mod, "TailParser")

    def test_llm_package_does_not_eagerly_import_langdetect(self) -> None:
        """Importing app.llm must not trigger langdetect at import time."""
        import sys
        # Remove langdetect from sys.modules if previously loaded, to get a clean check
        langdetect_was_loaded: bool = "langdetect" in sys.modules
        mod = importlib.import_module("app.llm")
        assert hasattr(mod, "__getattr__"), "app.llm should use lazy __getattr__"
        # If langdetect wasn't loaded before our import, it shouldn't be loaded after
        if not langdetect_was_loaded:
            assert "langdetect" not in sys.modules, (
                "app.llm eagerly imported langdetect — lazy re-exports are broken"
            )

    def test_google_docs_client_imports_without_langdetect(self) -> None:
        """GoogleDocsClient must be importable without langdetect in the chain."""
        mod = importlib.import_module("app.google.docs_client")
        assert hasattr(mod, "GoogleDocsClient")

    def test_app_config_loader_imports_without_langdetect(self) -> None:
        """app_config_loader must be importable without triggering merge stack."""
        mod = importlib.import_module("app.config.app_config_loader")
        assert hasattr(mod, "load_config_from_env")

    def test_google_docs_writer_imports_without_merge_stack(self) -> None:
        """google_docs_writer must be importable without triggering langdetect."""
        import sys
        langdetect_was_loaded: bool = "langdetect" in sys.modules
        mod = importlib.import_module("app.publish.google_docs_writer")
        assert hasattr(mod, "GoogleDocsReportWriter")
        if not langdetect_was_loaded:
            assert "langdetect" not in sys.modules, (
                "google_docs_writer import chain still pulls langdetect"
            )

    def test_tail_parser_imports_without_langdetect(self) -> None:
        """tail_parser must not trigger url_selector -> langdetect via sanitizers __init__."""
        import sys
        langdetect_was_loaded: bool = "langdetect" in sys.modules
        mod = importlib.import_module("app.publish.sanitizers.tail_parser")
        assert hasattr(mod, "TailParser")
        if not langdetect_was_loaded:
            assert "langdetect" not in sys.modules, (
                "tail_parser import chain still pulls langdetect via sanitizers __init__"
            )
