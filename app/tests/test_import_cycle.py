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
