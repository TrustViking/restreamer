from __future__ import annotations

from typing import Any

__all__: list[str] = [
    "TailParser",
    "AuthoritativeUrlSelector",
    "DescriptionComposer",
    "PublishQualityGate",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "TailParser": ("app.publish.sanitizers.tail_parser", "TailParser"),
    "AuthoritativeUrlSelector": ("app.publish.sanitizers.url_selector", "AuthoritativeUrlSelector"),
    "DescriptionComposer": ("app.publish.sanitizers.description_composer", "DescriptionComposer"),
    "PublishQualityGate": ("app.publish.sanitizers.quality_gate", "PublishQualityGate"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'app.publish.sanitizers' has no attribute {name!r}")
