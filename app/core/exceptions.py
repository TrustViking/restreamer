from __future__ import annotations


class RestreamerError(RuntimeError):
    """Base application error."""


class ConfigurationError(RestreamerError):
    """Configuration/validation error."""


class TemplateError(ConfigurationError):
    """Template loading/validation error."""
