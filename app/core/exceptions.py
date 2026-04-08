from __future__ import annotations


class RestreamerError(RuntimeError):
    """Base application error."""


# Legacy alias
StreamertgError = RestreamerError


class ConfigurationError(RestreamerError):
    """Configuration/validation error."""


class TemplateError(ConfigurationError):
    """Template loading/validation error."""
