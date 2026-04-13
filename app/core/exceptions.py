from __future__ import annotations


class ApplicationError(RuntimeError):
    """Base application error."""


class ConfigurationError(ApplicationError):
    """Configuration/validation error."""


class TemplateError(ConfigurationError):
    """Template loading/validation error."""
