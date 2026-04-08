from __future__ import annotations

from app.application.application import RestreamerApplication

# Legacy alias for backward compatibility
StreamertgApplication = RestreamerApplication

__all__: list[str] = ["RestreamerApplication", "StreamertgApplication"]
