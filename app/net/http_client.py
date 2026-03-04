from __future__ import annotations

from typing import Any, Dict, cast

import requests


class HttpClient:
    def __init__(self, timeout_seconds: float = 20.0) -> None:
        self._timeout_seconds: float = timeout_seconds

    def get_bytes(self, url: str) -> bytes:
        response: requests.Response = requests.get(url, timeout=self._timeout_seconds)
        response.raise_for_status()
        return response.content

    def post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response: requests.Response = requests.post(
            url,
            json=payload,
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        return cast(Dict[str, Any], response.json())
