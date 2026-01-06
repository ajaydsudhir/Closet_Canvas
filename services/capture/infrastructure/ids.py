from __future__ import annotations

import secrets


class TokenIdProvider:
    def __init__(self, prefix: str, length: int) -> None:
        self._prefix = prefix
        self._length = length

    def generate(self) -> str:
        token = secrets.token_urlsafe(self._length)
        return f"{self._prefix}_{token[: self._length]}"
