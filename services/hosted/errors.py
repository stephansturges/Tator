"""Stable errors shared by hosted persistence, HTTP and resource admission."""

from __future__ import annotations

from typing import Any


class DomainError(Exception):
    def __init__(self, code: str, status: int = 409, detail: Any = None) -> None:
        self.code = code
        self.status = status
        self.detail = detail
        super().__init__(code)

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "detail": self.detail}
