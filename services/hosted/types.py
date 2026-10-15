"""Authenticated identities and optimistic mutation preconditions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Actor:
    user_id: str
    role: str
    username: str = ""
    # Server-issued proof, never accepted from an HTTP request body. Empty
    # values are reserved for trusted administrative CLI/component callers.
    session_id: str = ""
    instance_id: str = ""


@dataclass(frozen=True)
class MutationContext:
    image_id: str
    expected_version: str
    generation: int
    editor_id: str
    operation_id: str
