from __future__ import annotations

from fastapi.routing import APIRoute

import localinferenceapi


def test_api_route_method_path_pairs_are_unique() -> None:
    seen: set[tuple[str, str]] = set()
    duplicates: list[tuple[str, str]] = []
    count = 0
    for route in localinferenceapi.app.routes:
        if not isinstance(route, APIRoute):
            continue
        path = route.path
        for method in route.methods or set():
            if method in {"HEAD", "OPTIONS"}:
                continue
            key = (method, path)
            count += 1
            if key in seen:
                duplicates.append(key)
            else:
                seen.add(key)
    assert not duplicates
    # Guardrail: ensure router registration remains broad and accidental route drops are caught.
    assert count >= 170
