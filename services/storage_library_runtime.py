"""Keep storage-use protection alive through ASGI streaming and disconnects."""

from __future__ import annotations

from typing import Any, Callable
import re

from starlette.responses import JSONResponse

from services.storage_library import StorageLibraryError


class StorageLibraryMiddleware:
    """Serialize legacy resource access against library removal.

    A FileResponse opens its file after the route function returns. Holding the
    admission guard around the complete ASGI call also covers the final body and
    background response callbacks. Jobs outliving that call have separate guards.
    Library mutations and legacy management routes enter their own coordinator.
    """

    def __init__(
        self,
        app: Any,
        *,
        service_factory: Callable[[], Any],
        owns_request: Callable[[dict[str, Any]], bool],
    ) -> None:
        self.app = app
        self.service_factory = service_factory
        self.owns_request = owns_request

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http" and scope.get("method") in {"POST", "PUT", "PATCH", "DELETE"}:
            backend = dict(scope.get("headers") or []).get(b"x-tator-backend-id", b"").decode("utf-8", errors="replace")
            if backend and backend != self.service_factory().library.backend_id:
                await JSONResponse({"detail": {"code": "storage_library_backend_changed", "message": "The backend identity changed. Reconnect before saving or removing data."}}, status_code=409)(scope, receive, send)
                return
        if scope["type"] != "http" or not self.owns_request(scope):
            await self.app(scope, receive, send)
            return
        started = False
        status = 500

        async def tracked_send(message: dict[str, Any]) -> None:
            nonlocal started, status
            if message["type"] == "http.response.start":
                started = True
                status = message["status"]
            await send(message)

        try:
            service = self.service_factory()
            with service.library.protect(purpose="http", owner="request"):
                await self.app(scope, receive, tracked_send)
                path = str(scope.get("path") or "")
                publishes_assets = path in {"/datasets/upload", "/datasets/register_path", "/datasets/open_path"} or bool(
                    re.fullmatch(r"/datasets/(?:upload_session/[^/]+/(?:finalize|cancel)|trash/[^/]+/restore|transient/[^/]+/save)", path)
                    or re.fullmatch(r"/class_analysis/jobs/[^/]+/annotation_session", path)
                )
                if status < 400 and scope.get("method") in {"POST", "PUT", "PATCH", "DELETE"} and publishes_assets:
                    service.library.invalidate()
        except StorageLibraryError as exc:
            if started:
                raise
            await JSONResponse(
                {"detail": exc.detail}, status_code=exc.status_code
            )(scope, receive, send)
