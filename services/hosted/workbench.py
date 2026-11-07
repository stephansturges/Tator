"""Administrator-only access to the full legacy workbench through the hosted origin.

The workbench is the original single-user Tator backend (localinferenceapi),
run as a separate service with its own private storage root. It has no sign-in
of its own, so production serves it on a Unix socket inside a 0700 directory;
plain loopback HTTP is accepted for development. The
hosted control process never imports it; it serves the legacy page with an API
root under /workbench/api and forwards those requests, streaming both ways, only
for an authenticated enabled administrator. Canonical hosted data changes only
through the hosted apply step, never through this proxy.
"""
from __future__ import annotations

import json
import os
import re
import stat
from http.cookiejar import CookieJar, DefaultCookiePolicy
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from .errors import DomainError

TABS = ("labeling", "data-ingestion", "class-split", "dataset-pruning", "prepass-builder", "training",
        "qwen-train", "sam3-train", "yolo-train", "rfdetr-train", "agent-mining", "prompt-helper",
        "datasets", "sam3-prompt-models", "detectors", "active", "qwen", "predictors", "settings")
HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
               "trailers", "transfer-encoding", "upgrade"}
# Hosted credentials and browser origin stay at the boundary; the legacy service
# is unauthenticated and trusts only its loopback caller.
REQUEST_DROP = HOP_HEADERS | {"host", "cookie", "origin", "referer", "x-tator-csrf", "x-tator-instance"}
RESPONSE_DROP = HOP_HEADERS | {"set-cookie"}
PREFIX = "/workbench/api/"
UPLOAD_LIMIT = 64 * 1024 ** 3


UNIX_BASE = "http://workbench"


def workbench_url(value: str | None) -> str:
    """Validate the configured address: unix:/abs/path.sock, or loopback HTTP."""
    value = (value or "").strip().rstrip("/")
    if not value:
        return ""
    if value.startswith("unix:"):
        path = Path(value[5:])
        if not path.is_absolute() or len(str(path)) > 100:
            raise RuntimeError("TATOR_HOSTED_WORKBENCH_URL unix: needs an absolute socket path.")
        return value
    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or not parsed.port \
            or parsed.path or parsed.query or parsed.username:
        raise RuntimeError("TATOR_HOSTED_WORKBENCH_URL must be unix:/path/to/socket or a loopback address such as http://127.0.0.1:8020.")
    return value


def require_private_socket(url: str) -> None:
    """uvicorn makes its socket world-writable; only a private directory protects it."""
    if not url.startswith("unix:"):
        return
    try:
        info = os.stat(Path(url[5:]).parent)
    except FileNotFoundError:
        raise DomainError("workbench_unavailable", 503, "The workbench service is not running.") from None
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise DomainError("workbench_socket_not_private", 503,
                          "The workbench socket directory must belong to the service user with mode 0700.")


def client_options(url: str, *, asynchronous: bool) -> dict:
    """httpx client arguments for a validated workbench address.

    The client is shared by every administrator, so it must never keep cookies
    the legacy service sets and replay them on someone else's request.
    """
    jar = CookieJar(policy=DefaultCookiePolicy(allowed_domains=[]))
    if url.startswith("unix:"):
        transport = (httpx.AsyncHTTPTransport if asynchronous else httpx.HTTPTransport)(uds=url[5:])
        return {"base_url": UNIX_BASE, "transport": transport, "trust_env": False, "cookies": jar}
    return {"base_url": url, "trust_env": False, "cookies": jar}


def _page_unavailable() -> HTMLResponse:
    return HTMLResponse('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Workbench unavailable</title>'
                        '<link rel="stylesheet" href="/hosted.css"></head><body><main class="hosted-login">'
                        '<h1>Workbench unavailable</h1><p>This server has no workbench service configured. The hosted '
                        'admin tools remain available.</p><p><a href="/">Back to the admin workspace</a></p></main></body></html>',
                        status_code=503)


def install_workbench(app, *, auth, origin: str, cookie: str, web_root: Path, url: str):
    holder = {"client": None}

    def client() -> httpx.AsyncClient:
        if holder["client"] is None:
            holder["client"] = httpx.AsyncClient(**client_options(url, asynchronous=True), follow_redirects=False,
                                                 timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=30.0))
        return holder["client"]

    def administrator(request: Request, *, mutation: bool):
        identity = auth.authenticate(request.cookies.get(cookie, ""))
        auth.require_admin(identity)
        # The session cookie is SameSite=Strict; a changing request must also
        # name this exact origin, which browsers send for same-origin fetches.
        if mutation and request.headers.get("origin") != origin:
            raise DomainError("origin_required", 403, "Open the workbench from this server.")
        return identity

    @app.get("/workbench", response_class=HTMLResponse)
    async def workbench_page(request: Request, tab: str = "class-split", dataset: str = ""):
        try:
            await run_in_threadpool(administrator, request, mutation=False)
        except DomainError as error:
            if error.status == 401:
                return RedirectResponse("/", status_code=303)
            raise
        if not url:
            return _page_unavailable()
        boot = {"tab": tab if tab in TABS else "class-split",
                "dataset": dataset if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", dataset) else ""}
        head = ('<head><base href="/"><script>window.TATOR_WORKBENCH=Object.assign('
                + json.dumps(boot).replace("<", "\\u003c")
                + ',{apiRoot:location.origin+"/workbench/api"});</script>'
                '<link rel="stylesheet" href="/hosted_workbench.css">')
        html = (web_root / "tator.html").read_text(encoding="utf-8")
        html = html.replace("<head>", head, 1).replace("</body>", '<script src="/hosted_workbench.js"></script></body>', 1)
        return HTMLResponse(html)

    @app.api_route(PREFIX + "{path:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def workbench_proxy(path: str, request: Request):
        await run_in_threadpool(administrator, request, mutation=request.method not in ("GET", "HEAD", "OPTIONS"))
        if not url:
            raise DomainError("workbench_not_configured", 503, "This server has no workbench service configured.")
        require_private_socket(url)
        # Forward the exact encoded path so %23, %3F, %2F and %25 in file names
        # reach the legacy routes unchanged.
        raw = request.scope.get("raw_path") or request.url.path.encode()
        if not raw.startswith(PREFIX.encode()):
            raise DomainError("not_found", 404)
        query = request.scope.get("query_string") or b""
        target = httpx.URL(client().base_url, raw_path=raw[len(PREFIX) - 1:] + (b"?" + query if query else b""))
        headers = [(key, value) for key, value in request.headers.items() if key.lower() not in REQUEST_DROP]
        body = request.stream() if request.method not in ("GET", "HEAD", "OPTIONS") else None
        try:
            upstream = await client().send(client().build_request(request.method, target, headers=headers, content=body), stream=True)
        except httpx.HTTPError:
            raise DomainError("workbench_unavailable", 503, "The workbench service is not running. Ask the operator to start it.") from None
        response_headers = {key: value for key, value in upstream.headers.items() if key.lower() not in RESPONSE_DROP}
        location = upstream.headers.get("location")
        if location:
            # Keep legacy redirects inside the proxied API instead of sending the
            # browser to the private upstream address or the hosted root.
            base = str(client().base_url).rstrip("/")
            if location.startswith(base + "/"):
                location = location[len(base):]
            if location.startswith("/") and not location.startswith("//"):
                response_headers["location"] = PREFIX.rstrip("/") + location
            elif location.startswith("//") or urlsplit(location).scheme:
                response_headers.pop("location", None)  # another host: never follow
        return StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code, headers=response_headers,
                                 background=BackgroundTask(upstream.aclose))

    async def close():
        if holder["client"] is not None:
            await holder["client"].aclose()
            holder["client"] = None
    return close
