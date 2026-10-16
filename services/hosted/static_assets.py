"""Revalidated, compressed delivery of public application JavaScript and CSS."""

from __future__ import annotations

import gzip
import hashlib
import os
import stat
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from starlette.requests import Request
from starlette.responses import Response

from services.hosted.errors import DomainError


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _encoding(header: str) -> str:
    quality: dict[str, float] = {}
    for item in header.lower().split(","):
        name, *parameters = item.strip().split(";")
        if not name:
            continue
        value = 1.0
        for parameter in parameters:
            key, separator, raw = parameter.strip().partition("=")
            if key == "q":
                try:
                    value = float(raw) if separator else 0.0
                except ValueError:
                    value = 0.0
                if not 0 <= value <= 1:
                    value = 0.0
        quality[name.strip()] = value
    compressed = quality.get("gzip", quality.get("*", 0.0))
    identity = quality.get("identity", 0.0 if quality.get("*") == 0 else 1.0)
    if compressed > 0 and compressed >= identity:
        return "gzip"
    if identity > 0:
        return "identity"
    raise DomainError("encoding_not_acceptable", 406)


@dataclass(frozen=True)
class _Asset:
    identity: tuple[int, ...]
    original: bytes
    compressed: bytes
    original_etag: str
    compressed_etag: str

    @property
    def size(self) -> int:
        return len(self.original) + len(self.compressed)


class PublicStaticAssets:
    """Cache only code assets, bounded by both total bytes and file count.

    The browser must revalidate on every reuse. File replacement or editing
    invalidates the server's cached representations without requiring a restart.
    """

    def __init__(self, root: Path, *, max_bytes: int = 64 * 1024 * 1024, max_files: int = 64):
        self.root = root.resolve()
        self.max_bytes = max_bytes
        self.max_files = max_files
        self._cache: OrderedDict[Path, _Asset] = OrderedDict()
        self._bytes = 0
        self._lock = Lock()

    def _load(self, path: Path) -> _Asset:
        with self._lock:
            for _attempt in range(2):
                info = path.stat()
                if not stat.S_ISREG(info.st_mode):
                    raise DomainError("not_found", 404)
                identity = _file_identity(info)
                previous = self._cache.get(path)
                if previous and previous.identity == identity:
                    self._cache.move_to_end(path)
                    return previous
                # The resolved direct child must not turn into a symlink while
                # opening it. An in-progress edit is retried, never cached.
                descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(descriptor, "rb") as stream:
                    before = os.fstat(stream.fileno())
                    if not stat.S_ISREG(before.st_mode):
                        raise DomainError("not_found", 404)
                    original = stream.read()
                    after = os.fstat(stream.fileno())
                if (identity != _file_identity(before) or identity != _file_identity(after)
                        or identity != _file_identity(path.stat()) or len(original) != info.st_size):
                    continue
                compressed = gzip.compress(original, compresslevel=6, mtime=0)
                asset = _Asset(identity, original, compressed,
                    '"' + hashlib.sha256(original).hexdigest() + '"',
                    '"' + hashlib.sha256(compressed).hexdigest() + '"')
                if previous:
                    self._bytes -= self._cache.pop(path).size
                if asset.size <= self.max_bytes and self.max_files > 0:
                    while self._cache and (self._bytes + asset.size > self.max_bytes
                                           or len(self._cache) >= self.max_files):
                        self._bytes -= self._cache.popitem(last=False)[1].size
                    self._cache[path] = asset
                    self._bytes += asset.size
                return asset
        raise DomainError("asset_changing", 503, "The application file is being updated. Retry shortly.")

    def response(self, name: str, request: Request) -> Response:
        candidate = Path(name)
        if candidate.name != name or candidate.suffix not in {".js", ".css"}:
            raise DomainError("not_found", 404)
        try:
            path = (self.root / name).resolve(strict=True)
            if path.parent != self.root:
                raise DomainError("not_found", 404)
            encoding = _encoding(request.headers.get("accept-encoding", ""))
            asset = self._load(path)
        except (OSError, RuntimeError):
            raise DomainError("not_found", 404) from None
        body = asset.compressed if encoding == "gzip" else asset.original
        etag = asset.compressed_etag if encoding == "gzip" else asset.original_etag
        headers = {"Cache-Control": "public, no-cache", "ETag": etag, "Vary": "Accept-Encoding"}
        if encoding == "gzip":
            headers["Content-Encoding"] = "gzip"
        matches = []
        for value in request.headers.get("if-none-match", "").split(","):
            value = value.strip()
            matches.append(value[2:] if value.startswith("W/") else value)
        request.state.public_static_response = True
        if "*" in matches or etag in matches:
            return Response(status_code=304, headers=headers)
        headers["Content-Length"] = str(len(body))
        return Response(b"" if request.method == "HEAD" else body, headers=headers,
            media_type="text/javascript" if candidate.suffix == ".js" else "text/css")
