"""Derived preview thumbnails for authorized hosted images.

Thumbnails are a disposable cache. They are made from the same immutable
working view the canvas displays, so orientation and alpha handling match the
annotation grid. Originals, working views and annotations are never modified,
and deleting the cache directory only costs regeneration time.
"""
from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

from PIL import Image, ImageOps

from .errors import DomainError
from .media import MEDIA_VIEW_POLICY

THUMBNAIL_VERSION = "jpeg-q82-v1"
THUMBNAIL_SIZES = (192, 384, 768)
DEFAULT_THUMBNAIL_SIZE = 384


class Thumbnails:
    def __init__(self, root: str | Path, media, artifacts, *, concurrency: int = 3):
        self.root = Path(root).absolute()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.media, self.artifacts = media, artifacts
        self._slots = threading.BoundedSemaphore(concurrency)
        self._guard = threading.Lock()
        self._keys: dict[tuple[str, int], threading.Lock] = {}

    @staticmethod
    def etag(digest: str, size: int) -> str:
        return f'"{digest}-{size}-{THUMBNAIL_VERSION}-{MEDIA_VIEW_POLICY}"'

    def path(self, digest: str, size: int) -> Path:
        return self.root / THUMBNAIL_VERSION / MEDIA_VIEW_POLICY / digest[:2] / f"{digest}-{size}.jpg"

    def get(self, digest: str, size: int) -> Path:
        """Return the cached thumbnail, generating it once if needed."""
        if size not in THUMBNAIL_SIZES:
            raise DomainError("thumbnail_size_invalid", 422, f"Use one of {', '.join(map(str, THUMBNAIL_SIZES))} pixels.")
        target = self.path(digest, size)
        if target.is_file():
            return target
        with self._guard:
            key = self._keys.setdefault((digest, size), threading.Lock())
        with key, self._slots:
            if not target.is_file():
                self._render(digest, size, target)
        with self._guard:
            self._keys.pop((digest, size), None)
        return target

    def _render(self, digest: str, size: int, target: Path) -> None:
        view = self.media.resolve(digest)
        with self.artifacts.open(self.media.reference(view)) as stream, Image.open(stream) as picture:
            # JPEG can decode at a reduced scale, which avoids full-resolution work.
            picture.draft("RGB", (size * 2, size * 2))
            picture.load()
            if picture.mode in ("RGBA", "LA", "PA") or (picture.mode == "P" and "transparency" in picture.info):
                rgba = picture.convert("RGBA")
                image = Image.new("RGB", rgba.size, "white")
                image.paste(rgba, mask=rgba.getchannel("A"))
            else:
                image = picture.convert("RGB")
            image = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS) if max(image.size) > size else image
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=".thumb-", suffix=".jpg")
        try:
            with os.fdopen(descriptor, "wb") as output:
                image.save(output, "JPEG", quality=82, optimize=True, progressive=True)
                output.flush()
                os.fsync(output.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
