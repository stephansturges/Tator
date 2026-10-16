"""Encoded-coordinate media admission and immutable working representations.

The source digest remains membership/provenance identity. A working digest names
the exact neutral bytes used by a browser or native file consumer. No operation
rewrites source bytes or guesses the coordinate frame of existing annotations.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
from typing import BinaryIO

from PIL import Image, UnidentifiedImageError

from .artifacts import ArtifactRef, ArtifactStore, open_regular
from .errors import DomainError


MAX_MEDIA_BYTES = 256 * 1024 * 1024
MAX_MEDIA_PIXELS = 64_000_000
MEDIA_VIEW_POLICY = "encoded-rgb8-white-v2"
LEGACY_MEDIA_VIEW_POLICY = "encoded-rgb-v1"
COORDINATE_POLICY = "encoded-coordinates-v1"
FORMATS = {"JPEG": ("image/jpeg", ".jpg"), "PNG": ("image/png", ".png"),
           "WEBP": ("image/webp", ".webp"), "TIFF": ("image/tiff", ".tif"),
           "BMP": ("image/bmp", ".bmp")}


@dataclass(frozen=True)
class MediaInfo:
    source_sha256: str
    size_bytes: int
    width: int
    height: int
    format: str
    orientation: int
    needs_working_view: bool
    frame_count: int = 1

    def as_dict(self):
        return asdict(self)


def _read(source: bytes | BinaryIO | str | Path) -> bytes:
    if isinstance(source, bytes):
        data = source
    elif isinstance(source, (str, Path)):
        with open_regular(Path(source)) as stream:
            data = stream.read(MAX_MEDIA_BYTES + 1)
    else:
        source.seek(0)
        data = source.read(MAX_MEDIA_BYTES + 1)
        source.seek(0)
    if not isinstance(data, bytes) or len(data) > MAX_MEDIA_BYTES:
        raise DomainError("hosted_image_bytes_exceeded", 413)
    return data


def _png16_transparent(data, width, height):
    """Match tRNS in original sample precision before high-byte reduction."""
    import numpy as np
    try:
        import png
    except ImportError as exc:
        raise DomainError('media_decoder_unavailable', 503, 'Install requirements-hosted.txt for 16-bit PNG transparency.') from exc
    try:
        actual_width, actual_height, rows, metadata = png.Reader(bytes=data).read()
        if (actual_width, actual_height, metadata['bitdepth']) != (width, height, 16):
            raise ValueError('PNG dimensions or bit depth changed')
        planes = metadata['planes']
        if planes not in (1, 3) or 'transparent' not in metadata:
            raise ValueError('PNG transparency changed')
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        key = np.asarray(metadata['transparent'], dtype=np.uint16)
        count = 0
        for index, row in enumerate(rows):
            if index >= height:
                raise ValueError('PNG row count changed')
            samples = np.asarray(row, dtype=np.uint16).reshape(width, planes)
            reduced = (samples >> 8).astype(np.uint8)
            reduced[np.all(samples == key, axis=1)] = 255
            pixels[index] = reduced
            count += 1
        if count != height:
            raise ValueError('PNG row count changed')
        return Image.fromarray(pixels, mode='RGB')
    except (png.Error, ValueError, KeyError) as exc:
        raise DomainError('invalid_image_import', 422, 'The image cannot be fully decoded.') from exc


def decode_encoded(source: bytes | BinaryIO | str | Path) -> tuple[Image.Image, MediaInfo]:
    """Fully decode the first encoded image grid, ignoring display orientation.

    TIFF's Pillow decoder applies orientation during load_end, unlike JPEG.
    Neutralize its in-memory tag before loading; source bytes stay untouched.
    The fresh RGB result contains no metadata that can trigger another transform.
    """
    data = _read(source)
    try:
        with Image.open(io.BytesIO(data)) as opened:
            width, height = opened.size
            encoding = opened.format
            if encoding not in FORMATS:
                raise DomainError("invalid_image_import", 422, "Unsupported image encoding.")
            if width <= 0 or height <= 0 or width * height > MAX_MEDIA_PIXELS:
                raise DomainError("hosted_image_pixels_exceeded", 413)
            orientation = opened.getexif().get(274, 1)
            if type(orientation) is not int or orientation not in range(1, 9):
                raise DomainError("image_orientation_invalid", 422)
            frames = getattr(opened, "n_frames", 1)
            mode = opened.mode
            # Pillow exposes 16-bit truecolour PNG as RGB after decoding.
            depth = data[24] if encoding == "PNG" and len(data) > 25 else 8
            unsigned16 = mode.startswith("I;16") or (encoding == "PNG" and mode == "I" and depth == 16)
            transforms = any(key in opened.info for key in (
                "transparency", "icc_profile", "gamma", "chromaticity", "srgb", "xmp", "XML:com.adobe.xmp"))
            compatible = (encoding in {"JPEG", "PNG", "WEBP"} and mode in {"RGB", "L"}
                          and depth == 8 and orientation == 1 and frames == 1 and not transforms)
        # PNG EXIF access may load metadata/pixels. Its verify() requires a
        # pristine parser, so verify a separate open before the full decode.
        with Image.open(io.BytesIO(data)) as verifier:
            verifier.verify()
        with Image.open(io.BytesIO(data)) as opened:
            # getexif may derive orientation from XMP, so set the loaded EXIF
            # value even when the source does not contain an EXIF segment.
            opened.getexif()[274] = 1
            if encoding == "TIFF":
                opened.tag_v2[274] = 1
            opened.load()
            if opened.size != (width, height):
                raise DomainError("image_encoded_dimensions_changed", 422)
            if encoding == 'PNG' and depth == 16 and 'transparency' in opened.info:
                pixels = _png16_transparent(data, width, height)
            elif unsigned16:
                import numpy as np
                # Fixed source-range conversion, never contrast normalization.
                samples = np.asarray(opened).astype(np.uint32)
                reduced = (samples >> 8).astype(np.uint8)
                if 'transparency' in opened.info:
                    reduced[samples == opened.info['transparency']] = 255
                pixels = Image.fromarray(reduced, mode="L").convert("RGB")
            elif "A" in opened.getbands() or "transparency" in opened.info:
                rgba = opened.convert("RGBA")
                try:
                    matte = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                    try:
                        composited = Image.alpha_composite(matte, rgba)
                        try:
                            pixels = composited.convert("RGB")
                        finally:
                            composited.close()
                    finally:
                        matte.close()
                finally:
                    rgba.close()
            else:
                pixels = opened.convert("RGB")
            pixels.load()
            pixels.info.clear()
        info = MediaInfo(hashlib.sha256(data).hexdigest(), len(data), width, height,
                         encoding, orientation,
                         not compatible,
                         frames)
        return pixels, info
    except DomainError:
        raise
    except (OSError, ValueError, SyntaxError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise DomainError("invalid_image_import", 422, "The image cannot be fully decoded.") from error


def inspect_media(source: bytes | BinaryIO | str | Path) -> MediaInfo:
    pixels, info = decode_encoded(source)
    pixels.close()
    return info


def working_media_bytes(source: bytes | BinaryIO | str | Path) -> tuple[bytes, dict]:
    data = _read(source)
    pixels, info = decode_encoded(data)
    try:
        encoding = info.format
        if info.needs_working_view:
            buffer = io.BytesIO()
            pixels.save(buffer, format="PNG", compress_level=6)
            data, encoding = buffer.getvalue(), "PNG"
            if len(data) > MAX_MEDIA_BYTES:
                raise DomainError("hosted_image_bytes_exceeded", 413)
        mime, suffix = FORMATS[encoding]
        return data, {"policy": MEDIA_VIEW_POLICY, "coordinate_space": "encoded",
                      "source_sha256": info.source_sha256, "source_size_bytes": info.size_bytes,
                      "source_format": info.format, "orientation": info.orientation,
                      "frame_index": 0, "frame_count": info.frame_count,
                      "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
                      "width": info.width, "height": info.height,
                      "content_type": mime, "suffix": suffix}
    finally:
        pixels.close()


def validate_working_view(source, view: dict, *, source_sha256=None,
                          source_size_bytes=None, require_current_compatible=False) -> MediaInfo:
    """Verify exact captured working bytes; never replace them with today's view.

    Old opaque, neutral inputs remain executable. Unsafe or unidentified legacy
    captures remain readable evidence but cannot silently start new inference.
    """
    data = _read(source)
    info = inspect_media(data)
    if not isinstance(view, dict) or view.get("policy") not in {MEDIA_VIEW_POLICY, LEGACY_MEDIA_VIEW_POLICY}:
        raise DomainError("media_policy_upgrade_required", 409)
    expected = (view.get("sha256"), view.get("size_bytes"), view.get("width"), view.get("height"))
    if expected != (info.source_sha256, info.size_bytes, info.width, info.height):
        raise DomainError("snapshot_media_mismatch", 409)
    if (view.get("coordinate_space") != "encoded"
            or source_sha256 is not None and view.get("source_sha256") != source_sha256
            or source_size_bytes is not None and view.get("source_size_bytes") != source_size_bytes
            or (view.get("content_type"), view.get("suffix")) != FORMATS[info.format]):
        raise DomainError("snapshot_media_mismatch", 409)
    if (view["policy"] == MEDIA_VIEW_POLICY or require_current_compatible) and info.needs_working_view:
        raise DomainError("media_policy_upgrade_required", 409)
    return info


def _info(value):
    if isinstance(value, MediaInfo):
        return value.as_dict()
    return value


def _encoded_point(x, y, width, height, orientation):
    # Inverse EXIF transform on continuous pixel edges, not pixel-centre indices.
    return {1: lambda: (x, y), 2: lambda: (width - x, y),
            3: lambda: (width - x, height - y), 4: lambda: (x, height - y),
            5: lambda: (y, x), 6: lambda: (y, height - x),
            7: lambda: (width - y, height - x), 8: lambda: (width - y, x)}[orientation]()


def prepare_annotations(annotations: dict | None, info: MediaInfo | dict,
                        label_coordinate_space: str | None = None) -> dict | None:
    if label_coordinate_space not in (None, "encoded", "exif_display"):
        raise DomainError("label_coordinate_space_invalid", 422)
    if annotations is None:
        return None
    result = copy.deepcopy(annotations)
    if not isinstance(result, dict):
        raise DomainError("invalid_annotations", 422)
    details = _info(info)
    orientation = details["orientation"]
    objects = result.get("objects", [])
    if objects and orientation != 1 and label_coordinate_space is None:
        raise DomainError("label_coordinate_space_required", 422,
                          "Declare encoded or exif_display for annotations on oriented media.")
    if not objects or (orientation == 1 and label_coordinate_space is None):
        return result
    width, height = details["width"], details["height"]
    if label_coordinate_space == "exif_display":
        display_width, display_height = (height, width) if orientation >= 5 else (width, height)
        for obj in objects:
            try:
                x, y, w, h = obj["bbox"]
                if (any(type(value) not in (int, float) or not math.isfinite(value) for value in (x, y, w, h))
                        or min(x, y) < 0 or min(w, h) <= 0
                        or x + w > display_width or y + h > display_height):
                    raise ValueError("invalid display geometry")
                corners = [_encoded_point(a, b, width, height, orientation)
                           for a, b in ((x, y), (x + w, y), (x, y + h), (x + w, y + h))]
                left, top = min(p[0] for p in corners), min(p[1] for p in corners)
                right, bottom = max(p[0] for p in corners), max(p[1] for p in corners)
                converted_w, converted_h = right - left, bottom - top
                if left + converted_w > width:
                    converted_w = math.nextafter(converted_w, 0.0)
                if top + converted_h > height:
                    converted_h = math.nextafter(converted_h, 0.0)
                obj["bbox"] = [left, top, converted_w, converted_h]
                if "polygon" in obj:
                    for a, b in obj["polygon"]:
                        if (any(type(value) not in (int, float) or not math.isfinite(value) for value in (a, b))
                                or not 0 <= a <= display_width or not 0 <= b <= display_height):
                            raise ValueError("invalid display polygon")
                    obj["polygon"] = [list(_encoded_point(a, b, width, height, orientation))
                                      for a, b in obj["polygon"]]
            except (TypeError, ValueError, KeyError) as error:
                raise DomainError("invalid_bbox", 422) from error
    result["media_coordinate_provenance"] = {
        "policy": COORDINATE_POLICY, "source_sha256": details["source_sha256"],
        "source_coordinate_space": label_coordinate_space or "encoded",
        "coordinate_space": "encoded", "source_orientation": orientation,
        "encoded_width": width, "encoded_height": height,
    }
    return result


def prepare_yolo_objects(label_text: str, class_names: list[str], info: MediaInfo | dict,
                         *, key: str, label_coordinate_space: str | None = None,
                         clip: bool = False) -> tuple[list[dict], dict | None]:
    from .migration import _objects
    details = _info(info)
    width, height = details["width"], details["height"]
    if label_coordinate_space == "exif_display" and details["orientation"] in (5, 6, 7, 8):
        width, height = height, width
    objects = _objects(label_text, width, height, class_names, clip=clip, key=key)
    body = prepare_annotations({"objects": objects}, info, label_coordinate_space)
    return body["objects"], body.get("media_coordinate_provenance")


class HostedMedia:
    """Trusted internal storage helper; callers authorize source membership first."""
    def __init__(self, store, artifacts: ArtifactStore):
        self.store, self.artifacts = store, artifacts
        with store.transaction() as con:
            con.execute("CREATE TABLE IF NOT EXISTS hosted_media_views (source_sha256 TEXT NOT NULL,policy TEXT NOT NULL,view_json TEXT NOT NULL,PRIMARY KEY(source_sha256,policy))")
            con.execute("CREATE TABLE IF NOT EXISTS hosted_artifact_references (kind TEXT NOT NULL,owner_id TEXT NOT NULL,sha256 TEXT NOT NULL,size_bytes INTEGER NOT NULL,created_at REAL NOT NULL,PRIMARY KEY(kind,owner_id,sha256))")
            for verb in ("UPDATE", "DELETE"):
                con.execute(f"CREATE TRIGGER IF NOT EXISTS hosted_media_views_immutable_{verb.lower()} BEFORE {verb} ON hosted_media_views BEGIN SELECT RAISE(ABORT,'immutable media view'); END")

    @staticmethod
    def reference(view: dict) -> ArtifactRef:
        return ArtifactRef(view["sha256"], view["size_bytes"])

    def stage(self, source: bytes | BinaryIO | str | Path) -> dict:
        """Validate source and working bytes before membership, without refs.

        Admission callers hold artifacts.exclusive() through their membership
        transaction. A failed import leaves an ordinary collectible orphan;
        successful membership retains the source. resolve() lazily registers
        the immutable working representation on its first actual consumption.
        """
        original = _read(source)
        _, view = working_media_bytes(original)
        self.artifacts.put_bytes(original, expected_sha256=view["source_sha256"])
        return view

    def ingest(self, source: bytes | BinaryIO | str | Path) -> dict:
        original = _read(source)
        working, view = working_media_bytes(original)
        with self.artifacts.exclusive():
            self.artifacts.put_bytes(original, expected_sha256=view["source_sha256"])
            return self._retain(working, view)

    def _retain(self, working, view):
        reference = self.artifacts.put_bytes(working, expected_sha256=view["sha256"])
        encoded = json.dumps(view, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self.store.transaction() as con:
            existing = con.execute("SELECT view_json FROM hosted_media_views WHERE source_sha256=? AND policy=?",
                                   (view["source_sha256"], view["policy"])).fetchone()
            if existing:
                old = json.loads(existing[0])
                self.artifacts.verify(self.reference(old))
                return old
            con.execute("INSERT INTO hosted_media_views VALUES(?,?,?)", (view["source_sha256"], view["policy"], encoded))
            owner = view["source_sha256"] + ":" + view["policy"]
            for ref in (ArtifactRef(view["source_sha256"], view["source_size_bytes"]), reference):
                con.execute("INSERT OR IGNORE INTO hosted_artifact_references VALUES('media_view',?,?,?,?)",
                            (owner, ref.sha256, ref.size_bytes, self.store.clock()))
                if con.execute("SELECT 1 FROM sqlite_master WHERE name='hosted_orphan_observations'").fetchone():
                    con.execute("DELETE FROM hosted_orphan_observations WHERE sha256=?", (ref.sha256,))
        return view

    def resolve(self, source: ArtifactRef | str, *, policy=MEDIA_VIEW_POLICY) -> dict:
        digest = source.sha256 if isinstance(source, ArtifactRef) else source
        with self.artifacts.exclusive():
            self.artifacts.verify(source)
            with self.store.read_connection() as con:
                row = con.execute("SELECT view_json FROM hosted_media_views WHERE source_sha256=? AND policy=?", (digest, policy)).fetchone()
            if row:
                view = json.loads(row[0])
                self.artifacts.verify(self.reference(view))
                return view
            if policy != MEDIA_VIEW_POLICY:
                raise DomainError("media_policy_upgrade_required", 409)
            with self.artifacts.open(source) as stream:
                working, view = working_media_bytes(stream)
            return self._retain(working, view)

    def capture_views(self, images) -> dict:
        """Read immutable descriptors cheaply; pin/open verifies their bytes."""
        result, missing = {}, []
        with self.store.read_connection() as con:
            for image in images:
                row = con.execute("SELECT view_json FROM hosted_media_views WHERE source_sha256=? AND policy=?",
                                  (image['media_hash'], MEDIA_VIEW_POLICY)).fetchone()
                if row:
                    result[image['id']] = json.loads(row[0])
                else:
                    missing.append(image)
        for image in missing:
            result[image['id']] = self.resolve(image['media_hash'])
        return result

    def audit(self) -> list[dict]:
        """Read-only historical audit; never changes media or annotation state."""
        findings = []
        with self.store.read_connection() as con:
            rows = con.execute("SELECT i.id,i.media_hash,v.id AS version_id,v.annotations_json FROM hosted_images i JOIN hosted_annotation_versions v ON v.image_id=i.id ORDER BY i.id,v.id").fetchall()
        inspected = {}
        for row in rows:
            try:
                if row["media_hash"] not in inspected:
                    with self.artifacts.open(row["media_hash"]) as stream:
                        inspected[row["media_hash"]] = inspect_media(stream)
                info = inspected[row["media_hash"]]
                body = json.loads(row["annotations_json"])
                if info.orientation != 1 and body.get("objects") and not body.get("media_coordinate_provenance"):
                    findings.append({"image_id": row["id"], "version_id": row["version_id"], "source_sha256": info.source_sha256,
                                     "issue": "historical_label_coordinate_space_unknown", "orientation": info.orientation})
            except (DomainError, OSError, ValueError) as error:
                findings.append({"image_id": row["id"], "version_id": row["version_id"], "source_sha256": row["media_hash"],
                                 "issue": getattr(error, "code", "media_unavailable")})
        return findings
