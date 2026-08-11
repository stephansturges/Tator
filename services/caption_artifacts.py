"""Hash-keyed caption artifact storage.

This store is intentionally file-backed and append-only. The annotation UI can
load datasets from many places, but generated captions should be keyed by the
image bytes, not by a mutable frontend path or a Batch shard. JSONL is the
durable source of truth; any richer index can be rebuilt from these records.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
import zipfile
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CAPTION_ARTIFACT_STORE_FORMAT = "tator_caption_artifact_store_v1"
CAPTION_ARTIFACT_SET_MANIFEST_FORMAT = "tator_caption_set_manifest_v1"
CAPTION_ARTIFACT_EXPORT_FORMAT = "tator_caption_set_export_v1"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_sanitize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_sanitize(item) for item in value]
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(json_sanitize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any, *, prefix: str = "") -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}{digest}" if prefix else digest


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalize_set_id(value: Any) -> str:
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    raw = raw.strip("._-")
    return raw[:120] or f"caption_set_{uuid.uuid4().hex[:12]}"


def normalize_artifact_type(value: Any) -> str:
    raw = re.sub(r"[\s-]+", "_", str(value or "").strip().lower())
    if raw in {"base_caption", "caption", "caption0"}:
        return "base_caption"
    if raw in {"qa", "qa_pair", "generated_qa", "question_answer"}:
        return "qa_pair"
    return raw or "artifact"


def normalize_lifecycle_status(value: Any) -> str:
    raw = re.sub(r"[\s-]+", "_", str(value or "").strip().lower())
    return raw if raw in {"active", "superseded", "overflow", "rejected", "archived"} else "active"


class CaptionArtifactStore:
    """Append-only caption artifact store keyed by image SHA-256."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self._lock = RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_json_if_missing(
            self.root / "store.json",
            {
                "format": CAPTION_ARTIFACT_STORE_FORMAT,
                "created_at": utc_now(),
            },
        )

    def image_record_path(self, image_sha256: str) -> Path:
        digest = self._require_sha256(image_sha256)
        return self.root / "images" / digest[:2] / digest[2:4] / f"{digest}.json"

    def image_artifacts_path(self, image_sha256: str) -> Path:
        digest = self._require_sha256(image_sha256)
        return self.root / "artifacts" / "by_image" / digest[:2] / digest[2:4] / f"{digest}.jsonl"

    def prompt_context_path(self, prompt_context_hash: str) -> Path:
        return self.root / "prompt_contexts" / f"{self._require_hash(prompt_context_hash)}.json"

    def generation_spec_path(self, generation_spec_hash: str) -> Path:
        return self.root / "generation_specs" / f"{self._require_hash(generation_spec_hash)}.json"

    def attempt_path(self, attempt_id: str) -> Path:
        safe = normalize_set_id(attempt_id)
        return self.root / "attempts" / safe[:2] / f"{safe}.json"

    def set_manifest_path(self, caption_set_id: str) -> Path:
        return self.root / "sets" / normalize_set_id(caption_set_id) / "set.json"

    def set_events_path(self, caption_set_id: str) -> Path:
        return self.root / "sets" / normalize_set_id(caption_set_id) / "events.jsonl"

    def register_image(
        self,
        image_path: Path | str,
        *,
        aliases: Optional[Sequence[Mapping[str, Any]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        path = Path(image_path)
        digest = sha256_file(path)
        now = utc_now()
        aliases_list = [dict(alias) for alias in aliases or [] if isinstance(alias, Mapping)]
        metadata_payload = dict(metadata or {})
        record_path = self.image_record_path(digest)
        with self._lock:
            existing = self._read_json(record_path)
            if isinstance(existing, Mapping) and existing.get("image_sha256") == digest:
                merged_aliases = self._merge_aliases(existing.get("aliases"), aliases_list)
                merged_metadata = {
                    **(dict(existing.get("metadata") or {}) if isinstance(existing.get("metadata"), Mapping) else {}),
                    **metadata_payload,
                }
                updated = {
                    **dict(existing),
                    "aliases": merged_aliases,
                    "metadata": json_sanitize(merged_metadata),
                    "last_seen_at": now,
                }
                self._write_json(record_path, updated)
                return updated
            record = {
                "format": CAPTION_ARTIFACT_STORE_FORMAT,
                "image_sha256": digest,
                "created_at": now,
                "last_seen_at": now,
                "bytes": path.stat().st_size if path.exists() else None,
                "aliases": aliases_list,
                "metadata": json_sanitize(metadata_payload),
            }
            self._write_json(record_path, record)
            return record

    def ensure_prompt_context(self, payload: Mapping[str, Any]) -> str:
        clean_payload = json_sanitize(dict(payload or {}))
        context_hash = canonical_hash(clean_payload, prefix="ctx_")
        path = self.prompt_context_path(context_hash)
        with self._lock:
            self._write_json_if_missing(
                path,
                {
                    "format": CAPTION_ARTIFACT_STORE_FORMAT,
                    "prompt_context_hash": context_hash,
                    "created_at": utc_now(),
                    "payload": clean_payload,
                },
            )
        return context_hash

    def ensure_generation_spec(self, payload: Mapping[str, Any]) -> str:
        clean_payload = json_sanitize(dict(payload or {}))
        spec_hash = canonical_hash(clean_payload, prefix="spec_")
        path = self.generation_spec_path(spec_hash)
        with self._lock:
            self._write_json_if_missing(
                path,
                {
                    "format": CAPTION_ARTIFACT_STORE_FORMAT,
                    "generation_spec_hash": spec_hash,
                    "created_at": utc_now(),
                    "payload": clean_payload,
                },
            )
        return spec_hash

    def create_attempt(
        self,
        *,
        image_sha256: str,
        generation_spec_hash: str,
        prompt_context_hash: str,
        run_id: Optional[str] = None,
        provider: Optional[str] = None,
        status: str = "succeeded",
        transport: Optional[Mapping[str, Any]] = None,
        raw_output_paths: Optional[Sequence[str]] = None,
        usage: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        attempt_id = f"attempt_{uuid.uuid4().hex}"
        record = {
            "format": CAPTION_ARTIFACT_STORE_FORMAT,
            "attempt_id": attempt_id,
            "run_id": str(run_id or "").strip(),
            "image_sha256": self._require_sha256(image_sha256),
            "generation_spec_hash": self._require_hash(generation_spec_hash),
            "prompt_context_hash": self._require_hash(prompt_context_hash),
            "provider": str(provider or "").strip(),
            "status": str(status or "succeeded").strip(),
            "transport": json_sanitize(dict(transport or {})),
            "raw_output_paths": [str(path) for path in raw_output_paths or [] if str(path).strip()],
            "usage": json_sanitize(dict(usage or {})),
            "metadata": json_sanitize(dict(metadata or {})),
            "created_at": utc_now(),
        }
        with self._lock:
            self._write_json(self.attempt_path(attempt_id), record)
        return record

    def append_artifact(
        self,
        *,
        image_sha256: str,
        artifact_type: str,
        payload: Mapping[str, Any],
        generation_spec_hash: str,
        prompt_context_hash: str,
        attempt_id: Optional[str] = None,
        run_id: Optional[str] = None,
        source: Optional[str] = None,
        lifecycle_status: str = "active",
        metadata: Optional[Mapping[str, Any]] = None,
        caption_set_id: Optional[str] = None,
        set_role: Optional[str] = None,
    ) -> Dict[str, Any]:
        image_hash = self._require_sha256(image_sha256)
        artifact_kind = normalize_artifact_type(artifact_type)
        now = utc_now()
        artifact_id = f"artifact_{uuid.uuid4().hex}"
        payload_clean = json_sanitize(dict(payload or {}))
        artifact = {
            "format": CAPTION_ARTIFACT_STORE_FORMAT,
            "artifact_id": artifact_id,
            "image_sha256": image_hash,
            "artifact_type": artifact_kind,
            "payload": payload_clean,
            "generation_spec_hash": self._require_hash(generation_spec_hash),
            "prompt_context_hash": self._require_hash(prompt_context_hash),
            "attempt_id": str(attempt_id or "").strip(),
            "run_id": str(run_id or "").strip(),
            "source": str(source or "").strip(),
            "lifecycle_status": normalize_lifecycle_status(lifecycle_status),
            "metadata": json_sanitize(dict(metadata or {})),
            "created_at": now,
        }
        with self._lock:
            self._append_jsonl(self.image_artifacts_path(image_hash), artifact)
            if caption_set_id:
                self.ensure_caption_set(
                    caption_set_id=caption_set_id,
                    name=caption_set_id,
                    generation_spec_hash=generation_spec_hash,
                    prompt_context_hash=prompt_context_hash,
                )
                self.append_set_event(
                    caption_set_id,
                    "add_artifact",
                    image_sha256=image_hash,
                    artifact_id=artifact_id,
                    role=set_role or artifact_kind,
                    metadata={"source": source or "", "artifact_type": artifact_kind},
                )
        return artifact

    def ensure_caption_set(
        self,
        *,
        caption_set_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        generation_spec_hash: Optional[str] = None,
        prompt_context_hash: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        safe_id = normalize_set_id(caption_set_id)
        path = self.set_manifest_path(safe_id)
        with self._lock:
            existing = self._read_json(path)
            if isinstance(existing, Mapping) and existing.get("caption_set_id"):
                changed = False
                updated = dict(existing)
                if name and not str(updated.get("name") or "").strip():
                    updated["name"] = str(name)
                    changed = True
                if generation_spec_hash and not str(updated.get("generation_spec_hash") or "").strip():
                    updated["generation_spec_hash"] = str(generation_spec_hash)
                    changed = True
                if prompt_context_hash and not str(updated.get("prompt_context_hash") or "").strip():
                    updated["prompt_context_hash"] = str(prompt_context_hash)
                    changed = True
                if metadata:
                    updated["metadata"] = {
                        **(dict(updated.get("metadata") or {}) if isinstance(updated.get("metadata"), Mapping) else {}),
                        **json_sanitize(dict(metadata)),
                    }
                    changed = True
                if changed:
                    updated["updated_at"] = utc_now()
                    self._write_json(path, updated)
                return updated
            record = {
                "format": CAPTION_ARTIFACT_SET_MANIFEST_FORMAT,
                "caption_set_id": safe_id,
                "name": str(name or safe_id),
                "description": str(description or ""),
                "generation_spec_hash": str(generation_spec_hash or ""),
                "prompt_context_hash": str(prompt_context_hash or ""),
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "metadata": json_sanitize(dict(metadata or {})),
            }
            self._write_json(path, record)
            return record

    def append_set_event(
        self,
        caption_set_id: str,
        event_type: str,
        *,
        image_sha256: Optional[str] = None,
        artifact_id: Optional[str] = None,
        role: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        safe_id = normalize_set_id(caption_set_id)
        event = {
            "format": CAPTION_ARTIFACT_STORE_FORMAT,
            "event_id": f"event_{uuid.uuid4().hex}",
            "caption_set_id": safe_id,
            "event_type": str(event_type or "").strip() or "event",
            "image_sha256": self._require_sha256(image_sha256) if image_sha256 else "",
            "artifact_id": str(artifact_id or "").strip(),
            "role": str(role or "").strip(),
            "metadata": json_sanitize(dict(metadata or {})),
            "created_at": utc_now(),
        }
        with self._lock:
            self._append_jsonl(self.set_events_path(safe_id), event)
        return event

    def list_caption_sets(self) -> List[Dict[str, Any]]:
        sets_root = self.root / "sets"
        items: List[Dict[str, Any]] = []
        if not sets_root.exists():
            return []
        for manifest_path in sorted(sets_root.glob("*/set.json")):
            record = self._read_json(manifest_path)
            if not isinstance(record, Mapping):
                continue
            events_path = manifest_path.parent / "events.jsonl"
            event_count = sum(1 for _item in self._read_jsonl(events_path))
            items.append({**dict(record), "event_count": event_count})
        items.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
        return items

    def image_artifacts(self, image_sha256: str) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._read_jsonl(self.image_artifacts_path(image_sha256))]

    def caption_set_events(self, caption_set_id: str) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._read_jsonl(self.set_events_path(caption_set_id))]

    def caption_set_artifact_ids(self, caption_set_id: str) -> Optional[set[str]]:
        path = self.set_manifest_path(caption_set_id)
        if not path.exists():
            return set()
        active: set[str] = set()
        for event in self.caption_set_events(caption_set_id):
            artifact_id = str(event.get("artifact_id") or "").strip()
            if not artifact_id:
                continue
            event_type = str(event.get("event_type") or "").strip()
            if event_type == "add_artifact":
                active.add(artifact_id)
            elif event_type in {"archive_artifact", "remove_artifact"}:
                active.discard(artifact_id)
        return active

    def summarize_images(
        self,
        image_sha256_values: Sequence[str],
        *,
        caption_set_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        set_artifact_ids = self.caption_set_artifact_ids(caption_set_id) if caption_set_id else None
        images: List[Dict[str, Any]] = []
        totals = {
            "image_count": 0,
            "base_caption_count": 0,
            "qa_pair_count": 0,
            "artifact_count": 0,
        }
        for raw_hash in image_sha256_values:
            try:
                image_hash = self._require_sha256(raw_hash)
            except ValueError:
                continue
            artifacts = [
                artifact
                for artifact in self.image_artifacts(image_hash)
                if normalize_lifecycle_status(artifact.get("lifecycle_status")) == "active"
                and (set_artifact_ids is None or str(artifact.get("artifact_id") or "") in set_artifact_ids)
            ]
            base_count = sum(1 for artifact in artifacts if artifact.get("artifact_type") == "base_caption")
            qa_count = sum(1 for artifact in artifacts if artifact.get("artifact_type") == "qa_pair")
            image_record = self._read_json(self.image_record_path(image_hash))
            images.append(
                {
                    "image_sha256": image_hash,
                    "registered": isinstance(image_record, Mapping),
                    "base_caption_count": base_count,
                    "qa_pair_count": qa_count,
                    "artifact_count": len(artifacts),
                    "aliases": list(image_record.get("aliases") or []) if isinstance(image_record, Mapping) else [],
                    "artifacts": artifacts[-50:],
                    "artifact_sample_limit": 50,
                }
            )
            totals["image_count"] += 1
            totals["base_caption_count"] += base_count
            totals["qa_pair_count"] += qa_count
            totals["artifact_count"] += len(artifacts)
        return {
            "format": CAPTION_ARTIFACT_STORE_FORMAT,
            "caption_set_id": normalize_set_id(caption_set_id) if caption_set_id else "",
            "totals": totals,
            "images": images,
        }

    def export_caption_set(
        self,
        *,
        caption_set_id: str,
        image_sha256_values: Optional[Sequence[str]] = None,
        output_dir: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        safe_id = normalize_set_id(caption_set_id)
        manifest = self._read_json(self.set_manifest_path(safe_id))
        if not isinstance(manifest, Mapping):
            raise FileNotFoundError(f"caption set not found: {safe_id}")
        if image_sha256_values is None:
            image_sha256_values = sorted(
                {
                    str(event.get("image_sha256") or "").strip()
                    for event in self.caption_set_events(safe_id)
                    if str(event.get("image_sha256") or "").strip()
                }
            )
        summary = self.summarize_images(list(image_sha256_values), caption_set_id=safe_id)
        export_id = f"caption_export_{uuid.uuid4().hex[:12]}"
        export_dir = Path(output_dir) if output_dir else self.root / "exports" / export_id
        export_dir.mkdir(parents=True, exist_ok=True)
        manifest_out = {
            "format": CAPTION_ARTIFACT_EXPORT_FORMAT,
            "export_id": export_id,
            "caption_set": manifest,
            "created_at": utc_now(),
            "summary": summary.get("totals"),
        }
        self._write_json(export_dir / "caption_set_manifest.json", manifest_out)
        self._write_json(export_dir / "caption_set_summary.json", summary)
        artifacts_path = export_dir / "caption_artifacts.jsonl"
        with artifacts_path.open("w", encoding="utf-8") as handle:
            for image in summary.get("images") or []:
                for artifact in image.get("artifacts") or []:
                    handle.write(json.dumps(artifact, ensure_ascii=False, sort_keys=True) + "\n")
        zip_path = export_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(export_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(export_dir).as_posix())
        return {
            "format": CAPTION_ARTIFACT_EXPORT_FORMAT,
            "export_id": export_id,
            "caption_set_id": safe_id,
            "export_dir": str(export_dir),
            "zip_path": str(zip_path),
            "summary": summary.get("totals"),
        }

    @staticmethod
    def _require_sha256(value: Any) -> str:
        digest = str(value or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("image_sha256_required")
        return digest

    @staticmethod
    def _require_hash(value: Any) -> str:
        raw = str(value or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{8,128}", raw):
            raise ValueError("hash_required")
        return raw

    @staticmethod
    def _merge_aliases(existing: Any, new_aliases: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        aliases: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for alias in [
            *(existing if isinstance(existing, list) else []),
            *new_aliases,
        ]:
            if not isinstance(alias, Mapping):
                continue
            clean = json_sanitize(dict(alias))
            key = canonical_json(clean)
            if key in seen:
                continue
            aliases.append(clean)
            seen.add(key)
        return aliases

    def _read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            if not path.exists():
                return None
            loaded = json.loads(path.read_text(encoding="utf-8"))
            return dict(loaded) if isinstance(loaded, Mapping) else None
        except Exception:
            return None

    def _read_jsonl(self, path: Path) -> Iterable[Dict[str, Any]]:
        try:
            if not path.exists():
                return []
            rows: List[Dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        loaded = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(loaded, Mapping):
                        rows.append(dict(loaded))
            return rows
        except Exception:
            return []

    def _write_json_if_missing(self, path: Path, payload: Mapping[str, Any]) -> None:
        if path.exists():
            return
        self._write_json(path, payload)

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(json_sanitize(dict(payload)), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_sanitize(dict(payload)), ensure_ascii=False, sort_keys=True) + "\n")
