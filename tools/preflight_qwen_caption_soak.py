#!/usr/bin/env python3
"""Preflight a long Qwen caption soak before leaving it unattended."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.qwen_mlx import (  # noqa: E402
    QWEN_MLX_CAPTION_DEFAULT_MODEL,
    QWEN_MLX_MODEL_IDS,
)
from services.qwen_model_catalog import (  # noqa: E402
    QWEN_TRANSFORMERS_MODEL_IDS,
    is_qwen_mlx_model_id,
)
from tools import audit_qwen_caption_soak as audit  # noqa: E402
from tools import run_qwen_caption_flow_benchmark as runner  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "tmp" / "qwen_caption_benchmark" / "dataset_soak"
DEFAULT_MIN_FREE_GB = 5.0
DEFAULT_DISK_SAFETY_FACTOR = 1.25
STRUCTURED_BYTES_PER_ATTEMPT = 262_144
STRUCTURED_BYTES_PER_CASE = 32_768
FIXED_ARTIFACT_OVERHEAD_BYTES = 50 * 1024 * 1024
DIAGNOSTIC_TEXT_FILES_PER_ATTEMPT = 4
STATUS_RANK = {"ok": 0, "warn": 1, "error": 2}
UNSAFE_ARTIFACT_AUDIT_ERROR_CHECKS = {
    "manifest",
    "results_jsonl",
    "captions_jsonl",
    "caption_coverage",
}
WEIGHT_FILE_PATTERNS = (
    "*.safetensors",
    "*.bin",
    "*.gguf",
    "*.npz",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
IMAGE_SPECIFIC_REQUEST_KEYS = {
    "image_base64",
    "image_token",
    "image_name",
    "label_hints",
    "image_width",
    "image_height",
}
REQUEST_MODEL_FIELD_MAP = {
    "model_id": "model_id",
    "model_variant": "model_variant",
    "refinement_model_id": "refinement_model_id",
    "caption_fallback_model_id": "fallback_model_id",
    "caption_loop_recovery_mode": "loop_recovery",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add_check(report: dict[str, Any], name: str, status: str, detail: str, **fields: Any) -> None:
    check = {"name": name, "status": status, "detail": detail, **fields}
    report.setdefault("checks", []).append(check)
    if STATUS_RANK.get(status, 0) > STATUS_RANK.get(str(report.get("status") or "ok"), 0):
        report["status"] = status


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _request_template_path(args: argparse.Namespace) -> Path | None:
    raw = getattr(args, "request_json", None)
    if not raw:
        return None
    return Path(raw)


def _load_request_template(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _request_template_path(args)
    if path is None:
        return {}, {"provided": False, "status": "ok", "applied_keys": []}
    meta: dict[str, Any] = {
        "provided": True,
        "path": str(path),
        "status": "ok",
        "applied_keys": [],
        "ignored_image_keys": [],
    }
    if not path.exists():
        meta.update({"status": "error", "detail": f"request template not found: {path}"})
        return {}, meta
    try:
        loaded = _read_json(path)
    except Exception as exc:  # noqa: BLE001
        meta.update({
            "status": "error",
            "detail": f"request template is not valid JSON: {exc}",
            "error_type": type(exc).__name__,
        })
        return {}, meta
    if not isinstance(loaded, Mapping):
        meta.update({"status": "error", "detail": "request template must be a JSON object"})
        return {}, meta
    template = dict(loaded)
    ignored = sorted(key for key in IMAGE_SPECIFIC_REQUEST_KEYS if key in template)
    for key in ignored:
        template.pop(key, None)
    meta["ignored_image_keys"] = ignored
    meta["applied_keys"] = sorted(template.keys())
    return template, meta


def _pid_is_alive(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except (TypeError, ValueError, OverflowError):
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _normalized_runner_capabilities(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        capability = str(item or "").strip()
        if capability and capability not in seen:
            seen.add(capability)
            normalized.append(capability)
    return normalized


def _runner_capability_state(output_dir: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    lock_capabilities = _normalized_runner_capabilities(lock.get("runner_capabilities"))
    heartbeat_capabilities: list[str] = []
    heartbeat_path = output_dir / "heartbeat.json"
    heartbeat: Mapping[str, Any] = {}
    if heartbeat_path.exists():
        try:
            loaded = _read_json(heartbeat_path)
        except Exception:
            loaded = {}
        if isinstance(loaded, Mapping):
            heartbeat = loaded
            heartbeat_capabilities = _normalized_runner_capabilities(heartbeat.get("runner_capabilities"))
    capabilities = lock_capabilities or heartbeat_capabilities
    sources = []
    if lock_capabilities:
        sources.append("runner_lock")
    if heartbeat_capabilities:
        sources.append("heartbeat")
    return {
        "runner_capabilities": capabilities,
        "runner_supports_graceful_restart": runner.RUNNER_CAPABILITY_GRACEFUL_RESTART in set(capabilities),
        "runner_capability_sources": sources,
        "lock_runner_capabilities": lock_capabilities,
        "heartbeat_runner_capabilities": heartbeat_capabilities,
        "heartbeat_status": str(heartbeat.get("status") or "").strip() if heartbeat else "",
        "heartbeat_phase": str(heartbeat.get("phase") or "").strip() if heartbeat else "",
    }


def _existing_parent(path: Path) -> Path:
    current = path.resolve()
    while not current.exists() and current.parent != current:
        current = current.parent
    return current if current.exists() else Path.cwd()


def _load_requested_cases(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_root = args.dataset_root.resolve()
    meta: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "source": "cases_json" if args.cases_json else "dataset",
    }
    if args.cases_json:
        loaded = _read_json(args.cases_json)
        if not isinstance(loaded, list):
            raise ValueError("cases_json_must_be_list")
        cases = [dict(case) for case in loaded if isinstance(case, dict)]
    else:
        items = runner.discover_items(dataset_root)
        meta["discovered_images"] = len(items)
        cases = (
            runner.select_all_image_cases(items, caption_mode=args.caption_mode)
            if args.all_images
            else runner.select_cases(items)
        )
    if args.sample_size and args.sample_size > 0:
        cases, sample_meta = runner.sample_cases_with_meta(
            cases,
            sample_size=int(args.sample_size),
            sample_seed=int(args.sample_seed),
        )
        meta["sample_selection"] = sample_meta
    if args.case:
        wanted = {str(item) for item in args.case}
        cases = [
            case
            for case in cases
            if case.get("name") in wanted
            or case.get("stem") in wanted
            or runner.case_key(case) in wanted
        ]
    if args.limit:
        cases = cases[: int(args.limit)]
    meta["requested_cases"] = len(cases)
    return cases, meta


def _completed_case_keys(latest_rows: Mapping[str, Mapping[str, Any]]) -> set[str]:
    completed: set[str] = set()
    for key, row in latest_rows.items():
        final_status = str(row.get("final_status") or row.get("status") or "").strip()
        if final_status in {"ok", "skipped_completed", "skipped_existing_caption"}:
            completed.add(key)
    return completed


def _estimate_required_bytes(
    *,
    selected_cases: int,
    remaining_cases: int,
    attempts: int,
    max_artifact_log_bytes: int,
) -> dict[str, Any]:
    attempts = max(1, int(attempts or 1))
    remaining_cases = max(0, int(remaining_cases or 0))
    selected_cases = max(0, int(selected_cases or 0))
    max_artifact_log_bytes = runner.normalize_artifact_log_bytes(max_artifact_log_bytes)
    structured_bytes = (
        remaining_cases * attempts * STRUCTURED_BYTES_PER_ATTEMPT
        + selected_cases * STRUCTURED_BYTES_PER_CASE
        + FIXED_ARTIFACT_OVERHEAD_BYTES
    )
    if max_artifact_log_bytes <= 0:
        return {
            "bounded": False,
            "estimated_bytes": None,
            "structured_bytes": structured_bytes,
            "diagnostic_bytes": None,
            "diagnostic_files_per_attempt": DIAGNOSTIC_TEXT_FILES_PER_ATTEMPT,
        }
    diagnostic_bytes = (
        remaining_cases
        * attempts
        * DIAGNOSTIC_TEXT_FILES_PER_ATTEMPT
        * max_artifact_log_bytes
    )
    return {
        "bounded": True,
        "estimated_bytes": structured_bytes + diagnostic_bytes,
        "structured_bytes": structured_bytes,
        "diagnostic_bytes": diagnostic_bytes,
        "diagnostic_files_per_attempt": DIAGNOSTIC_TEXT_FILES_PER_ATTEMPT,
    }


def _format_bytes(value: Any) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError, OverflowError):
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TiB"


def _manifest_case_keys(manifest: Mapping[str, Any]) -> set[str]:
    return {
        runner.case_key(case)
        for case in (manifest.get("cases") or [])
        if isinstance(case, Mapping)
    }


def _probe_directory_writable(path: Path, *, label: str) -> dict[str, Any]:
    probe_path: Path | None = None
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe_path = path / f".tator_qwen_caption_write_probe_{os.getpid()}_{time.time_ns()}"
        probe_path.write_text("ok\n")
        probe_path.unlink()
        return {
            "status": "ok",
            "detail": f"{label} is writable",
            "path": str(path),
        }
    except Exception as exc:  # noqa: BLE001
        if probe_path is not None:
            try:
                probe_path.unlink()
            except Exception:
                pass
        return {
            "status": "error",
            "detail": f"{label} is not writable: {exc}",
            "path": str(path),
            "error_type": type(exc).__name__,
        }


def _text_label_target_dirs(dataset_root: Path, cases: Sequence[Mapping[str, Any]]) -> list[Path]:
    dirs: set[Path] = set()
    for case in cases:
        image_path_raw = str(case.get("image_path") or "").strip()
        if not image_path_raw:
            continue
        dirs.add(runner.dataset_text_label_path(dataset_root, Path(image_path_raw)).parent)
    return sorted(dirs, key=lambda item: str(item))


def _arg_present(args: argparse.Namespace, name: str) -> bool:
    return hasattr(args, name)


def _arg_str(args: argparse.Namespace, name: str, default: str = "") -> str:
    return str(getattr(args, name, default) or "").strip()


def _hf_cache_repo_path(model_id: str) -> Path | None:
    raw = str(model_id or "").strip()
    if not raw or raw.startswith(("http://", "https://")):
        return None
    candidate = Path(raw).expanduser()
    if candidate.exists():
        return candidate.resolve()
    cache_root = (
        os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("HF_HUB_CACHE")
        or os.environ.get("TRANSFORMERS_CACHE")
    )
    if cache_root:
        hub_dir = Path(cache_root).expanduser()
    else:
        hub_dir = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    return hub_dir / f"models--{raw.replace('/', '--')}"


def _hf_cache_snapshot_path(model_id: str) -> Path | None:
    raw = str(model_id or "").strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if candidate.exists():
        return candidate.resolve()
    repo_path = _hf_cache_repo_path(raw)
    if repo_path and repo_path.exists():
        for ref_name in ("main", "master"):
            ref_path = repo_path / "refs" / ref_name
            try:
                commit = ref_path.read_text(encoding="utf-8").strip()
            except Exception:
                commit = ""
            if commit:
                snapshot_path = repo_path / "snapshots" / commit
                if snapshot_path.exists():
                    return snapshot_path.resolve()
        snapshots_dir = repo_path / "snapshots"
        if snapshots_dir.exists():
            snapshot_dirs = [path for path in snapshots_dir.iterdir() if path.is_dir()]
            if snapshot_dirs:
                return max(snapshot_dirs, key=lambda path: path.stat().st_mtime).resolve()
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return None
    for filename in ("config.json", "tokenizer_config.json", "preprocessor_config.json"):
        try:
            cached = try_to_load_from_cache(raw, filename)
        except Exception:
            cached = None
        if isinstance(cached, str) and cached and os.path.exists(cached):
            cached_path = Path(cached).resolve()
            return cached_path.parent if "snapshots" in cached_path.parts else cached_path.parent
    return None


def _snapshot_has_weights(snapshot_path: Path | None) -> bool:
    if snapshot_path is None or not snapshot_path.exists():
        return False
    if snapshot_path.is_file():
        return snapshot_path.suffix.lower() in {".safetensors", ".bin", ".gguf", ".npz"}
    if not snapshot_path.is_dir():
        return False
    for pattern in WEIGHT_FILE_PATTERNS:
        try:
            if any(snapshot_path.glob(pattern)):
                return True
        except Exception:
            continue
    return False


def _model_cache_state(model_id: str) -> dict[str, Any]:
    raw = str(model_id or "").strip()
    repo_path = _hf_cache_repo_path(raw)
    snapshot_path = _hf_cache_snapshot_path(raw)
    has_weights = _snapshot_has_weights(snapshot_path)
    repo_exists = bool(repo_path and repo_path.exists())
    snapshot_exists = bool(snapshot_path and snapshot_path.exists())
    partial = bool((repo_exists or snapshot_exists) and not has_weights)
    return {
        "model_id": raw,
        "cache_repo": str(repo_path) if repo_path else None,
        "cache_snapshot": str(snapshot_path) if snapshot_path else None,
        "local": bool(has_weights),
        "partial": partial,
        "needs_download": not bool(has_weights),
    }


def _model_id_is_thinking(model_id: str) -> bool:
    return "thinking" in str(model_id or "").lower()


def _instruct_variant_model_id(model_id: str) -> str:
    return str(model_id or "").replace("Thinking", "Instruct").replace("thinking", "Instruct")


def _known_model_ids(model_id: str) -> set[str]:
    if is_qwen_mlx_model_id(model_id):
        return {str(item) for item in QWEN_MLX_MODEL_IDS}
    return {str(item) for item in QWEN_TRANSFORMERS_MODEL_IDS}


def _default_caption_refinement_model_id(desired_model_id: str) -> str:
    desired = str(desired_model_id or "").strip() or runner.DEFAULT_MODEL
    if not _model_id_is_thinking(desired):
        return desired
    exact_instruct = _instruct_variant_model_id(desired)
    known_ids = _known_model_ids(desired)
    if exact_instruct in known_ids:
        exact_state = _model_cache_state(exact_instruct)
        if exact_state.get("local") and not exact_state.get("partial"):
            return exact_instruct
    if is_qwen_mlx_model_id(desired):
        configured = (
            os.environ.get("QWEN_MLX_CAPTION_MODEL_NAME", QWEN_MLX_CAPTION_DEFAULT_MODEL).strip()
            or QWEN_MLX_CAPTION_DEFAULT_MODEL
        )
        return _instruct_variant_model_id(configured)
    small_instruct = "Qwen/Qwen3-VL-4B-Instruct"
    small_state = _model_cache_state(small_instruct)
    if small_state.get("local") and not small_state.get("partial"):
        return small_instruct
    if exact_instruct in known_ids:
        return exact_instruct
    return small_instruct


def _caption_default_model_id() -> str:
    return (
        os.environ.get("QWEN_MLX_CAPTION_MODEL_NAME", QWEN_MLX_CAPTION_DEFAULT_MODEL).strip()
        or QWEN_MLX_CAPTION_DEFAULT_MODEL
    )


def _resolve_primary_model_id(args: argparse.Namespace) -> str | None:
    raw = _arg_str(args, "model_id")
    lowered = raw.lower()
    if lowered in {"", "default"}:
        raw = runner.DEFAULT_MODEL
    elif lowered in {"active", "auto"}:
        return None
    variant = _arg_str(args, "model_variant", "auto").lower()
    if variant in {"instruct", "thinking"} and "qwen3-vl" in raw.lower():
        target = "Instruct" if variant == "instruct" else "Thinking"
        raw = raw.replace("Instruct", target).replace("Thinking", target)
    return raw


def _effective_caption_model_args(
    args: argparse.Namespace,
    request_template: Mapping[str, Any] | None = None,
) -> tuple[argparse.Namespace, dict[str, Any]]:
    data = {
        "preview_only": bool(getattr(args, "preview_only", False)),
        "allow_model_download": bool(getattr(args, "allow_model_download", False)),
        "model_id": _arg_str(args, "model_id", runner.DEFAULT_MODEL),
        "model_variant": _arg_str(args, "model_variant", "Instruct") or "Instruct",
        "refinement_model_id": _arg_str(args, "refinement_model_id", "same") or "same",
        "fallback_model_id": _arg_str(args, "fallback_model_id", "auto") or "auto",
        "loop_recovery": _arg_str(args, "loop_recovery", "safe_retry_fallback") or "safe_retry_fallback",
    }
    applied: dict[str, str] = {}
    for template_key, attr_name in REQUEST_MODEL_FIELD_MAP.items():
        if request_template is None or template_key not in request_template:
            continue
        raw_value = request_template.get(template_key)
        if attr_name == "model_id":
            raw = str(raw_value or "").strip()
            if raw.lower() in {"", "active", "auto", "default"}:
                data[attr_name] = _caption_default_model_id()
                applied[template_key] = f"{data[attr_name]} (resolved app caption default)"
            else:
                data[attr_name] = raw
                applied[template_key] = raw
            continue
        raw = str(raw_value or "").strip()
        if attr_name == "model_variant":
            data[attr_name] = raw if raw in {"auto", "Instruct", "Thinking"} else "auto"
        elif attr_name == "loop_recovery":
            data[attr_name] = raw if raw in {"off", "safe_retry", "safe_retry_fallback"} else "safe_retry_fallback"
        elif attr_name == "refinement_model_id":
            data[attr_name] = raw or "same"
        elif attr_name == "fallback_model_id":
            data[attr_name] = raw or "auto"
        applied[template_key] = str(data[attr_name])
    meta = {"request_model_overrides": applied}
    return argparse.Namespace(**data), meta


def _resolve_secondary_model_id(
    raw: str,
    *,
    role: str,
    primary_model_id: str,
    loop_recovery: str,
) -> str | None:
    lowered = str(raw or "").strip().lower()
    if role == "refinement":
        if lowered in {"", "auto"}:
            return _default_caption_refinement_model_id(primary_model_id)
        if lowered == "same":
            return primary_model_id
    if role == "fallback":
        if loop_recovery != "safe_retry_fallback":
            return None
        if lowered in {"", "auto"}:
            return _default_caption_refinement_model_id(primary_model_id)
        if lowered in {"none", "off"}:
            return None
        if lowered == "same":
            return primary_model_id
    if lowered in {"active", "default"}:
        return None
    return str(raw or "").strip() or None


def _caption_model_roles(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[str]]:
    if bool(getattr(args, "preview_only", False)):
        return [], []
    if not _arg_present(args, "model_id"):
        return [], []
    unresolved: list[str] = []
    primary_model_id = _resolve_primary_model_id(args)
    if not primary_model_id:
        unresolved.append("caption model uses active/auto selection that preflight cannot resolve without loading backend state")
        return [], unresolved
    roles = [{"role": "caption", "model_id": primary_model_id}]
    if _arg_present(args, "refinement_model_id"):
        refinement = _resolve_secondary_model_id(
            _arg_str(args, "refinement_model_id", "same"),
            role="refinement",
            primary_model_id=primary_model_id,
            loop_recovery=_arg_str(args, "loop_recovery", "safe_retry_fallback"),
        )
        if refinement:
            roles.append({"role": "refinement", "model_id": refinement})
        else:
            unresolved.append("refinement model uses active/default selection that preflight cannot resolve")
    if _arg_present(args, "fallback_model_id"):
        fallback = _resolve_secondary_model_id(
            _arg_str(args, "fallback_model_id", "auto"),
            role="fallback",
            primary_model_id=primary_model_id,
            loop_recovery=_arg_str(args, "loop_recovery", "safe_retry_fallback"),
        )
        if fallback:
            roles.append({"role": "fallback", "model_id": fallback})
        elif _arg_str(args, "fallback_model_id", "auto").lower() in {"active", "default"}:
            unresolved.append("fallback model uses active/default selection that preflight cannot resolve")
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for role in roles:
        key = (role["role"], role["model_id"])
        if key not in seen:
            seen.add(key)
            deduped.append(role)
    return deduped, unresolved


def _add_model_cache_check(
    report: dict[str, Any],
    args: argparse.Namespace,
    request_template: Mapping[str, Any] | None = None,
) -> None:
    effective_args, effective_meta = _effective_caption_model_args(args, request_template)
    if bool(getattr(args, "preview_only", False)):
        report["model_cache"] = {
            "preview_only": True,
            "models": [],
            "unresolved": [],
            "allow_model_download": bool(getattr(args, "allow_model_download", False)),
            **effective_meta,
        }
        _add_check(report, "model_cache", "ok", "preview-only run does not require generation model cache")
        return
    roles, unresolved = _caption_model_roles(effective_args)
    allow_download = bool(getattr(effective_args, "allow_model_download", False))
    models: list[dict[str, Any]] = []
    for role in roles:
        state = _model_cache_state(role["model_id"])
        models.append({**role, **state})
    missing = [item for item in models if item.get("needs_download")]
    partial = [item for item in models if item.get("partial")]
    report["model_cache"] = {
        "preview_only": False,
        "models": models,
        "unresolved": unresolved,
        "allow_model_download": allow_download,
        **effective_meta,
    }
    if partial and not allow_download:
        detail = "model cache is partial or missing weights: " + ", ".join(item["model_id"] for item in partial)
        _add_check(report, "model_cache", "error", detail, models=models, unresolved=unresolved)
    elif missing and not allow_download:
        detail = "model cache is missing; download or choose a local model: " + ", ".join(
            item["model_id"] for item in missing
        )
        _add_check(report, "model_cache", "error", detail, models=models, unresolved=unresolved)
    elif missing:
        detail = "model cache is missing, but downloads were explicitly allowed: " + ", ".join(
            item["model_id"] for item in missing
        )
        _add_check(report, "model_cache", "warn", detail, models=models, unresolved=unresolved)
    elif unresolved:
        _add_check(
            report,
            "model_cache",
            "warn",
            "; ".join(unresolved),
            models=models,
            unresolved=unresolved,
        )
    elif models:
        _add_check(
            report,
            "model_cache",
            "ok",
            "all selected concrete caption models have local weight files",
            models=models,
            unresolved=unresolved,
        )
    else:
        _add_check(report, "model_cache", "ok", "no concrete model selection provided to preflight")


def _artifact_audit_has_unsafe_errors(audit_report: Mapping[str, Any]) -> bool:
    checks = audit_report.get("checks")
    if not isinstance(checks, list):
        return True
    for check in checks:
        if not isinstance(check, Mapping):
            return True
        if str(check.get("status") or "").lower() != "error":
            continue
        if str(check.get("name") or "") in UNSAFE_ARTIFACT_AUDIT_ERROR_CHECKS:
            return True
    return False


def _artifact_audit_preflight_status(
    audit_report: Mapping[str, Any],
    *,
    resume_requested: bool,
) -> tuple[str, str, dict[str, Any]]:
    audit_status = str(audit_report.get("status") or "error").lower()
    if audit_status not in {"ok", "warn", "error"}:
        return "error", f"existing artifact audit status is {audit_status}", {
            "audit_status": audit_status,
            "recoverable_interrupted_state": False,
        }
    fields: dict[str, Any] = {
        "audit_status": audit_status,
        "recoverable_interrupted_state": False,
    }
    if audit_status != "error":
        return audit_status, f"existing artifact audit status is {audit_status}", fields
    if resume_requested and not _artifact_audit_has_unsafe_errors(audit_report):
        fields["recoverable_interrupted_state"] = True
        return (
            "warn",
            "existing artifact audit status is error, but the errors are recoverable by resume",
            fields,
        )
    return "error", "existing artifact audit status is error", fields


def preflight_soak(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    report: dict[str, Any] = {
        "status": "ok",
        "checked_at": _now_iso(),
        "output_dir": str(output_dir),
        "checks": [],
    }
    request_template, request_template_meta = _load_request_template(args)
    report["request_template"] = request_template_meta
    run_settings: dict[str, Any] | None = None
    if request_template_meta.get("provided"):
        template_status = str(request_template_meta.get("status") or "ok")
        if template_status == "error":
            _add_check(
                report,
                "request_template",
                "error",
                str(request_template_meta.get("detail") or "invalid request template"),
                path=request_template_meta.get("path"),
                error_type=request_template_meta.get("error_type"),
            )
        else:
            applied = request_template_meta.get("applied_keys") or []
            _add_check(
                report,
                "request_template",
                "ok",
                f"request template loaded with {len(applied)} applied keys",
                path=request_template_meta.get("path"),
                applied_keys=applied,
                ignored_image_keys=request_template_meta.get("ignored_image_keys") or [],
            )
    if str(request_template_meta.get("status") or "ok") != "error":
        run_settings = runner.run_settings_payload(args, request_template=request_template)
        report["run_settings"] = run_settings
    try:
        cases, case_meta = _load_requested_cases(args)
    except Exception as exc:  # noqa: BLE001
        _add_check(report, "case_selection", "error", str(exc), error_type=type(exc).__name__)
        return report
    selected_keys = {runner.case_key(case) for case in cases}
    if cases:
        _add_check(report, "case_selection", "ok", f"{len(cases)} requested cases", **case_meta)
    else:
        _add_check(report, "case_selection", "error", "no caption cases selected", **case_meta)

    results_path = output_dir / "results.jsonl"
    captions_path = output_dir / "captions.jsonl"
    invalid_resume_rows: list[Mapping[str, Any]] = []
    invalid_caption_rows: list[Mapping[str, Any]] = []
    caption_rows = 0
    try:
        latest_rows = runner.load_latest_rows(results_path)
    except runner.ResultsJsonlError as exc:
        latest_rows = {}
        invalid_resume_rows = exc.errors
    try:
        caption_rows = len(runner.validate_captions_jsonl(captions_path))
    except runner.CaptionsJsonlError as exc:
        invalid_caption_rows = exc.errors
    completed_keys = _completed_case_keys(latest_rows)
    remaining_keys = selected_keys - completed_keys
    failed_latest = [
        key
        for key, row in latest_rows.items()
        if key in selected_keys and str(row.get("final_status") or row.get("status") or "") not in {"ok", "skipped_completed", "skipped_existing_caption"}
    ]
    report["resume"] = {
        "resume_requested": bool(args.resume),
        "latest_rows": len(latest_rows),
        "completed_cases": len(completed_keys & selected_keys),
        "failed_latest_cases": len(failed_latest),
        "remaining_cases": len(remaining_keys),
    }
    if invalid_resume_rows:
        _add_check(
            report,
            "resume_rows",
            "error",
            f"results.jsonl has {len(invalid_resume_rows)} invalid row(s)",
            invalid_rows=invalid_resume_rows[:10],
        )
    elif latest_rows:
        _add_check(
            report,
            "resume_rows",
            "ok",
            f"{len(completed_keys & selected_keys)} completed, {len(failed_latest)} failed latest, {len(remaining_keys)} remaining",
        )
    elif args.resume:
        _add_check(report, "resume_rows", "ok", "resume requested; no prior rows, run will start from zero")
    else:
        _add_check(report, "resume_rows", "ok", "fresh run requested")
    if invalid_caption_rows:
        _add_check(
            report,
            "caption_rows",
            "error",
            f"captions.jsonl has {len(invalid_caption_rows)} invalid row(s)",
            invalid_rows=invalid_caption_rows[:10],
        )
    elif captions_path.exists():
        _add_check(report, "caption_rows", "ok", f"{caption_rows} caption rows")
    elif args.resume:
        _add_check(report, "caption_rows", "ok", "resume requested; no prior caption rows")

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = _read_json(manifest_path)
            if not isinstance(manifest, Mapping):
                raise ValueError("manifest_is_not_object")
            manifest_keys = _manifest_case_keys(manifest)
            if selected_keys and manifest_keys and selected_keys != manifest_keys:
                _add_check(
                    report,
                    "resume_manifest",
                    "error",
                    "existing manifest case set does not match requested case set",
                    existing_cases=len(manifest_keys),
                    requested_cases=len(selected_keys),
                )
            else:
                _add_check(report, "resume_manifest", "ok", f"existing manifest has {len(manifest_keys)} cases")
            if run_settings is not None:
                settings_status, settings_detail = runner.manifest_run_settings_status(manifest, run_settings)
                check_fields: dict[str, Any] = {
                    "requested_fingerprint": run_settings.get("fingerprint"),
                }
                previous_settings = (
                    manifest.get("run_settings")
                    if isinstance(manifest.get("run_settings"), Mapping)
                    else None
                )
                if isinstance(previous_settings, Mapping):
                    check_fields["existing_fingerprint"] = previous_settings.get("fingerprint")
                if settings_status == "mismatch":
                    _add_check(report, "resume_settings", "error", settings_detail, **check_fields)
                elif settings_status == "legacy":
                    _add_check(report, "resume_settings", "warn", settings_detail, **check_fields)
                else:
                    _add_check(report, "resume_settings", "ok", settings_detail, **check_fields)
        except Exception as exc:  # noqa: BLE001
            _add_check(report, "resume_manifest", "error", f"invalid manifest: {exc}")
    elif args.resume and latest_rows and args.cases_json:
        _add_check(
            report,
            "resume_manifest",
            "warn",
            "results exist but manifest.json is missing; using cases-json as resume case source",
        )
    elif args.resume and latest_rows:
        _add_check(report, "resume_manifest", "error", "results exist but manifest.json is missing")
    else:
        _add_check(report, "resume_manifest", "ok", "no existing manifest conflict")
        if run_settings is not None:
            _add_check(
                report,
                "resume_settings",
                "ok",
                "no existing manifest settings conflict",
                requested_fingerprint=run_settings.get("fingerprint"),
            )

    if not args.resume and any((output_dir / name).exists() for name in ("results.jsonl", "captions.jsonl", "summary.json")):
        _add_check(report, "fresh_output", "warn", "fresh run will replace previous result/caption artifacts")

    lock_path = output_dir / runner.RUNNER_LOCK_NAME
    live_runner_lock = False
    if lock_path.exists():
        try:
            lock = _read_json(lock_path)
            if not isinstance(lock, Mapping):
                raise ValueError("lock_is_not_object")
            pid_alive = _pid_is_alive(lock.get("pid"))
            age = max(0.0, time.time() - float(lock.get("heartbeat_epoch") or 0.0))
            capability_state = _runner_capability_state(output_dir, lock)
            if pid_alive:
                live_runner_lock = True
                _add_check(
                    report,
                    "runner_lock",
                    "error",
                    "output directory is currently owned by a live runner",
                    pid=lock.get("pid"),
                    age_seconds=age,
                    **capability_state,
                )
            else:
                _add_check(
                    report,
                    "runner_lock",
                    "warn",
                    "stale/dead runner lock is present; runner can remove it before resume",
                    pid=lock.get("pid"),
                    age_seconds=age,
                    **capability_state,
                )
        except Exception as exc:  # noqa: BLE001
            _add_check(
                report,
                "runner_lock",
                "warn",
                f"invalid runner lock is present; runner can remove it before resume: {exc}",
                error_type=type(exc).__name__,
            )
    else:
        _add_check(report, "runner_lock", "ok", "no active runner lock")

    if output_dir.exists() and any((output_dir / name).exists() for name in ("manifest.json", "results.jsonl", "heartbeat.json", "summary.json")):
        audit_report = audit.audit_soak(
            output_dir,
            max_heartbeat_age_seconds=args.max_heartbeat_age,
            allow_running_incomplete=True,
        )
        report["artifact_audit"] = audit_report
        artifact_status, artifact_detail, artifact_fields = _artifact_audit_preflight_status(
            audit_report,
            resume_requested=bool(args.resume),
        )
        _add_check(
            report,
            "artifact_audit",
            artifact_status,
            artifact_detail,
            **artifact_fields,
        )
    else:
        _add_check(report, "artifact_audit", "ok", "no existing run artifacts to audit")

    write_probes: list[dict[str, Any]] = []
    if live_runner_lock:
        _add_check(
            report,
            "artifact_write",
            "ok",
            "artifact output write probe skipped because a live runner owns the directory",
            path=str(output_dir),
            skipped=True,
        )
    else:
        artifact_probe = _probe_directory_writable(output_dir, label="artifact output directory")
        write_probes.append(artifact_probe)
        _add_check(
            report,
            "artifact_write",
            str(artifact_probe.get("status") or "error"),
            str(artifact_probe.get("detail") or "artifact output directory write probe failed"),
            path=artifact_probe.get("path"),
            error_type=artifact_probe.get("error_type"),
        )
    if bool(getattr(args, "save_dataset_text_labels", False)):
        label_dirs = _text_label_target_dirs(args.dataset_root.resolve(), cases)
        label_failures: list[dict[str, Any]] = []
        for label_dir in label_dirs:
            probe = _probe_directory_writable(label_dir, label="dataset text-label directory")
            write_probes.append(probe)
            if probe.get("status") != "ok":
                label_failures.append(probe)
        if label_failures:
            _add_check(
                report,
                "text_label_write",
                "error",
                f"{len(label_failures)} of {len(label_dirs)} dataset text-label directories are not writable",
                failures=label_failures[:5],
                checked_directories=len(label_dirs),
            )
        else:
            _add_check(
                report,
                "text_label_write",
                "ok",
                f"{len(label_dirs)} dataset text-label directories are writable",
                checked_directories=len(label_dirs),
            )
    else:
        _add_check(report, "text_label_write", "ok", "dataset text-label writes not requested")
    report["write_probes"] = write_probes

    _add_model_cache_check(report, args, request_template)

    max_artifact_log_bytes = runner.normalize_artifact_log_bytes(args.max_artifact_log_bytes)
    estimate = _estimate_required_bytes(
        selected_cases=len(cases),
        remaining_cases=len(remaining_keys),
        attempts=args.attempts,
        max_artifact_log_bytes=max_artifact_log_bytes,
    )
    target = _existing_parent(output_dir)
    usage = shutil.disk_usage(target)
    min_free_bytes = int(max(0.0, float(args.min_free_gb or 0.0)) * 1024 * 1024 * 1024)
    required_with_safety = None
    if estimate.get("estimated_bytes") is not None:
        required_with_safety = int(float(estimate["estimated_bytes"]) * max(1.0, float(args.disk_safety_factor or 1.0)))
    report["disk"] = {
        "target": str(target),
        "free_bytes": usage.free,
        "free_human": _format_bytes(usage.free),
        "min_free_bytes": min_free_bytes,
        "min_free_human": _format_bytes(min_free_bytes),
        "max_artifact_log_bytes": max_artifact_log_bytes,
        "attempts": int(args.attempts or 1),
        "safety_factor": float(args.disk_safety_factor or 1.0),
        **estimate,
        "required_with_safety_bytes": required_with_safety,
        "required_with_safety_human": _format_bytes(required_with_safety),
    }
    if usage.free < min_free_bytes:
        _add_check(
            report,
            "disk_budget",
            "error",
            f"free disk {_format_bytes(usage.free)} is below minimum {_format_bytes(min_free_bytes)}",
        )
    elif required_with_safety is not None and usage.free < required_with_safety:
        _add_check(
            report,
            "disk_budget",
            "error",
            f"free disk {_format_bytes(usage.free)} is below estimated need {_format_bytes(required_with_safety)}",
        )
    elif required_with_safety is None:
        _add_check(
            report,
            "disk_budget",
            "warn",
            "raw logs are uncapped, so disk use cannot be bounded",
            structured_estimate_human=_format_bytes(estimate.get("structured_bytes")),
        )
    else:
        _add_check(
            report,
            "disk_budget",
            "ok",
            f"free disk {_format_bytes(usage.free)} covers estimated need {_format_bytes(required_with_safety)}",
        )

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=runner.DEFAULT_DATASET)
    parser.add_argument("--cases-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--all-images", action="store_true")
    parser.add_argument("--caption-mode", choices=["full", "windowed"], default="full")
    parser.add_argument("--windowed-full-image-strategy", choices=["visual", "text_only"], default="visual")
    parser.add_argument("--model-id", default=runner.DEFAULT_MODEL)
    parser.add_argument("--model-variant", choices=["auto", "Instruct", "Thinking"], default="Instruct")
    parser.add_argument("--refinement-model-id", default="same")
    parser.add_argument("--fallback-model-id", default="auto")
    parser.add_argument("--loop-recovery", choices=["off", "safe_retry", "safe_retry_fallback"], default="safe_retry_fallback")
    parser.add_argument(
        "--request-json",
        type=Path,
        default=None,
        help="Apply the same caption request template as the runner before validating effective model settings.",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Downgrade missing concrete model cache from error to warn because downloads are intentional.",
    )
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-dataset-text-labels", action="store_true")
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--max-artifact-log-bytes", type=int, default=runner.DEFAULT_ARTIFACT_LOG_BYTES)
    parser.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    parser.add_argument("--disk-safety-factor", type=float, default=DEFAULT_DISK_SAFETY_FACTOR)
    parser.add_argument("--max-heartbeat-age", type=float, default=900.0)
    parser.add_argument("--max-boxes", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--final-sentences", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=672)
    parser.add_argument("--window-overlap", type=float, default=0.1)
    parser.add_argument("--mlx-max-image-side", type=int, default=512)
    parser.add_argument("--retry-image-side-scale", type=float, default=runner.DEFAULT_RETRY_IMAGE_SIDE_SCALE)
    parser.add_argument("--min-retry-image-side", type=int, default=runner.DEFAULT_MIN_RETRY_IMAGE_SIDE)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--use-sampling", action="store_true")
    parser.add_argument("--prompt", default=runner.DEFAULT_PROMPT)
    parser.add_argument("--fail-on-warn", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = preflight_soak(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    status = str(report.get("status") or "error")
    if status == "error" or (args.fail_on_warn and status == "warn"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
