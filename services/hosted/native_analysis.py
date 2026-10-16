"""Trusted heavy factory for native Tator engines in a private child process.

The worker-side factory never imports the legacy application. Only its fresh
child does, after rebuilding environment, cwd, storage and snapshot authority.
No ASGI server or legacy mutation route is exposed.
"""
from __future__ import annotations

import asyncio
import ctypes
from contextlib import closing
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time

from .analysis import _hash_file, _inside, legacy_source_manifest, read_candidate_set, read_canonical_snapshot
from .artifacts import open_regular
from .errors import DomainError
from .operations import HostedOperations
from .store import _json


OPERATIONS = {"statistics", "dqe", "dqe_review", "pruning", "ingestion", "salad_training", "caption_export", "yolo_training", "sam3_training", "rfdetr_training", "qwen_training", "clip_training"}
NAMESPACES = {"dqe_cache": "class_analysis/cache", "dqe_parent": "class_analysis/review_parent", "pruning_cache": "class_analysis/pruning/cache", "salad_heads": "salad_heads", "huggingface": "home/huggingface", "models": "models", "classifier_cache": "clip_cache"}
SOURCE_KEYS = {"dataset_id", "dataset_root", "source_mode", "annotation_source", "workspace_dir", "workspace_manifest", "workspace_manifest_path", "session_id", "snapshot_id", "output_dir", "experiment_log_dir", "base_weights", "pretrain_weights", "staged_temp_dir", "random_split", "images_path", "labels_path", "model_output", "labelmap_output", "input_labelmap", "device", "progress_cb", "metrics_cb", "should_cancel"}
SOURCE_KEYS |= {'candidate_image_ids', 'candidate_uploads', 'reference_uploads', 'train_uploads', 'reference_source',
                'reference_snapshot_id', 'reference_dataset_id', 'reference_dataset_kind', 'reference_session_id',
                'reference_storage_mode', 'reference_kind', 'reference_open_path',
                'use_server_reference_dataset', 'use_backend_reference_dataset'}


def _safe_relative(value):
    if (not isinstance(value, str) or not value or "\\" in value or "\0" in value
            or PurePosixPath(value).is_absolute() or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise DomainError("native_cache_path_invalid", 422)
    return value


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(_json(value))
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)


def _copy_verified(source, destination, sha256, size_bytes):
    if _hash_file(source) != (sha256, size_bytes):
        raise DomainError("native_cache_input_changed", 409)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open_regular(source) as incoming, destination.open("xb") as output:
        shutil.copyfileobj(incoming, output, 1024 * 1024)
        output.flush()
        os.fsync(output.fileno())
    if _hash_file(destination) != (sha256, size_bytes):
        raise DomainError("native_cache_copy_changed", 409)


def _rebase_descriptor_cache(directory, copied):
    """Move only attested descriptor references; unmatched rows become misses.

    Native keys, vector slices and normalization stay unchanged. The copied
    acceleration database must never reach back into an original dataset/job.
    """
    from services.dataset_pruning_cache import file_state
    mapping = {}
    for item in copied:
        previous = mapping.get(item['original_path'])
        # Content-addressed pins can legitimately supply two filenames with
        # identical bytes (model packages and attached head aliases do this).
        if previous and (previous['sha256'], previous['size_bytes']) != (item['sha256'], item['size_bytes']):
            raise DomainError("native_cache_reference_ambiguous", 409)
        mapping.setdefault(item['original_path'], item)
    summary = {"rebound_rows": 0, "discarded_unavailable_rows": 0}
    for item in copied:
        if item["namespace"] != "pruning_cache" or item["relative_path"] != "descriptors.sqlite3":
            continue
        database = directory / NAMESPACES[item["namespace"]] / item["relative_path"]
        with closing(sqlite3.connect(database)) as con, con:
            if [row[1] for row in con.execute("PRAGMA table_info(descriptors)")] != ["key", "dim", "vector", "reference"]:
                raise DomainError("native_descriptor_schema_invalid", 409)
            after = ""
            while rows := con.execute("SELECT key,reference FROM descriptors WHERE reference IS NOT NULL AND key>? ORDER BY key LIMIT 512", (after,)).fetchall():
                for key, raw in rows:
                    ref = json.loads(raw)
                    producer = mapping.get(ref.get("path"))
                    qualified = producer is not None and (
                        ref.get("sha256") == producer["sha256"] if ref.get("sha256") else ref.get("state") == producer.get("source_state"))
                    if not qualified:
                        con.execute("DELETE FROM descriptors WHERE key=?", (key,))
                        summary["discarded_unavailable_rows"] += 1
                        continue
                    path = directory / NAMESPACES[producer["namespace"]] / producer["relative_path"]
                    ref.update(path=str(path), state=file_state(path), sha256=producer["sha256"])
                    con.execute("UPDATE descriptors SET reference=? WHERE key=?", (_json(ref), key))
                    summary["rebound_rows"] += 1
                after = rows[-1][0]
    return summary


def native_environment(directory: Path, device: str) -> dict[str, str]:
    """Construct, rather than inherit, the application's storage environment."""
    root = str(directory)
    environment = {"PATH": os.environ.get("PATH", os.defpath), "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "PYTHONUNBUFFERED": "1", "HOME": str(directory / "home"), "TMPDIR": str(directory / "tmp"),
        "HF_HOME": str(directory / "home/huggingface"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "CLIP_EMBED_CACHE": str(directory / "clip_cache"),
        "TATOR_SKIP_CLIP_LOAD": "1", "TATOR_INFERENCE_DEVICE": device, "SAM3_DEVICE": device,
        "CLASS_ANALYSIS_ROOT": str(directory / "class_analysis"), "CLASS_ANALYSIS_CACHE_ROOT": str(directory / "class_analysis/cache"),
        "CLASS_ANALYSIS_PURGE_ON_START": "0", "DATASET_ROOT": str(directory / "registry"),
        "TATOR_SAM3_GENERATED_CONFIG_ROOT": str(directory / "vendor/sam3/sam3/train/configs/generated"),
        "DATASET_LINK_ROOTS": root, "DATASET_TRANSIENT_STATE_ROOT": str(directory / "transient"),
        "DATASET_ANNOTATION_EXPORT_CACHE_ROOT": str(directory / "exports"), "LOCAL_SALAD_HEAD_ROOT": str(directory / "salad_heads"),
        "QWEN_TRAINING_ROOT": str(directory / "qwen_runs"), "SAM3_TRAINING_ROOT": str(directory / "sam3_runs"),
        "YOLO_TRAINING_ROOT": str(directory / "yolo_runs"), "YOLO_MODEL_ROOT": str(directory / "yolo_models"),
        "RFDETR_TRAINING_ROOT": str(directory / "rfdetr_runs"), "RFDETR_MODEL_ROOT": str(directory / "rfdetr_models"),
        "DATA_INGESTION_ROOT": str(directory / "ingestion"), "DETECTOR_PREFS_ROOT": str(directory / "detectors"),
        "SEGMENTATION_ROOT": str(directory / "segmentation"), "SAM3_PROMPT_HELPER_ROOT": str(directory / "prompts"),
        "TATOR_HOSTED_NATIVE_PARENT_PID": str(os.getpid())}
    for name in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if name in os.environ:
            environment[name] = os.environ[name]
    return environment


class NativeAnalysisAdapter:
    def __init__(self, *, config, device):
        self.config, self.device = dict(config), str(device)
        self.root = Path(self.config["native_root"]).absolute()
        if any(path.is_symlink() for path in (self.root, *self.root.parents)):
            raise DomainError("native_root_invalid", 422)
        self.root.mkdir(parents=True, exist_ok=True)

    def execute(self, operation, payload, context):
        if operation not in OPERATIONS:
            raise DomainError("native_operation_unknown", 422)
        manifest = read_canonical_snapshot(payload["canonical_snapshot"], allowed_root=self.config["canonical_root"])
        if context.get("dataset_id") != manifest["dataset_id"] or context.get("annotation_version") != manifest["snapshot_id"]:
            raise DomainError("native_snapshot_context_changed", 409)
        if operation == "caption_export":
            from .language_schema import validate_instruction_manifest
            validate_instruction_manifest(manifest)
        candidates = None
        if operation == 'ingestion':
            candidates = read_candidate_set(payload['candidate_set'], allowed_root=self.config['canonical_root'])
            if candidates['dataset_id'] != manifest['dataset_id']:
                raise DomainError('ingestion_candidate_dataset_changed', 409)
        elif 'candidate_set' in payload:
            raise DomainError('ingestion_candidates_not_applicable', 422)
        requested = {key: value for key, value in payload.items() if key not in {"canonical_snapshot", "candidate_set"}}
        if set(requested) - set(self.config.get("allowed_parameters", [])) or set(requested) & SOURCE_KEYS:
            raise DomainError("native_parameters_not_allowed", 422)
        parameters = {**self.config.get("parameters", {}), **requested}
        if set(parameters) & SOURCE_KEYS:
            raise DomainError("native_source_parameter_forbidden", 422)
        directory = Path(tempfile.mkdtemp(prefix="native-", dir=self.root))
        for name in ("home", "tmp", "registry", "class_analysis", "models"):
            (directory / name).mkdir(exist_ok=True)
        copied = []
        from services.dataset_pruning_cache import file_state
        for item in self.config.get("cache_inputs", []):
            namespace = item["namespace"]
            if namespace not in NAMESPACES:
                raise DomainError("native_cache_namespace_invalid", 422)
            source = Path(item["path"]).absolute()
            allowed = [Path(root).resolve(strict=True) for root in self.config.get("cache_input_roots", [])]
            matched = next((root for root in allowed if source.resolve(strict=True).is_relative_to(root)), None)
            if matched is None:
                raise DomainError("native_cache_source_not_allowed", 422)
            source = _inside(source, matched)
            relative = NAMESPACES[namespace] + "/" + _safe_relative(item["relative_path"])
            state = file_state(source)
            _copy_verified(source, directory / relative, item["sha256"], item["size_bytes"])
            if file_state(source) != state:
                raise DomainError("native_cache_input_changed", 409)
            copied.append({"namespace": namespace, "relative_path": item["relative_path"], "sha256": item["sha256"], "size_bytes": item["size_bytes"],
                           "original_path": item.get("original_path", str(source)), "source_state": state})
        cache_rebase = _rebase_descriptor_cache(directory, copied)
        request = {"operation": operation, "parameters": parameters, "snapshot_reference": payload["canonical_snapshot"],
                   "canonical_root": self.config["canonical_root"], "cache_inputs": copied, "device": self.device,
                   "cache_rebase": cache_rebase,
                   "timeout": float(self.config.get("timeout", 86400))}
        if candidates is not None:
            head_id = parameters.get('salad_head_id')
            digest = self.config.get('hosted_salad_heads', {}).get(head_id)
            if not digest or not any(item['namespace'] == 'salad_heads' and item['relative_path'] == str(head_id) + '.pt'
                                     and item['sha256'] == digest for item in copied):
                raise DomainError('ingestion_salad_head_not_attached', 409)
            request.update(candidate_reference=payload['candidate_set'], hosted_salad_heads={head_id: digest})
        if operation == 'dqe_review':
            parent = self.config.get('parent_analysis', {})
            if parent.get('snapshot_id') != manifest['snapshot_id'] or not isinstance(parameters.get('point_id'), str):
                raise DomainError('dqe_parent_snapshot_changed', 409)
            relative = _safe_relative(self.config.get('review_model_directory', ''))
            if relative != 'models/review' or not any(item['namespace'] == 'models' and item['relative_path'] == 'review/config.json' for item in copied):
                raise DomainError('dqe_review_model_not_verified', 409)
            request.update(parent_analysis=parent, review_model_directory=relative)
        _write_json(directory / "request.json", request)
        environment = native_environment(directory, self.device)
        # Model paths are trusted, relative references to the verified copies.
        for name, relative in self.config.get("model_environment", {}).items():
            if name not in {"SAM_CHECKPOINT_PATH", "SAM3_CHECKPOINT_PATH", "SAM3_BPE_PATH"}:
                raise DomainError("native_model_environment_invalid", 422)
            path = _inside(directory / _safe_relative(relative), directory)
            if str(path.relative_to(directory)) not in {NAMESPACES[item["namespace"]] + "/" + item["relative_path"] for item in copied}:
                raise DomainError("native_model_not_verified", 409)
            environment[name] = str(path)
        with (directory / "native.log").open("xb") as log:
            process = subprocess.run([sys.executable, "-m", "services.hosted.native_analysis", "--child", str(directory)],
                cwd=directory, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
        if process.returncode:
            raise DomainError("native_engine_failed", 409, {"exit_code": process.returncode, "diagnostic_directory": str(directory)})
        # SQLite can checkpoint its last connection during interpreter exit.
        # Attest only after the complete child has exited and these writes end.
        _publication_receipt(directory, operation)
        receipt_hash, receipt_size = _hash_file(directory / "receipt.json")
        return {"operation": operation, "state": "complete", "receipt_sha256": receipt_hash, "receipt_size_bytes": receipt_size,
                "native_directory": str(directory), "snapshot_id": manifest["snapshot_id"],
                "dataset_id": manifest["dataset_id"], "revision": manifest["revision"]}


class _PinnedMedia:
    def __init__(self, manifest):
        self.paths = {image["media_hash"]: Path(image["media_path"]) for image in manifest["images"]}

    def open(self, reference, *, verify=True):
        digest = reference if isinstance(reference, str) else reference.sha256
        return open_regular(self.paths[digest])


def _materialize(manifest, directory):
    serializer = object.__new__(HostedOperations)
    serializer.artifacts = _PinnedMedia(manifest)
    dataset = directory / "class_analysis/input"
    dataset.mkdir(parents=True)
    inputs = {}
    for entry in serializer._snapshot_entries(manifest, unsplit_as_train=True):
        path = dataset / entry.path
        path.parent.mkdir(parents=True, exist_ok=True)
        if entry.artifact is not None:
            source = serializer.artifacts.paths[entry.artifact.sha256]
            _copy_verified(source, path, entry.artifact.sha256, entry.artifact.size_bytes)
        else:
            path.write_bytes(entry.data)
        inputs[entry.path] = _hash_file(path)
    source = legacy_source_manifest(manifest, dataset)
    _write_json(dataset / "native-source.json", source)
    return dataset, source, inputs


def _wait(get_status, job_id, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = get_status(job_id)
        state = status.get("status")
        if state in {"completed", "succeeded"}:
            # Terminal state may precede final cache/provenance flushing in a
            # native worker's finally block. Join that exact job's threads.
            for thread in threading.enumerate():
                if thread is not threading.current_thread() and job_id[:8] in thread.name:
                    thread.join(max(0, deadline - time.monotonic()))
                    if thread.is_alive():
                        raise TimeoutError("Native job did not finish publishing its outputs")
            final = get_status(job_id)
            if final.get("status") not in {"completed", "succeeded"}:
                raise RuntimeError("Native terminal job changed during publication")
            return final
        if state in {"failed", "cancelled", "interrupted", "blocked"}:
            raise RuntimeError(f"native job {state}: {status.get('error') or status.get('message')}")
        time.sleep(0.2)
    raise TimeoutError("Native engine deadline exceeded")


def _training_membership(manifest):
    from .splits import capture_groups
    membership = {image["id"]: image["split"] for image in manifest["images"]}
    if "unsplit" in membership.values() or not {"train", "val"}.issubset(membership.values()):
        raise RuntimeError("Training requires explicit train/validation membership for every image")
    groups = {}
    for image in manifest["images"]:
        for identity in capture_groups(image['annotations']):
            previous = groups.setdefault(identity, image["split"])
            if previous != image["split"]:
                raise RuntimeError("A capture/video group crosses the declared dataset split")
    return membership


def _classifier_split(membership):
    """Only replace sample admission, never fitting, losses or feature recipes."""
    def split(labels, groups, **_unused):
        import numpy as np
        try:
            splits = [membership[str(group)] for group in groups]
        except KeyError as error:
            raise RuntimeError("Classifier crop lost its canonical image identity") from error
        train = np.asarray([index for index, value in enumerate(splits) if value == "train"], dtype=np.int64)
        val = np.asarray([index for index, value in enumerate(splits) if value == "val"], dtype=np.int64)
        if len(train) + len(val) != len(labels) or not len(train) or not len(val):
            raise RuntimeError("Classifier samples require nonempty explicit train/validation splits")
        train_classes = {str(labels[index]) for index in train}
        if len(train_classes) < 2 or not {str(value) for value in labels}.issubset(train_classes):
            raise RuntimeError("The fixed training split lacks classes; review the dataset split")
        return train, val, True
    return split


def _run_classifier(request, manifest, dataset, directory):
    from tools import clip_training
    membership = _training_membership(manifest)
    inputs, outputs = directory / "classifier-input", directory / "classifier-output"
    (inputs / "images").mkdir(parents=True)
    (inputs / "labels").mkdir()
    outputs.mkdir()
    for split in ("train", "val"):
        for image in (dataset / split / "images").iterdir():
            shutil.copy2(image, inputs / "images" / image.name)
            shutil.copy2(dataset / split / "labels" / (image.stem + ".txt"), inputs / "labels" / (image.stem + ".txt"))
    # This module exists only in the disposable child process. A supplied split
    # is authoritative; the native trainer's fallback may not move crop samples.
    clip_training._split_train_test_indices = _classifier_split(membership)
    artifacts = clip_training.train_clip_from_yolo(str(inputs / "images"), str(inputs / "labels"),
        str(outputs / "classifier.pkl"), str(outputs / "labelmap.pkl"),
        input_labelmap=str(dataset / "labelmap.txt"), device=request["device"], **request["parameters"])
    return {"artifacts": asdict(artifacts), "sample_split": "canonical_train_val", "held_out_test_images": sum(value == "test" for value in membership.values())}


def _stage_sam3_sources(directory):
    """Copy the trusted training package so Hydra-generated files stay private."""
    source = Path(__file__).resolve().parents[2] / "sam3"
    if not (source / "sam3/train/train.py").is_file():
        raise RuntimeError("Bundled SAM3 training sources are not installed")
    destination = directory / "vendor/sam3"
    allowed = {".py", ".pyi", ".yaml", ".yml", ".json", ".txt", ".gz", ".so", ".pyd", ".toml", ".jinja"}
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if any(part in {".git", "__pycache__", "generated"} for part in relative.parts):
            continue
        if path.is_symlink():
            raise RuntimeError("Bundled SAM3 source contains an unqualified symlink")
        if path.is_file() and path.suffix in allowed:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return destination


class _LocalUpload:
    """The native upload API reads one immutable source at a time, without HTTP."""
    def __init__(self, path, filename):
        self.path, self.filename, self.stream = Path(path), filename, None

    async def read(self, size=-1):
        if self.stream is None:
            self.stream = open_regular(self.path)
        return self.stream.read(size)

    async def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def _run_ingestion(api, request, manifest, source, directory):
    """Adapt population authority only; use the existing trainer and ranking job."""
    references = [_LocalUpload(row['image_path'], row['image_relpath']) for row in source['images']]
    parameters = {**request['parameters'], 'source_mode': 'uploaded_media',
                  'reference_source': 'hosted_confirmed_snapshot', 'reference_dataset_id': manifest['dataset_id'],
                  'reference_dataset_kind': '', 'reference_storage_mode': '', 'reference_kind': '', 'reference_session_id': '',
                  'reference_snapshot_id': manifest['snapshot_id'], 'reference_dataset_label': manifest['dataset_name'],
                  'use_server_reference_dataset': False, 'use_backend_reference_dataset': False}
    candidates, uploads = None, []
    if request['operation'] == 'ingestion':
        candidates = read_candidate_set(request['candidate_reference'], allowed_root=request['canonical_root'])
        if candidates['dataset_id'] != manifest['dataset_id']:
            raise DomainError('ingestion_candidate_dataset_changed', 409)
        head_id = parameters.get('salad_head_id')
        digest = request.get('hosted_salad_heads', {}).get(head_id)
        path = _inside(directory / 'salad_heads' / (_safe_relative(head_id) + '.pt'), directory)
        if not digest or _hash_file(path)[0] != digest:
            raise DomainError('ingestion_salad_head_not_attached', 409)
        original_match = api._local_salad_head_reference_matches_request

        def matches(metadata, incoming):
            # The original validator still loads this exact immutable head and
            # checks whole-image scope. Hosted attachment replaces only the old
            # source registry-ID comparison; trained metadata is never rewritten.
            if (incoming.get('reference_source') == 'hosted_confirmed_snapshot'
                    and incoming.get('reference_dataset_id') == manifest['dataset_id']
                    and incoming.get('reference_snapshot_id') == manifest['snapshot_id']
                    and incoming.get('salad_head_id') == head_id):
                return True
            return original_match(metadata, incoming)

        from PIL import Image
        for image in candidates['images']:
            with open_regular(image['media_path']) as stream, Image.open(stream) as decoded:
                suffix = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp', 'TIFF': '.tiff', 'BMP': '.bmp'}.get(decoded.format)
                if suffix is None or decoded.size != (image['width'], image['height']):
                    raise DomainError('ingestion_candidate_encoding_invalid', 409)
            uploads.append(_LocalUpload(image['media_path'], image['id'] + suffix))
    try:
        if candidates is None:
            created = asyncio.run(api.create_local_salad_training_job(_json(parameters), references))
        else:
            api._local_salad_head_reference_matches_request = matches
            created = asyncio.run(api.create_data_ingestion_analysis_job(_json(parameters), uploads, references))
        status = _wait(api.get_data_ingestion_job, created['job_id'], request['timeout'])
        result = api.get_data_ingestion_result(created['job_id'])
        if candidates is not None:
            for key, population in (('items', candidates['images']), ('reference_items', manifest['images'])):
                identities = {image['id']: image for image in population}
                seen = set()
                for row in result.get(key, []):
                    image_id = Path(row.get('filename', '')).stem
                    if image_id not in identities or image_id in seen:
                        raise DomainError('ingestion_output_identity_changed', 409)
                    image = identities[image_id]
                    seen.add(image_id)
                    row['hosted_identity'] = {name: image[name] for name in ('id', 'version_id', 'media_hash')}
                if seen != set(identities):
                    raise DomainError('ingestion_output_membership_changed', 409)
            distribution = api.get_data_ingestion_distribution(created['job_id'])
            identities = {row['id']: row for row in [*candidates['images'], *manifest['images']]}
            for point in distribution.get('points', []):
                image = identities.get(Path(point.get('filename', '')).stem)
                if image is None:
                    raise DomainError('ingestion_output_identity_changed', 409)
                point['hosted_identity'] = {name: image[name] for name in ('id', 'version_id', 'media_hash')}
            result['distribution'] = distribution
        return {'native_job_id': created['job_id'], 'status': status, 'result': result,
                'candidate_set_id': candidates['candidate_set_id'] if candidates is not None else None,
                'reference_snapshot_id': manifest['snapshot_id']}
    finally:
        for upload in [*references, *uploads]:
            if upload.stream is not None:
                upload.stream.close()
        if candidates is not None:
            api._local_salad_head_reference_matches_request = original_match


def _run_native(request, manifest, dataset, source, directory):
    operation, parameters = request["operation"], request["parameters"]
    if operation == "statistics":
        from utils.datasets import _iter_yolo_images
        from services.dataset_pruning import parse_objects
        counts = {split: len(_iter_yolo_images(dataset / split / "images")) for split in ("train", "val", "test", "unsplit")}
        objects = {name: 0 for name in source["labelmap"]}
        for row in source["images"]:
            parsed, errors = parse_objects(row["label_lines"], len(source["labelmap"]))
            if errors:
                raise RuntimeError("Frozen annotation conversion failed")
            for item in parsed:
                objects[source["labelmap"][item["class_id"]]] += 1
        if sum(counts.values()) != len(source["images"]):
            raise RuntimeError("Snapshot membership changed")
        original_counts = {split: sum(row["original_split"] == split for row in source["images"]) for split in counts}
        return {"image_count": sum(counts.values()), "splits": original_counts, "object_counts": objects,
                "mechanisms": ["utils.datasets._iter_yolo_images", "services.dataset_pruning.parse_objects"]}
    if operation == "clip_training":
        return _run_classifier(request, manifest, dataset, directory)
    if operation.endswith("_training") and operation != 'salad_training':
        _training_membership(manifest)
    if operation == 'dqe_review':
        parent = request['parent_analysis']
        native_id = parent.get('native_job_id', '')
        if (not isinstance(native_id, str) or not native_id.startswith('ca_')
                or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in native_id)):
            raise RuntimeError('Invalid frozen parent analysis')
        parent_dir = directory / 'class_analysis' / native_id
        (directory / 'class_analysis/review_parent').rename(parent_dir)
        config_path = parent_dir / 'config.json'
        config = json.loads(config_path.read_text())
        config.update(source_mode='active_workspace', workspace_dir=str(dataset), workspace_manifest=source,
                      workspace_manifest_path='', snapshot_id=manifest['snapshot_id'], yolo_layout='split', dataset_id='', session_id='')
        config_path.chmod(0o600)
        config_path.write_text(_json(config))
    private_sam3 = _stage_sam3_sources(directory) if operation == "sam3_training" else None

    # This import is permitted ONLY in this fresh confined child, never in the
    # control process or ordinary per-user inference worker.
    import localinferenceapi as api
    if private_sam3 is not None:
        # Exact copied vendor code, with the same Hydra package-relative config
        # name; only its working/generated-file location changes.
        api.SAM3_VENDOR_ROOT = private_sam3
        api.SAM3_PACKAGE_ROOT = private_sam3 / "sam3"
    dataset_id = manifest["dataset_id"]
    training_dataset = dataset
    if operation in {"sam3_training", "qwen_training"}:
        training_dataset = directory / "training-input"
        shutil.copytree(dataset, training_dataset, ignore=lambda path, names: {"test"} if Path(path) == dataset else set())
    if operation in {"yolo_training", "rfdetr_training"} and parameters.get("task", "detect") == "segment":
        if any(not obj.get("polygon") for image in manifest["images"] for obj in image["annotations"]["objects"]):
            raise RuntimeError("Segmentation training requires polygons for every labelled object")
        training_dataset = directory / "training-segments"
        shutil.copytree(dataset, training_dataset)
        for label in training_dataset.glob("*/labels/*.txt"):
            segment = label.parent.parent / "segments" / label.name
            label.write_bytes(segment.read_bytes() if segment.exists() else b"")
    registry = directory / "registry" / dataset_id
    registry.mkdir()
    _write_json(registry / "dataset.json", {"id": dataset_id, "label": manifest["dataset_name"], "classes": source["labelmap"],
        "storage_mode": "linked", "linked_root": str(training_dataset), "type": "bbox", "signature": hashlib.sha256(_json(manifest).encode()).hexdigest(),
        "context": manifest.get("settings", {}).get("context", "")})
    if operation in {'ingestion', 'salad_training'}:
        return _run_ingestion(api, request, manifest, source, directory)
    if operation == 'caption_export':
        from services.hosted.caption_export import export_native_instructions
        return export_native_instructions(api, manifest, source, directory, parameters)
    if operation == "dqe":
        payload = {**parameters, "source_mode": "active_workspace", "workspace_dir": str(dataset), "workspace_manifest": source,
                   "workspace_manifest_path": "", "snapshot_id": manifest["snapshot_id"], "yolo_layout": "split", "dataset_id": "", "session_id": ""}
        created = api.create_class_analysis_job(payload)
        status = _wait(api.get_class_analysis_job, created["job_id"], request["timeout"])
        return {"native_job_id": created["job_id"], "status": status, "result": api.get_class_analysis_result(created["job_id"])}
    if operation == 'dqe_review':
        point_id = parameters['point_id']
        payload = {key: value for key, value in parameters.items() if key != 'point_id'}
        # The selected model is a verified portable registry package. Never
        # permit a browser model ID or an automatic fallback to another model.
        payload.update(model_id=str(directory / request['review_model_directory']), keep_vlm_loaded=False)
        created = api.create_class_analysis_qwen_review(native_id, point_id, payload)
        status = _wait(api.get_class_analysis_qwen_review, created['review_id'], request['timeout'])
        return {'native_job_id': native_id, 'native_review_id': created['review_id'],
                'parent_analysis_job_id': parent['job_id'], 'point_id': point_id,
                'status': status, 'result': status.get('result'), 'review': status}
    if operation == "pruning":
        from contextlib import nullcontext
        from services.dataset_pruning_adapter import TatorPruningAdapter
        from services.dataset_pruning_jobs import PruningService
        class SnapshotAdapter(TatorPruningAdapter):
            def guard(self):
                return nullcontext()
            def validate_request(self, payload):
                if payload["expected_revision"] != manifest["revision"]:
                    raise RuntimeError("Snapshot revision mismatch")
            def source(self, payload):
                self.validate_request(payload)
                rows = [{**row, "media_row": {"path": row["image_path"], "split": row["split"], "image_relpath": row["image_relpath"], "image_sha256": row["image_sha256"]}} for row in source["images"]]
                return {"kind": "hosted", "id": dataset_id, "identity": "hosted:" + dataset_id + ":" + manifest["snapshot_id"],
                        "revision": manifest["revision"], "dataset_root": str(dataset), "registry_path": str(registry), "storage_mode": "hosted_snapshot",
                        "annotation_states": {}, "labelmap": source["labelmap"], "image_count": len(rows), "images": rows,
                        "dataset_label": manifest["dataset_name"], "analysis_job_id": ""}
            def check_revision(self, _source):
                _verify_inputs(dataset, initial_inputs)
            def check_inventory(self, _source, images):
                if len(images) != len(source["images"]):
                    raise RuntimeError("Snapshot membership changed")
                _verify_inputs(dataset, initial_inputs)
        initial_inputs = {path.relative_to(dataset).as_posix(): _hash_file(path) for path in dataset.rglob("*") if path.is_file()}
        service = PruningService(directory / "class_analysis/pruning", SnapshotAdapter(api))
        created = service.create({"config": parameters, "analysis_job_id": "", "expected_revision": manifest["revision"], "annotation_source": {"kind": "hosted", "id": dataset_id}})
        status = _wait(service.status, created["job_id"], request["timeout"])
        # Retain the actual fitted object rows, so post-analysis budget controls
        # need neither a model process nor path-bound producer cache references.
        from services.dataset_pruning_planning import replay_objects
        import numpy as np
        context = service.review_context(created["job_id"], {}, False)
        objects = replay_objects(service, context, lambda: False, lambda *_: None)
        np.save(service.directory(created["job_id"]) / "hosted_objects.npy", objects, allow_pickle=False)
        from services.hosted.pruning_components import retain_components
        retain_components(service.directory(created['job_id']), context[3], context[1]['recipe'],
                          context[1]['recipe'].get('component_manifest', {}), service.cache)
        _write_json(service.directory(created["job_id"]) / "hosted-capabilities.json", service.adapter.capabilities())
        return {"native_job_id": created["job_id"], "status": status, "result": service.result(created["job_id"])}
    splits = {row["original_split"] for row in source["images"]}
    if not {"train", "val"}.issubset(splits):
        raise RuntimeError("An explicit train/validation split is required")
    if operation in {"yolo_training", "rfdetr_training"}:
        model = api.YoloTrainRequest if operation == "yolo_training" else api.RfDetrTrainRequest
        payload = model(**{**parameters, "dataset_id": dataset_id, "dataset_root": None})
        create = api.create_yolo_training_job if operation == "yolo_training" else api.create_rfdetr_training_job
        status = api.get_yolo_training_job if operation == "yolo_training" else api.get_rfdetr_training_job
    elif operation == "sam3_training":
        payload = api.Sam3TrainRequest(**{**parameters, "dataset_id": dataset_id, "random_split": False})
        create, status = api.create_sam3_training_job, api.get_sam3_training_job
    elif operation == "qwen_training":
        converted = api.build_qwen_dataset_from_yolo(dataset_id)
        _complete_qwen_conversion(api, converted, training_dataset, source)
        # The converter may reuse the YOLO ID. Its private registry entry must
        # no longer shadow the converted Qwen dataset during native resolution.
        (registry / "dataset.json").unlink()
        payload = api.QwenTrainRequest(**{**parameters, "dataset_id": converted["id"], "random_split": False})
        create = api.create_qwen_training_job
        status = lambda identifier: api._serialize_qwen_job_impl(api._get_qwen_job(identifier))
    else:
        raise RuntimeError("Native operation is not implemented")
    created = create(payload)
    return {"native_job_id": created["job_id"], "status": _wait(status, created["job_id"], request["timeout"])}


def _complete_qwen_conversion(api, converted, dataset, source):
    """Preserve explicit negative members omitted by the legacy converter."""
    target = api._qwen_dataset_child_dir(converted["id"], create=False, detail="qwen_dataset_target_invalid")
    counts = {}
    for split in ("train", "val"):
        path = target / split / "annotations.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        seen = {row["image"] for row in rows}
        expected = [row for row in source["images"] if row["original_split"] == split]
        for row in expected:
            if row["image_relpath"] in seen:
                continue
            if row["label_lines"]:
                raise RuntimeError("Native Qwen conversion omitted a labelled canonical image")
            image = dataset / split / "images" / row["image_relpath"]
            shutil.copyfile(image, target / split / row["image_relpath"])
            rows.append({"image": row["image_relpath"], "context": api._build_qwen_context(source["labelmap"], converted.get("context", "")), "detections": []})
        if {row["image"] for row in rows} != {row["image_relpath"] for row in expected}:
            raise RuntimeError("Native Qwen conversion changed canonical membership")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        counts[split] = len(rows)
    metadata = target / "metadata.json"
    value = json.loads(metadata.read_text())
    value.update(train_count=counts["train"], val_count=counts["val"], image_count=sum(counts.values()), signature=api._compute_dir_signature_impl(target))
    metadata.write_text(_json(value))


def _verify_inputs(dataset, entries):
    for relative, checksum in entries.items():
        if _hash_file(_inside(dataset / relative, dataset)) != tuple(checksum):
            raise RuntimeError("Native engine changed a frozen input")


def _watch_parent():
    parent = int(os.environ["TATOR_HOSTED_NATIVE_PARENT_PID"])
    if sys.platform.startswith("linux"):
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0):
            raise OSError(ctypes.get_errno(), "native PR_SET_PDEATHSIG failed")
    if os.getppid() != parent:
        os._exit(71)
    def watch():
        while True:
            time.sleep(0.25)
            if os.getppid() != parent:
                if os.getpgrp() == parent:
                    os.killpg(os.getpgrp(), signal.SIGKILL)
                os._exit(71)
    threading.Thread(target=watch, daemon=True, name="native-parent-watch").start()


def _child(directory):
    _watch_parent()
    directory = Path(directory).resolve(strict=True)
    if directory != Path.cwd().resolve():
        raise RuntimeError("Native working directory mismatch")
    request = json.loads((directory / "request.json").read_text())
    manifest = read_canonical_snapshot(request["snapshot_reference"], allowed_root=request["canonical_root"])
    if request["operation"] == "caption_export":
        from .language_schema import validate_instruction_manifest
        validate_instruction_manifest(manifest)
    dataset, source, inputs = _materialize(manifest, directory)
    result = _run_native(request, manifest, dataset, source, directory)
    _verify_inputs(dataset, inputs)
    _write_json(directory / "result.json", {"snapshot_id": manifest["snapshot_id"], "dataset_id": manifest["dataset_id"], "revision": manifest["revision"],
        "class_version": manifest["class_version"], "settings_version": manifest["settings_version"], "settings": manifest["settings"],
        "parameters": request["parameters"], "imported_cache_artifacts": request["cache_inputs"], "cache_rebase": request["cache_rebase"], "source": source, "engine_result": result})
def _publication_receipt(directory, operation):
    dataset = directory / "class_analysis/input"
    # Seal committed WAL state into each standalone retained database.
    for path in directory.rglob("*.sqlite3"):
        with closing(sqlite3.connect(path)) as con:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    artifacts = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("Native output contains a symlink")
        if (not path.is_file() or path.name in {"native.log", "receipt.json"} or path.name.endswith(("-wal", "-shm"))
                or path.is_relative_to(dataset) or path.is_relative_to(directory / "tmp")):
            continue
        checksum, size = _hash_file(path)
        artifacts.append({"relative_path": path.relative_to(directory).as_posix(), "sha256": checksum, "size_bytes": size})
    _write_json(directory / "receipt.json", {"operation": operation, "state": "complete", "result_path": "result.json", "artifacts": artifacts})


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--child":
        raise SystemExit("This module is a private hosted heavy-job child.")
    _child(sys.argv[2])
