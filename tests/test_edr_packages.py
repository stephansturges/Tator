from __future__ import annotations

import json
import os
import stat
import zipfile
from pathlib import Path

import pytest

from services.edr_packages import (
    EDR_PACKAGE_MANIFEST_NAME,
    EDR_PACKAGE_META_NAME,
    EDR_PACKAGE_PAYLOAD_DIRNAME,
    EDR_PACKAGE_STAGE_META_NAME,
    EDR_PACKAGE_ZIP_NAME,
    _copy2_if_different,
    _copy_file,
    _copy_tree,
    _copy_tree_filtered,
    _prepare_output_file,
    _stage_tree_if_needed,
    _write_json,
    _write_text,
    edr_package_dir,
    export_edr_package,
    get_edr_package,
    import_edr_package_from_zip,
    list_edr_packages,
    materialize_canonical_edr_package,
    resolve_edr_package_runtime,
    _zip_payload,
)


def _write_min_runtime_payload(package_root: Path, payload_root: Path) -> None:
    (package_root / EDR_PACKAGE_META_NAME).write_text(
        json.dumps({"package_sha256": "dummy"}),
        encoding="utf-8",
    )
    (payload_root / EDR_PACKAGE_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "package_id": package_root.name,
                "runtime_contract": {"config": {}},
                "feature_contract": {},
            }
        ),
        encoding="utf-8",
    )
    (payload_root / "labelmap.txt").write_text("person\n", encoding="utf-8")
    (payload_root / "glossary.json").write_text(json.dumps({"glossary": ""}), encoding="utf-8")


def test_resolve_edr_package_runtime_falls_back_to_runtime_labelmap(tmp_path: Path) -> None:
    packages_root = tmp_path / "edr_packages"
    package_root = packages_root / "pkg1"
    payload_root = package_root / EDR_PACKAGE_PAYLOAD_DIRNAME
    payload_root.mkdir(parents=True, exist_ok=True)

    (package_root / EDR_PACKAGE_META_NAME).write_text(
        json.dumps({"package_sha256": "dummy"}),
        encoding="utf-8",
    )
    (payload_root / EDR_PACKAGE_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "package_id": "pkg1",
                "runtime_contract": {
                    "config": {
                        "labelmap": ["car", "person"],
                    }
                },
                "feature_contract": {},
            }
        ),
        encoding="utf-8",
    )
    (payload_root / "labelmap.txt").write_text("", encoding="utf-8")
    (payload_root / "glossary.json").write_text(json.dumps({"glossary": ""}), encoding="utf-8")

    runtime = resolve_edr_package_runtime(
        packages_root=packages_root,
        package_id="pkg1",
        upload_root=tmp_path / "uploads",
        yolo_job_root=tmp_path / "yolo_runs",
        rfdetr_job_root=tmp_path / "rfdetr_runs",
        calibration_root=tmp_path / "calibration_jobs",
        classifiers_root=tmp_path / "classifiers",
    )

    assert runtime["labelmap"] == ["car", "person"]
    assert (payload_root / "labelmap.txt").read_text(encoding="utf-8").splitlines() == [
        "car",
        "person",
    ]


def test_resolve_edr_package_runtime_replaces_symlinked_labelmap_without_target_write(
    tmp_path: Path,
) -> None:
    packages_root = tmp_path / "edr_packages"
    package_root = packages_root / "pkg1"
    payload_root = package_root / EDR_PACKAGE_PAYLOAD_DIRNAME
    payload_root.mkdir(parents=True, exist_ok=True)

    (package_root / EDR_PACKAGE_META_NAME).write_text(
        json.dumps({"package_sha256": "dummy"}),
        encoding="utf-8",
    )
    (payload_root / EDR_PACKAGE_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "package_id": "pkg1",
                "runtime_contract": {"config": {"labelmap": ["car", "person"]}},
                "feature_contract": {},
            }
        ),
        encoding="utf-8",
    )
    outside = tmp_path / "outside_labelmap.txt"
    outside.write_text("external\n", encoding="utf-8")
    try:
        (payload_root / "labelmap.txt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    (payload_root / "glossary.json").write_text(json.dumps({"glossary": ""}), encoding="utf-8")

    runtime = resolve_edr_package_runtime(
        packages_root=packages_root,
        package_id="pkg1",
        upload_root=tmp_path / "uploads",
        yolo_job_root=tmp_path / "yolo_runs",
        rfdetr_job_root=tmp_path / "rfdetr_runs",
        calibration_root=tmp_path / "calibration_jobs",
        classifiers_root=tmp_path / "classifiers",
    )

    assert runtime["labelmap"] == ["car", "person"]
    assert not (payload_root / "labelmap.txt").is_symlink()
    assert outside.read_text(encoding="utf-8") == "external\n"


def test_edr_text_write_is_atomic_and_replaces_symlink_targets_without_target_write(
    tmp_path: Path,
) -> None:
    labelmap_path = tmp_path / "package" / "payload" / "labelmap.txt"
    labelmap_path.parent.mkdir(parents=True)
    outside_tmp = tmp_path / "outside_tmp.txt"
    outside_final = tmp_path / "outside_final.txt"
    outside_tmp.write_text("external tmp", encoding="utf-8")
    outside_final.write_text("external final", encoding="utf-8")
    tmp_link = labelmap_path.with_suffix(labelmap_path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp_link.symlink_to(outside_tmp)
        labelmap_path.symlink_to(outside_final)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _write_text(labelmap_path, "building\n")

    assert not tmp_link.exists()
    assert not labelmap_path.is_symlink()
    assert labelmap_path.read_text(encoding="utf-8") == "building\n"
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp"
    assert outside_final.read_text(encoding="utf-8") == "external final"


def test_resolve_edr_package_runtime_stages_classifier_not_meta(tmp_path: Path) -> None:
    packages_root = tmp_path / "edr_packages"
    package_root = packages_root / "pkg1"
    payload_root = package_root / EDR_PACKAGE_PAYLOAD_DIRNAME
    classifier_root = payload_root / "models" / "classifier"
    classifier_root.mkdir(parents=True, exist_ok=True)

    (package_root / EDR_PACKAGE_META_NAME).write_text(
        json.dumps({"package_sha256": "dummy"}),
        encoding="utf-8",
    )
    (payload_root / EDR_PACKAGE_MANIFEST_NAME).write_text(
        json.dumps({"package_id": "pkg1", "runtime_contract": {"config": {}}, "feature_contract": {}}),
        encoding="utf-8",
    )
    (payload_root / "labelmap.txt").write_text("person\n", encoding="utf-8")
    (payload_root / "glossary.json").write_text(json.dumps({"glossary": ""}), encoding="utf-8")
    (classifier_root / "demo.meta.pkl").write_text("meta", encoding="utf-8")
    (classifier_root / "demo.pkl").write_text("model", encoding="utf-8")

    classifiers_root = tmp_path / "classifiers"
    runtime = resolve_edr_package_runtime(
        packages_root=packages_root,
        package_id="pkg1",
        upload_root=tmp_path / "uploads",
        yolo_job_root=tmp_path / "yolo_runs",
        rfdetr_job_root=tmp_path / "rfdetr_runs",
        calibration_root=tmp_path / "calibration_jobs",
        classifiers_root=classifiers_root,
    )

    assert runtime["staged_classifier_id"] == "pkg1__demo.pkl"
    assert (classifiers_root / "pkg1__demo.pkl").read_text(encoding="utf-8") == "model"
    assert (classifiers_root / "pkg1__demo.meta.pkl").read_text(encoding="utf-8") == "meta"


def test_resolve_edr_package_runtime_rejects_symlinked_classifier_root_without_target_write(
    tmp_path: Path,
) -> None:
    packages_root = tmp_path / "edr_packages"
    package_root = packages_root / "pkg1"
    payload_root = package_root / EDR_PACKAGE_PAYLOAD_DIRNAME
    classifier_root = payload_root / "models" / "classifier"
    classifier_root.mkdir(parents=True, exist_ok=True)
    _write_min_runtime_payload(package_root, payload_root)
    (classifier_root / "demo.pkl").write_text("model", encoding="utf-8")
    outside = tmp_path / "outside_classifiers"
    outside.mkdir()
    classifiers_root = tmp_path / "classifiers"
    try:
        classifiers_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(RuntimeError, match="edr_package_path_invalid"):
        resolve_edr_package_runtime(
            packages_root=packages_root,
            package_id="pkg1",
            upload_root=tmp_path / "uploads",
            yolo_job_root=tmp_path / "yolo_runs",
            rfdetr_job_root=tmp_path / "rfdetr_runs",
            calibration_root=tmp_path / "calibration_jobs",
            classifiers_root=classifiers_root,
        )

    assert list(outside.iterdir()) == []


def test_resolve_edr_package_runtime_skips_classifier_symlink_escape(tmp_path: Path) -> None:
    packages_root = tmp_path / "edr_packages"
    package_root = packages_root / "pkg1"
    payload_root = package_root / EDR_PACKAGE_PAYLOAD_DIRNAME
    classifier_root = payload_root / "models" / "classifier"
    classifier_root.mkdir(parents=True, exist_ok=True)
    _write_min_runtime_payload(package_root, payload_root)

    outside = tmp_path / "outside.pkl"
    outside.write_text("secret", encoding="utf-8")
    try:
        (classifier_root / "escape.pkl").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    classifiers_root = tmp_path / "classifiers"
    runtime = resolve_edr_package_runtime(
        packages_root=packages_root,
        package_id="pkg1",
        upload_root=tmp_path / "uploads",
        yolo_job_root=tmp_path / "yolo_runs",
        rfdetr_job_root=tmp_path / "rfdetr_runs",
        calibration_root=tmp_path / "calibration_jobs",
        classifiers_root=classifiers_root,
    )

    assert runtime["staged_classifier_id"] is None
    assert not (classifiers_root / "pkg1__escape.pkl").exists()


def test_resolve_edr_package_runtime_skips_run_symlink_dir_escape(tmp_path: Path) -> None:
    packages_root = tmp_path / "edr_packages"
    package_root = packages_root / "pkg1"
    payload_root = package_root / EDR_PACKAGE_PAYLOAD_DIRNAME
    yolo_root = payload_root / "models" / "yolo_run"
    yolo_root.mkdir(parents=True, exist_ok=True)
    _write_min_runtime_payload(package_root, payload_root)

    outside_run = tmp_path / "outside_run"
    outside_run.mkdir()
    (outside_run / "best.pt").write_text("secret", encoding="utf-8")
    try:
        (yolo_root / "escape_run").symlink_to(outside_run, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    runtime = resolve_edr_package_runtime(
        packages_root=packages_root,
        package_id="pkg1",
        upload_root=tmp_path / "uploads",
        yolo_job_root=tmp_path / "yolo_runs",
        rfdetr_job_root=tmp_path / "rfdetr_runs",
        calibration_root=tmp_path / "calibration_jobs",
        classifiers_root=tmp_path / "classifiers",
    )

    assert runtime["staged_yolo_id"] is None
    assert not (tmp_path / "yolo_runs" / "pkg1__yolo" / "best.pt").exists()


def test_resolve_edr_package_runtime_ignores_external_source_feature_repair(
    tmp_path: Path,
) -> None:
    packages_root = tmp_path / "edr_packages"
    package_root = packages_root / "pkg1"
    payload_root = package_root / EDR_PACKAGE_PAYLOAD_DIRNAME
    payload_root.mkdir(parents=True, exist_ok=True)
    discovery_root = tmp_path / "discovery"
    lane_root = discovery_root / "lanes" / "window"
    lane_root.mkdir(parents=True, exist_ok=True)

    import numpy as np

    np.savez(
        lane_root / "features.npz",
        feature_names=np.asarray(["f0", "f1"], dtype=object),
        classifier_classes=np.asarray(["__bg_0", "person"], dtype=object),
        labelmap=np.asarray(["person"], dtype=object),
        feature_schema_hash=np.asarray("abc123", dtype=object),
        feature_schema_version=np.asarray(1),
        support_iou=np.asarray(0.5),
        context_radius=np.asarray(0.075),
        label_iou=np.asarray(0.5),
        eval_iou=np.asarray(0.5),
        embed_proj_dim=np.asarray(1024),
        embed_proj_seed=np.asarray(42),
        image_embed_proj_dim=np.asarray(0),
        image_embed_proj_seed=np.asarray(4242),
        embed_l2_normalize=np.asarray(True),
    )

    (package_root / EDR_PACKAGE_META_NAME).write_text(
        json.dumps({"package_sha256": "dummy"}),
        encoding="utf-8",
    )
    (payload_root / EDR_PACKAGE_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "package_id": "pkg1",
                "runtime_contract": {
                    "config": {
                        "lane_selection": "window",
                        "canonical_edr_json": str(discovery_root / "canonical_edr.json"),
                        "resolved_classifier_id": "/tmp/classifier.pkl",
                    }
                },
                "feature_contract": {
                    "schema_version": 1,
                    "feature_names": [],
                    "feature_schema_hash": "",
                    "feature_schema_version": 1,
                    "support_iou": 0.5,
                    "context_radius": 0.075,
                    "label_iou": 0.5,
                    "eval_iou": 0.5,
                    "embed_proj_dim": 0,
                    "embed_proj_seed": 42,
                    "image_embed_proj_dim": 0,
                    "image_embed_proj_seed": 4242,
                    "embed_l2_normalize": True,
                    "classifier_classes": [],
                    "labelmap": [],
                    "labelmap_hash": "",
                    "context_variant_id": "",
                    "variant_config_json": "{}",
                },
            }
        ),
        encoding="utf-8",
    )
    (payload_root / "labelmap.txt").write_text("", encoding="utf-8")
    (payload_root / "glossary.json").write_text(json.dumps({"glossary": ""}), encoding="utf-8")

    runtime = resolve_edr_package_runtime(
        packages_root=packages_root,
        package_id="pkg1",
        upload_root=tmp_path / "uploads",
        yolo_job_root=tmp_path / "yolo_runs",
        rfdetr_job_root=tmp_path / "rfdetr_runs",
        calibration_root=tmp_path / "calibration_jobs",
        classifiers_root=tmp_path / "classifiers",
    )

    assert runtime["feature_contract"]["embed_proj_dim"] == 0
    assert runtime["feature_contract"]["feature_schema_hash"] == ""
    assert runtime["labelmap"] == []


def test_materialize_canonical_edr_package_rejects_detector_run_traversal(
    tmp_path: Path,
) -> None:
    uploads_root = tmp_path / "uploads"
    packages_root = tmp_path / "edr_packages"
    yolo_job_root = uploads_root / "yolo_runs"
    rfdetr_job_root = uploads_root / "rfdetr_runs"
    calibration_root = uploads_root / "calibration_jobs"
    classifiers_root = uploads_root / "classifiers"
    for root in (yolo_job_root, rfdetr_job_root, calibration_root, classifiers_root):
        root.mkdir(parents=True, exist_ok=True)

    outside_yolo = uploads_root / "outside_yolo"
    outside_yolo.mkdir()
    (outside_yolo / "secret.pt").write_text("do-not-package", encoding="utf-8")

    calibration_job = calibration_root / "calib1"
    calibration_job.mkdir()
    (calibration_job / "ensemble_xgb.meta.json").write_text(
        json.dumps(
            {
                "feature_names": ["cand_score_yolo"],
                "labelmap": ["person"],
                "classifier_classes": [],
                "feature_schema_hash": "schema123",
                "feature_schema_version": 1,
                "support_iou": 0.5,
                "context_radius": 0.075,
                "label_iou": 0.5,
                "eval_iou": 0.5,
                "min_crop_size": 4,
            }
        ),
        encoding="utf-8",
    )

    saved_recipe = {
        "id": "recipe1",
        "name": "Recipe",
        "config": {
            "dataset_id": "demo",
            "recipe_fingerprint": "fp123",
            "canonical_deployment_job_id": "calib1",
            "yolo_id": "../outside_yolo",
            "labelmap": ["person"],
        },
    }

    with pytest.raises(RuntimeError, match="edr_package_invalid_yolo_run_id"):
        materialize_canonical_edr_package(
            packages_root=packages_root,
            saved_recipe=saved_recipe,
            registry_entry={"dataset_id": "demo", "fingerprint": "fp123"},
            fingerprint_payload={},
            yolo_job_root=yolo_job_root,
            rfdetr_job_root=rfdetr_job_root,
            calibration_root=calibration_root,
            classifiers_root=classifiers_root,
            load_labelmap_fn=lambda _dataset_id: ["person"],
            resolve_classifier_path_fn=lambda _classifier_id: None,
            sam3_checkpoint_path=None,
        )

    package_root = packages_root / "canonical_edr_pkg_demo_fp123"
    assert not (package_root / "payload" / "models" / "yolo_run" / "secret.pt").exists()
    assert (outside_yolo / "secret.pt").read_text(encoding="utf-8") == "do-not-package"


def test_edr_package_dir_rejects_traversal_package_id(tmp_path: Path) -> None:
    packages_root = tmp_path / "edr_packages"

    with pytest.raises(ValueError, match="edr_package_id_invalid"):
        edr_package_dir(packages_root, "../edr_packages_evil", create=True)

    assert not (tmp_path / "edr_packages_evil").exists()


def test_edr_package_dir_rejects_symlinked_packages_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside_packages"
    outside.mkdir()
    packages_root = tmp_path / "edr_packages"
    try:
        packages_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(ValueError, match="edr_package_path_invalid"):
        edr_package_dir(packages_root, "pkg1", create=True)

    assert list(outside.iterdir()) == []


def test_edr_package_dir_rejects_symlinked_packages_parent_without_write(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    packages_parent = tmp_path / "linked_parent"
    try:
        packages_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(ValueError, match="edr_package_path_invalid"):
        edr_package_dir(packages_parent / "edr_packages", "pkg1", create=True)

    assert list(outside.iterdir()) == []


def test_edr_package_dir_rejects_nested_symlinked_packages_parent_without_write(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    packages_parent = tmp_path / "linked_parent"
    try:
        packages_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(ValueError, match="edr_package_path_invalid"):
        edr_package_dir(packages_parent / "nested" / "edr_packages", "pkg1", create=True)

    assert list(outside.iterdir()) == []


def test_edr_package_dir_rejects_symlink_package_dir_escape(tmp_path: Path) -> None:
    packages_root = tmp_path / "edr_packages"
    packages_root.mkdir()
    outside = tmp_path / "outside_pkg"
    outside.mkdir()
    try:
        (packages_root / "pkg1").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(ValueError, match="edr_package_path_invalid"):
        edr_package_dir(packages_root, "pkg1")

    assert outside.exists()


def test_list_edr_packages_skips_symlinked_package_dir_escape(tmp_path: Path) -> None:
    packages_root = tmp_path / "edr_packages"
    packages_root.mkdir()
    outside = tmp_path / "outside_pkg"
    outside.mkdir()
    (outside / EDR_PACKAGE_META_NAME).write_text(
        json.dumps({"id": "pkg1", "updated_at": 1}),
        encoding="utf-8",
    )
    try:
        (packages_root / "pkg1").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    assert list_edr_packages(packages_root) == []


def test_list_edr_packages_skips_symlinked_packages_root(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_packages"
    package_root = outside / "pkg1"
    package_root.mkdir(parents=True)
    (package_root / EDR_PACKAGE_META_NAME).write_text(
        json.dumps({"id": "pkg1", "updated_at": 1}),
        encoding="utf-8",
    )
    packages_root = tmp_path / "edr_packages"
    try:
        packages_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    assert list_edr_packages(packages_root) == []


def test_get_edr_package_rejects_symlinked_meta_escape(tmp_path: Path) -> None:
    packages_root = tmp_path / "edr_packages"
    package_root = packages_root / "pkg1"
    package_root.mkdir(parents=True)
    outside_meta = tmp_path / "outside_meta.json"
    outside_meta.write_text(json.dumps({"id": "pkg1"}), encoding="utf-8")
    try:
        (package_root / EDR_PACKAGE_META_NAME).symlink_to(outside_meta)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(FileNotFoundError):
        get_edr_package(packages_root, "pkg1")


def test_export_edr_package_rejects_symlinked_zip_escape(tmp_path: Path) -> None:
    packages_root = tmp_path / "edr_packages"
    package_root = packages_root / "pkg1"
    package_root.mkdir(parents=True)
    outside_zip = tmp_path / "outside.edr.zip"
    outside_zip.write_bytes(b"secret")
    try:
        (package_root / EDR_PACKAGE_ZIP_NAME).symlink_to(outside_zip)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(FileNotFoundError):
        export_edr_package(packages_root, "pkg1")


def test_import_edr_package_rejects_traversal_manifest_id(tmp_path: Path) -> None:
    zip_path = tmp_path / "bad.edr.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            EDR_PACKAGE_MANIFEST_NAME,
            json.dumps({"package_id": "../edr_packages_evil", "dataset_id": "demo"}),
        )

    with pytest.raises(ValueError, match="edr_package_id_invalid"):
        import_edr_package_from_zip(zip_path=zip_path, packages_root=tmp_path / "edr_packages")

    assert not (tmp_path / "edr_packages_evil").exists()


@pytest.mark.parametrize("member_name", ["C:/escape.txt", "\\\\server\\share\\escape.txt"])
def test_import_edr_package_rejects_windows_absolute_members(
    tmp_path: Path, member_name: str
) -> None:
    zip_path = tmp_path / "bad_windows.edr.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(EDR_PACKAGE_MANIFEST_NAME, json.dumps({"package_id": "pkg1"}))
        zf.writestr(member_name, b"nope")

    with pytest.raises(RuntimeError, match="edr_package_path_traversal"):
        import_edr_package_from_zip(zip_path=zip_path, packages_root=tmp_path / "edr_packages")

    assert not (tmp_path / "escape.txt").exists()


def test_import_edr_package_rejects_symlink_member(tmp_path: Path) -> None:
    zip_path = tmp_path / "symlink.edr.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(EDR_PACKAGE_MANIFEST_NAME, json.dumps({"package_id": "pkg1"}))
        info = zipfile.ZipInfo("payload/link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, b"../../outside")

    with pytest.raises(RuntimeError, match="edr_package_symlink_unsupported"):
        import_edr_package_from_zip(zip_path=zip_path, packages_root=tmp_path / "edr_packages")

    assert not (tmp_path / "outside").exists()


def test_import_edr_package_replaces_existing_symlink_children_without_target_write(
    tmp_path: Path,
) -> None:
    packages_root = tmp_path / "edr_packages"
    package_root = packages_root / "pkg1"
    package_root.mkdir(parents=True)
    outside_payload = tmp_path / "outside_payload"
    outside_payload.mkdir()
    (outside_payload / "marker.txt").write_text("keep", encoding="utf-8")
    outside_zip = tmp_path / "outside.edr.zip"
    outside_zip.write_bytes(b"do-not-overwrite")
    try:
        (package_root / EDR_PACKAGE_PAYLOAD_DIRNAME).symlink_to(
            outside_payload,
            target_is_directory=True,
        )
        (package_root / EDR_PACKAGE_ZIP_NAME).symlink_to(outside_zip)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    zip_path = tmp_path / "pkg1.edr.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            EDR_PACKAGE_MANIFEST_NAME,
            json.dumps({"package_id": "pkg1", "dataset_id": "demo"}),
        )

    summary = import_edr_package_from_zip(zip_path=zip_path, packages_root=packages_root)

    payload_root = package_root / EDR_PACKAGE_PAYLOAD_DIRNAME
    assert summary["id"] == "pkg1"
    assert not payload_root.is_symlink()
    assert (payload_root / EDR_PACKAGE_MANIFEST_NAME).exists()
    assert (outside_payload / "marker.txt").read_text(encoding="utf-8") == "keep"
    assert outside_zip.read_bytes() == b"do-not-overwrite"
    assert not (package_root / EDR_PACKAGE_ZIP_NAME).is_symlink()


def test_stage_tree_replaces_symlinked_destination_without_using_target(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "safe.txt").write_text("safe", encoding="utf-8")
    dest_root = tmp_path / "dest_root"
    dest_root.mkdir()
    outside = tmp_path / "outside_stage"
    outside.mkdir()
    (outside / EDR_PACKAGE_STAGE_META_NAME).write_text(
        json.dumps(
            {
                "package_id": "pkg1",
                "package_sha256": "sha",
                "kind": "classifier",
            }
        ),
        encoding="utf-8",
    )
    try:
        (dest_root / "pkg1__classifier").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _stage_tree_if_needed(
        src,
        dest_root / "pkg1__classifier",
        package_id="pkg1",
        package_sha256="sha",
        kind="classifier",
    )

    dest = dest_root / "pkg1__classifier"
    assert not dest.is_symlink()
    assert (dest / "safe.txt").read_text(encoding="utf-8") == "safe"
    assert not (outside / "safe.txt").exists()


def test_stage_tree_rejects_symlinked_stage_root_without_target_write(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "safe.txt").write_text("safe", encoding="utf-8")
    outside = tmp_path / "outside_stage"
    outside.mkdir()
    stage_root = tmp_path / "stage_root"
    try:
        stage_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(RuntimeError, match="edr_package_path_invalid"):
        _stage_tree_if_needed(
            src,
            stage_root / "pkg1__yolo",
            package_id="pkg1",
            package_sha256="sha",
            kind="yolo_run",
        )

    assert list(outside.iterdir()) == []


def test_stage_tree_rejects_symlinked_stage_parent_without_target_write(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "safe.txt").write_text("safe", encoding="utf-8")
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    stage_parent = tmp_path / "linked_parent"
    try:
        stage_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(RuntimeError, match="edr_package_path_invalid"):
        _stage_tree_if_needed(
            src,
            stage_parent / "yolo_runs" / "pkg1__yolo",
            package_id="pkg1",
            package_sha256="sha",
            kind="yolo_run",
        )

    assert list(outside.iterdir()) == []


def test_stage_tree_rejects_nested_symlinked_stage_parent_without_target_write(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "safe.txt").write_text("safe", encoding="utf-8")
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    stage_parent = tmp_path / "linked_parent"
    try:
        stage_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(RuntimeError, match="edr_package_path_invalid"):
        _stage_tree_if_needed(
            src,
            stage_parent / "nested" / "yolo_runs" / "pkg1__yolo",
            package_id="pkg1",
            package_sha256="sha",
            kind="yolo_run",
        )

    assert list(outside.iterdir()) == []


def test_edr_prepare_output_file_rejects_nested_symlinked_parent_before_mkdir(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(RuntimeError, match="edr_package_path_invalid"):
        _prepare_output_file(linked_parent / "nested" / "edr_packages" / "pkg1" / "package.meta.json")

    assert list(outside.iterdir()) == []


def test_edr_copy_file_rejects_nested_symlinked_parent_before_mkdir(
    tmp_path: Path,
) -> None:
    src = tmp_path / "source.bin"
    src.write_bytes(b"source")
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(RuntimeError, match="edr_package_path_invalid"):
        _copy_file(
            src,
            linked_parent / "nested" / "payload" / "source.bin",
            assets=[],
            kind="test_asset",
        )

    assert list(outside.iterdir()) == []


def test_edr_copy_tree_filtered_skips_internal_symlinked_parent(
    tmp_path: Path,
) -> None:
    src = tmp_path / "source_tree"
    src_nested = src / "linked_parent"
    src_nested.mkdir(parents=True)
    (src_nested / "source.bin").write_bytes(b"source")
    dest = tmp_path / "dest_tree"
    actual = dest / "actual"
    actual.mkdir(parents=True)
    try:
        (dest / "linked_parent").symlink_to(actual, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    copied = _copy_tree_filtered(src, dest)

    assert copied == []
    assert list(actual.iterdir()) == []


def test_zip_payload_skips_symlink_escape(tmp_path: Path) -> None:
    payload_root = tmp_path / "payload"
    payload_root.mkdir(parents=True, exist_ok=True)
    (payload_root / EDR_PACKAGE_MANIFEST_NAME).write_text(
        json.dumps({"package_id": "pkg1"}),
        encoding="utf-8",
    )
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        (payload_root / "escape.txt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    zip_path = tmp_path / "package.edr.zip"
    _zip_payload(payload_root, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
    assert EDR_PACKAGE_MANIFEST_NAME in names
    assert "escape.txt" not in names


def test_zip_payload_replaces_existing_zip_symlink_without_target_write(tmp_path: Path) -> None:
    payload_root = tmp_path / "payload"
    payload_root.mkdir(parents=True, exist_ok=True)
    (payload_root / EDR_PACKAGE_MANIFEST_NAME).write_text(
        json.dumps({"package_id": "pkg1"}),
        encoding="utf-8",
    )
    outside = tmp_path / "outside.edr.zip"
    outside.write_bytes(b"external")
    zip_path = tmp_path / "package.edr.zip"
    try:
        zip_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _zip_payload(payload_root, zip_path)

    assert not zip_path.is_symlink()
    assert outside.read_bytes() == b"external"
    with zipfile.ZipFile(zip_path, "r") as zf:
        assert EDR_PACKAGE_MANIFEST_NAME in set(zf.namelist())


def test_copy_tree_skips_symlink_escape(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "safe.txt").write_text("safe", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        (src / "escape.txt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    assets = []
    dest = tmp_path / "dest"
    _copy_tree(src, dest, assets=assets, kind="demo")

    assert (dest / "safe.txt").read_text(encoding="utf-8") == "safe"
    assert not (dest / "escape.txt").exists()
    assert [asset["path"] for asset in assets] == [str((dest / "safe.txt").as_posix())]


def test_copy2_if_different_replaces_symlink_to_source(tmp_path: Path) -> None:
    src = tmp_path / "source.bin"
    src.write_text("source", encoding="utf-8")
    dest = tmp_path / "dest.bin"
    try:
        dest.symlink_to(src)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _copy2_if_different(src, dest)

    assert not dest.is_symlink()
    assert dest.read_text(encoding="utf-8") == "source"


def test_write_json_replaces_symlink_targets_without_target_write(tmp_path: Path) -> None:
    json_path = tmp_path / "package" / EDR_PACKAGE_META_NAME
    json_path.parent.mkdir()
    outside_tmp = tmp_path / "outside_tmp.json"
    outside_final = tmp_path / "outside_final.json"
    outside_tmp.write_text("external tmp", encoding="utf-8")
    outside_final.write_text("external final", encoding="utf-8")
    tmp_link = json_path.with_suffix(json_path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp_link.symlink_to(outside_tmp)
        json_path.symlink_to(outside_final)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _write_json(json_path, {"id": "pkg1"})

    assert not tmp_link.exists()
    assert not json_path.is_symlink()
    assert json.loads(json_path.read_text(encoding="utf-8"))["id"] == "pkg1"
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp"
    assert outside_final.read_text(encoding="utf-8") == "external final"


def test_import_edr_package_rejects_invalid_zip(tmp_path: Path) -> None:
    zip_path = tmp_path / "broken.edr.zip"
    zip_path.write_bytes(b"not a zip")

    with pytest.raises(RuntimeError, match="edr_package_invalid_zip"):
        import_edr_package_from_zip(zip_path=zip_path, packages_root=tmp_path / "edr_packages")


def test_import_edr_package_rejects_oversize_uncompressed_total(tmp_path: Path) -> None:
    zip_path = tmp_path / "oversize.edr.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(EDR_PACKAGE_MANIFEST_NAME, json.dumps({"package_id": "pkg1"}))
        zf.writestr("payload/blob.bin", "x" * 2048)

    with pytest.raises(RuntimeError, match="edr_package_uncompressed_too_large"):
        import_edr_package_from_zip(
            zip_path=zip_path,
            packages_root=tmp_path / "edr_packages",
            max_extract_bytes=512,
        )

    assert not (tmp_path / "edr_packages" / "pkg1").exists()
