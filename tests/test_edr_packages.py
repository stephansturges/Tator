from __future__ import annotations

import json
from pathlib import Path

from services.edr_packages import (
    EDR_PACKAGE_MANIFEST_NAME,
    EDR_PACKAGE_META_NAME,
    EDR_PACKAGE_PAYLOAD_DIRNAME,
    resolve_edr_package_runtime,
)


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


def test_resolve_edr_package_runtime_repairs_feature_contract_from_source_features(tmp_path: Path) -> None:
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

    assert runtime["feature_contract"]["embed_proj_dim"] == 1024
    assert runtime["feature_contract"]["feature_schema_hash"] == "abc123"
    assert runtime["labelmap"] == ["person"]
