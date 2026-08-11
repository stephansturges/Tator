from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from tools import run_class_split_experiments as harness


def _write_dataset(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    dataset_root = tmp_path / "waldo_v4"
    image_dir = dataset_root / "images"
    image_dir.mkdir(parents=True)

    # Minimal placeholder image row; content is irrelevant for dry-run validation.
    (image_dir / "sample_crop1.jpg").write_bytes(b"fake-jpeg-bytes")

    labelmap = tmp_path / "labelmap.txt"
    labelmap.write_text("LightVehicle\nPerson\n", encoding="utf-8")

    label_zip = tmp_path / "labels.zip"
    with zipfile.ZipFile(label_zip, "w") as zf:
        zf.writestr("sample_crop1.txt", "0 0.5 0.5 0.2 0.2\n")

    return dataset_root, image_dir, label_zip, labelmap


def test_matrix_default_class_count_by_mode() -> None:
    assert harness._matrix_default_class_count("minimum") == 2
    assert harness._matrix_default_class_count("remaining") == 4
    assert harness._matrix_default_class_count("finalists") == 8
    assert harness._matrix_default_class_count("cradio") == 2


def test_selected_classes_uses_explicit_and_default_lists() -> None:
    labelmap = ["A", "B", "C"]
    args = SimpleNamespace(classes="", selected_class_count=0, matrix="minimum")
    assert harness._selected_classes(args, labelmap) == ["A", "B"]

    args = SimpleNamespace(classes="C,B", selected_class_count=0, matrix="minimum")
    assert harness._selected_classes(args, labelmap) == ["C", "B"]

    args = SimpleNamespace(classes="Unknown", selected_class_count=0, matrix="minimum")
    with pytest.raises(SystemExit):
        harness._selected_classes(args, labelmap)


def test_matrix_builders_cover_expected_variants_and_scopes() -> None:
    classes = ["LightVehicle", "Person"]

    minimum = harness._minimum_matrix(sample_cap=123, classes=classes)
    assert len(minimum) == 6 * (len(classes) + 1)
    minimum_variants = {
        harness._variant_from_run_id(r["run_id"], r["class_name"], r["analysis_scope"]) for r in minimum
    }
    assert minimum_variants == {"B1", "B2", "B3", "B4", "B5", "B6"}
    minimum_by_variant = {
        run["variant"]: run
        for run in minimum
        if run["analysis_scope"] == "all_classes"
    }
    assert minimum_by_variant["B4"]["projection_neighbor_k"] == 15
    assert minimum_by_variant["B5"]["projection_neighbor_k"] == 50
    assert minimum_by_variant["B6"]["projection_neighbor_k"] == 0

    final = harness._finalist_matrix(sample_cap=123, classes=classes)
    assert len(final) == 5 * (len(classes) + 1)
    final_variants = {
        harness._variant_from_run_id(r["run_id"], r["class_name"], r["analysis_scope"]) for r in final
    }
    assert final_variants == {"fast", "balanced", "balanced_umap", "precise_tight_context", "precise_tight_context_umap"}

    cradio = harness._cradio_matrix(sample_cap=123, classes=classes)
    assert len(cradio) == 5 * (len(classes) + 1)
    assert all(r["encoder_type"] == "cradio" for r in cradio)
    assert all(r["encoder_model"] == harness.DEFAULT_CRADIO for r in cradio)
    cradio_variants = {
        harness._variant_from_run_id(r["run_id"], r["class_name"], r["analysis_scope"]) for r in cradio
    }
    assert cradio_variants == {
        "cradio_summary_pca",
        "cradio_summary_umap",
        "cradio_spatial_mean_pca",
        "cradio_summary_spatial_concat_pca",
        "cradio_precise_tight_context_pca",
    }

    remaining = harness._remaining_lever_matrix(sample_cap=123, classes=classes)
    assert len(remaining) == 25 * (len(classes) + 1)
    remaining_variants = {
        harness._variant_from_run_id(r["run_id"], r["class_name"], r["analysis_scope"]) for r in remaining
    }
    for required in {"baseline", "fast_native_raw", "umap15", "umap50", "clip_vitb32", "cradio_summary", "cradio_spatial_mean"}:
        assert required in remaining_variants


def test_manifest_records_dataset_lock_stats_and_representative_classes(tmp_path) -> None:
    dataset_root, image_dir, label_zip, labelmap_path = _write_dataset(tmp_path)
    labelmap = harness._read_labelmap(labelmap_path)
    manifest = harness._build_manifest(
        image_dir=image_dir,
        label_zip=label_zip,
        labelmap=labelmap,
    )
    stats = manifest["dataset_stats"]
    assert stats["image_file_count"] == 1
    assert stats["label_file_count"] == 1
    assert stats["total_object_count"] == 1
    assert stats["malformed_label_line_count"] == 0
    assert stats["class_counts"] == {"LightVehicle": 1, "Person": 0}
    assert stats["class_bbox_area_medians"]["LightVehicle"] == pytest.approx(0.04)

    args = SimpleNamespace(classes="", selected_class_count=1, matrix="minimum")
    assert harness._selected_classes(args, labelmap, manifest) == ["LightVehicle"]


def test_run_ids_are_path_safe_and_unique() -> None:
    safe = harness._run_id("B1", "../A/B")
    assert "/" not in safe
    assert ".." not in safe
    assert safe.startswith("B1_A_B_")
    with pytest.raises(ValueError, match="duplicate experiment run ids"):
        harness._validate_unique_run_ids([{"run_id": "same"}, {"run_id": "same"}])


def test_stage_workspace_stays_inside_class_analysis_storage(tmp_path, monkeypatch) -> None:
    api = harness._get_api()
    class_root = tmp_path / "class_analysis"
    image_dir = tmp_path / "source_images"
    image_dir.mkdir()
    (image_dir / "sample.jpg").write_bytes(b"image-bytes")
    output_root = class_root / "experiments" / "smoke"
    output_root.mkdir(parents=True)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    manifest = {
        "labelmap": ["A"],
        "images": [{"image_relpath": "sample.jpg", "label_lines": []}],
    }

    workspace = harness._stage_workspace(
        manifest=manifest,
        image_dir=image_dir,
        output_root=output_root,
    )

    assert workspace == output_root / "_workspace"
    assert (workspace / "images" / "sample.jpg").read_bytes() == b"image-bytes"
    assert json.loads((workspace / "manifest.json").read_text())["labelmap"] == ["A"]
    with pytest.raises(ValueError, match="must be inside"):
        harness._stage_workspace(
            manifest=manifest,
            image_dir=image_dir,
            output_root=tmp_path / "outside",
        )


def test_run_one_writes_complete_artifacts_and_resumes(tmp_path, monkeypatch) -> None:
    api = harness._get_api()
    records = [
        {
            "point_id": f"p{idx}",
            "class_name": "A" if idx < 3 else "B",
            "width": 10 + idx,
            "height": 8 + idx,
            "crop_xyxy": [0, 0, 16 + idx, 16 + idx],
            "bbox_xyxy": [1, 1, 11 + idx, 9 + idx],
            "image_relpath": f"image-{idx}.jpg",
            "kind": "bbox",
        }
        for idx in range(6)
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
            [0.2, 0.8, 0.0],
        ],
        dtype=np.float32,
    )
    calls = {"collect": 0}

    def fake_collect(request, *, job, out_dir):
        calls["collect"] += 1
        return records, [], {
            "analysis_scope": "all_classes",
            "object_count": len(records),
            "raw_object_count": len(records),
            "sample_cap": 0,
        }

    monkeypatch.setattr(api, "_class_analysis_collect_records", fake_collect)
    monkeypatch.setattr(
        api,
        "_class_analysis_encode_crops",
        lambda *args, **kwargs: embeddings,
    )
    run = harness._minimum_matrix(sample_cap=0, classes=[])[2]
    run.update(
        {
            "run_id": "execution_smoke_all_classes",
            "variant": "execution_smoke",
            "projection": "pca",
            "projection_mode": "global_pca",
            "projection_preprocess": "zscore",
        }
    )
    output_root = tmp_path / "output"
    row = harness._run_one(
        run,
        manifest={"labelmap": ["A", "B"], "images": []},
        image_dir=tmp_path,
        output_root=output_root,
        force=False,
    )
    run_dir = output_root / run["run_id"]
    for name in (
        "result.json",
        "metrics.json",
        "config.json",
        "embeddings.npz",
        "metadata.jsonl",
        "review_queue.jsonl",
    ):
        assert (run_dir / name).is_file()
    assert row["projection_preprocess"] == "zscore"
    assert row["projection_trustworthiness"] is not None
    assert calls["collect"] == 1

    resumed = harness._run_one(
        run,
        manifest={"labelmap": ["A", "B"], "images": []},
        image_dir=tmp_path,
        output_root=output_root,
        force=False,
    )
    assert resumed["run_id"] == row["run_id"]
    assert calls["collect"] == 1


def test_main_dry_run_is_deterministic_and_respects_args(tmp_path, monkeypatch, capsys) -> None:
    dataset_root, image_dir, label_zip, labelmap = _write_dataset(tmp_path)
    output_root = tmp_path / "experiments"

    monkeypatch.setattr(harness, "_run_one", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unexpected run execution")))

    argv = [
        "run_class_split_experiments.py",
        "--dataset-root",
        str(dataset_root),
        "--label-zip",
        str(label_zip),
        "--labelmap",
        str(labelmap),
        "--image-dir",
        str(image_dir),
        "--output-root",
        str(output_root),
        "--matrix",
        "cradio",
        "--classes",
        "LightVehicle",
        "--sample-cap",
        "11",
        "--dry-run",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    harness.main()

    rows = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(rows) == 10
    first, second = rows[0], rows[1]
    assert first.startswith("cradio_summary_pca_LightVehicle")
    assert first.endswith("}")

    first_payload = json.loads(first.split(" ", 1)[1])
    assert first_payload["sample_cap"] == 11
    assert first_payload["analysis_scope"] == "selected_class"
    assert first_payload["class_name"] == "LightVehicle"
    second_payload = json.loads(rows[1].split(" ", 1)[1])
    assert second_payload["class_name"] == "LightVehicle"
    assert second_payload["analysis_scope"] == "selected_class"
