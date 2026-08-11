from __future__ import annotations

import json
import zipfile

from services.caption_artifacts import CaptionArtifactStore, sha256_file


def test_caption_artifact_store_keys_aliases_by_image_hash(tmp_path):
    store = CaptionArtifactStore(tmp_path / "store")
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "nested" / "b.jpg"
    image_b.parent.mkdir()
    payload = b"same-image-bytes"
    image_a.write_bytes(payload)
    image_b.write_bytes(payload)

    asset_a = store.register_image(
        image_a,
        aliases=[{"dataset_id": "ds1", "image_name": "a.jpg"}],
    )
    asset_b = store.register_image(
        image_b,
        aliases=[{"dataset_id": "ds2", "image_name": "b.jpg"}],
    )

    assert asset_a["image_sha256"] == sha256_file(image_a)
    assert asset_b["image_sha256"] == asset_a["image_sha256"]
    assert {alias["dataset_id"] for alias in asset_b["aliases"]} == {"ds1", "ds2"}


def test_caption_artifact_store_appends_artifacts_and_exports_set(tmp_path):
    store = CaptionArtifactStore(tmp_path / "store")
    image = tmp_path / "image.png"
    image.write_bytes(b"image-bytes")
    asset = store.register_image(image, aliases=[{"dataset_id": "ds", "image_name": "image.png"}])
    image_hash = asset["image_sha256"]
    context_hash = store.ensure_prompt_context({"labels": {"Building": 1}, "image_key": "train/image.png"})
    spec_hash = store.ensure_generation_spec({"provider": "openai", "model": "gpt-5.5", "qa": 8})
    attempt = store.create_attempt(
        image_sha256=image_hash,
        generation_spec_hash=spec_hash,
        prompt_context_hash=context_hash,
        run_id="run_1",
        provider="openai",
    )

    caption = store.append_artifact(
        image_sha256=image_hash,
        artifact_type="base_caption",
        payload={"caption": "A compact test caption."},
        generation_spec_hash=spec_hash,
        prompt_context_hash=context_hash,
        attempt_id=attempt["attempt_id"],
        caption_set_id="set_one",
    )
    qa = store.append_artifact(
        image_sha256=image_hash,
        artifact_type="qa_pair",
        payload={"question": "What is present?", "answer": "One building is present."},
        generation_spec_hash=spec_hash,
        prompt_context_hash=context_hash,
        attempt_id=attempt["attempt_id"],
        caption_set_id="set_one",
    )

    summary = store.summarize_images([image_hash], caption_set_id="set_one")

    assert summary["totals"]["base_caption_count"] == 1
    assert summary["totals"]["qa_pair_count"] == 1
    assert {artifact["artifact_id"] for artifact in summary["images"][0]["artifacts"]} == {
        caption["artifact_id"],
        qa["artifact_id"],
    }
    missing_set_summary = store.summarize_images([image_hash], caption_set_id="missing_set")
    assert missing_set_summary["totals"]["artifact_count"] == 0
    assert store.list_caption_sets()[0]["caption_set_id"] == "set_one"

    exported = store.export_caption_set(
        caption_set_id="set_one",
        image_sha256_values=[image_hash],
        output_dir=tmp_path / "exported",
    )

    assert exported["summary"]["artifact_count"] == 2
    with zipfile.ZipFile(exported["zip_path"]) as archive:
        names = set(archive.namelist())
        assert "caption_set_manifest.json" in names
        assert "caption_artifacts.jsonl" in names
        rows = [
            json.loads(line)
            for line in archive.read("caption_artifacts.jsonl").decode("utf-8").splitlines()
            if line.strip()
        ]
    assert len(rows) == 2
