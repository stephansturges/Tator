import json
import shutil
import uuid
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from .helpers.api import api_json, api_multipart
from .helpers.ui import go_to_tab, open_datasets_tab


pytestmark = [pytest.mark.ui, pytest.mark.ui_smoke]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _fixture_root() -> Path:
    return _repo_root() / "tests" / "fixtures" / "fuzz_pack"


def _make_yolo_zip_bytes() -> bytes:
    image_a = _fixture_root() / "images" / "img_0.png"
    image_b = _fixture_root() / "images" / "img_1.png"
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("labelmap.txt", "building\n")
        zf.writestr("train/images/img_0.png", image_a.read_bytes())
        zf.writestr("train/images/img_1.png", image_b.read_bytes())
        zf.writestr("train/labels/img_0.txt", "0 0.5 0.5 0.4 0.4\n")
        zf.writestr("train/labels/img_1.txt", "0 0.5 0.5 0.3 0.3\n")
    return payload.getvalue()


def _upload_test_dataset(dataset_id: str) -> dict:
    return api_multipart(
        "POST",
        "/datasets/upload",
        fields={"dataset_id": dataset_id, "dataset_type": "bbox"},
        files={"file": (f"{dataset_id}.zip", _make_yolo_zip_bytes(), "application/zip")},
    )


def _safe_remove_test_artifacts(dataset_id: str) -> None:
    if not dataset_id.startswith("pw_ui_"):
        raise AssertionError(f"Refusing to clean non-test dataset id: {dataset_id}")
    datasets_root = (_repo_root() / "uploads" / "datasets").resolve()

    def remove_path(path: Path, allowed_root: Path) -> None:
        raw = Path(path)
        if not raw.exists() and not raw.is_symlink():
            return
        resolved = raw.resolve(strict=False)
        allowed = allowed_root.resolve(strict=False)
        if resolved != allowed and allowed not in resolved.parents:
            raise AssertionError(f"Refusing to clean outside test root: {raw}")
        if raw.is_symlink() or raw.is_file():
            raw.unlink()
        else:
            shutil.rmtree(raw)

    remove_path(datasets_root / dataset_id, datasets_root)
    trash_root = datasets_root / ".trash"
    if not trash_root.exists():
        return
    for child in trash_root.iterdir():
        should_remove = child.name.startswith(dataset_id)
        meta_path = child / "deleted_dataset.json"
        if not should_remove and meta_path.exists() and not meta_path.is_symlink():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            should_remove = str(meta.get("original_id") or meta.get("id") or "").startswith(dataset_id)
        if should_remove:
            remove_path(child, trash_root)


def _cleanup_dataset(dataset_id: str) -> None:
    try:
        api_json("DELETE", f"/datasets/{dataset_id}", expected_statuses=(200, 404))
    except Exception:
        pass
    _safe_remove_test_artifacts(dataset_id)


# CASE_ID: DATASET_MANAGED_TRASH_RESTORE_UI

def test_managed_dataset_delete_moves_to_trash_and_restore_recovers(playwright_page):
    page, _ = playwright_page
    dataset_id = f"pw_ui_ds_{uuid.uuid4().hex[:8]}"
    _upload_test_dataset(dataset_id)
    try:
        open_datasets_tab(page)
        page.click("#datasetListRefreshTop")
        card = page.locator('[data-testid="card.datasets.entry"]').filter(has_text=dataset_id)
        card.first.wait_for(timeout=45000)

        page.once("dialog", lambda dialog: dialog.accept())
        card.first.locator('[data-testid="action.datasets.card.delete"]').click()
        page.wait_for_function(
            """
(datasetId) => !(Array.from(document.querySelectorAll('[data-testid="card.datasets.entry"]'))
  .some((card) => card.getAttribute('data-dataset-id') === datasetId))
""",
            arg=dataset_id,
            timeout=45000,
        )
        trash_card = page.locator('[data-testid="card.datasets.trash_entry"]').filter(has_text=dataset_id)
        trash_card.first.wait_for(timeout=45000)

        page.once("dialog", lambda dialog: dialog.accept())
        trash_card.first.locator('[data-testid="action.datasets.trash.restore"]').click()
        restored = page.locator('[data-testid="card.datasets.entry"]').filter(has_text=dataset_id)
        restored.first.wait_for(timeout=45000)
        page.wait_for_function(
            """
(datasetId) => !(Array.from(document.querySelectorAll('[data-testid="card.datasets.trash_entry"]'))
  .some((card) => card.textContent.includes(datasetId)))
""",
            arg=dataset_id,
            timeout=45000,
        )
    finally:
        _cleanup_dataset(dataset_id)


# CASE_ID: DATA_INGESTION_REFERENCE_PROFILE_GATING

def test_data_ingestion_requires_matching_backend_reference_profile(playwright_page, tmp_path):
    page, _ = playwright_page
    dataset_id = f"pw_ui_ingest_{uuid.uuid4().hex[:8]}"
    head_id = f"{dataset_id}_profile"
    _upload_test_dataset(dataset_id)
    candidate_path = tmp_path / "candidate.png"
    candidate_path.write_bytes((_fixture_root() / "images" / "img_2.png").read_bytes())
    posted_payloads: list[str] = []

    def fulfill_capabilities(route):
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "ok",
                    "default_encoder": "local_salad",
                    "default_profile_policy": "local_training_only",
                    "local_salad_heads": [
                        {
                            "id": head_id,
                            "label": "Playwright reference profile",
                            "status": "available",
                            "reference_source": "backend_dataset",
                            "source_mode": "backend_dataset",
                            "reference_dataset_id": dataset_id,
                            "reference_dataset_label": dataset_id,
                            "train_image_count": 2,
                            "encoder_type": "dinov3",
                        }
                    ],
                    "cradio_models": ["nvidia/C-RADIOv4-SO400M"],
                    "default_cradio_model": "nvidia/C-RADIOv4-SO400M",
                }
            ),
        )

    def fulfill_start_job(route):
        # Tab initialization probes the same collection URL with GET to recover active jobs.
        # Keep that separate from the POST body assertion below.
        if route.request.method != "POST":
            route.fulfill(status=200, content_type="application/json", body="[]")
            return
        body = route.request.post_data_buffer or b""
        posted_payloads.append(body.decode("utf-8", errors="replace"))
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"job_id": "pw_ingest_job"}),
        )

    def fulfill_job(route):
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "job_id": "pw_ingest_job",
                    "status": "completed",
                    "progress": 1,
                    "message": "done",
                    "kind": "data_ingestion",
                    "summary": {},
                }
            ),
        )

    def fulfill_result(route):
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "summary": {
                        "head_id": head_id,
                        "reference_source": "backend_dataset",
                        "reference_dataset_label": dataset_id,
                        "candidate_image_count": 1,
                        "reference_count": 2,
                        "keep_fraction": 0.2,
                        "selected_count": 1,
                    },
                    "items": [],
                }
            ),
        )

    page.route("**/data_ingestion/capabilities*", fulfill_capabilities)
    page.route("**/data_ingestion/jobs", fulfill_start_job)
    page.route("**/data_ingestion/jobs/pw_ingest_job", fulfill_job)
    page.route("**/data_ingestion/jobs/pw_ingest_job/result", fulfill_result)
    try:
        go_to_tab(page, "#tabDataIngestionButton", "#tabDataIngestion")
        page.wait_for_selector("#dataIngestionAnalyzeButton", timeout=15000)

        assert page.eval_on_selector("#dataIngestionReferenceActive", "el => !!el.checked") is True
        assert page.eval_on_selector("#dataIngestionReferenceDataset", "el => !!el.disabled") is True
        assert page.eval_on_selector("#dataIngestionAnalyzeButton", "el => !!el.disabled") is True

        page.check("#dataIngestionReferenceBackend")
        page.click("#dataIngestionRefreshButton")
        page.wait_for_function(
            """
(datasetId) => Array.from(document.querySelectorAll('#dataIngestionReferenceDataset option'))
  .some((option) => option.value === datasetId)
""",
            arg=dataset_id,
            timeout=45000,
        )
        page.select_option("#dataIngestionReferenceDataset", dataset_id)
        page.wait_for_function(
            """
(headId) => Array.from(document.querySelectorAll('#dataIngestionSaladHead option'))
  .some((option) => option.value === headId)
""",
            arg=head_id,
            timeout=15000,
        )
        page.select_option("#dataIngestionSaladHead", head_id)
        assert page.eval_on_selector("#dataIngestionReferenceDataset", "el => !!el.disabled") is False
        assert page.eval_on_selector("#dataIngestionBuildProfileButton", "el => !!el.disabled") is False
        assert page.eval_on_selector("#dataIngestionAnalyzeButton", "el => !!el.disabled") is True

        page.select_option("#dataIngestionTrainEncoder", "cradio")
        assert page.eval_on_selector("#dataIngestionTrainCradioModel", "el => !!el.disabled") is False
        page.select_option("#dataIngestionTrainEncoder", "dinov3")
        assert page.eval_on_selector("#dataIngestionTrainCradioModel", "el => !!el.disabled") is True

        page.set_input_files("#dataIngestionFiles", str(candidate_path))
        page.wait_for_function(
            "!document.querySelector('#dataIngestionAnalyzeButton')?.disabled",
            timeout=15000,
        )
        page.click("#dataIngestionAnalyzeButton")
        page.wait_for_function(
            "document.querySelector('#dataIngestionStatus')?.textContent?.toLowerCase().includes('completed')",
            timeout=15000,
        )
        assert posted_payloads, "Data Ingestion analysis request was not posted"
        posted = posted_payloads[0]
        assert f'"reference_dataset_id":"{dataset_id}"' in posted
        assert '"reference_source":"backend_dataset"' in posted
        assert 'name="candidate_files"' in posted
        assert 'name="reference_files"' not in posted
    finally:
        page.unroute("**/data_ingestion/capabilities*", fulfill_capabilities)
        page.unroute("**/data_ingestion/jobs", fulfill_start_job)
        page.unroute("**/data_ingestion/jobs/pw_ingest_job", fulfill_job)
        page.unroute("**/data_ingestion/jobs/pw_ingest_job/result", fulfill_result)
        _cleanup_dataset(dataset_id)
