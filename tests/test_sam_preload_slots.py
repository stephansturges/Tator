import threading

import pytest
from fastapi import HTTPException

import localinferenceapi as api
from models.schemas import SamPreloadRequest


def _preload_job(request_id, *, slot="current", variant="sam1", generation=1):
    return api.SamPreloadJob(
        request_id=request_id,
        variant=variant,
        generation=generation,
        image_token=f"token-{request_id}",
        image_base64=None,
        image_name=f"image-{request_id}.jpg",
        slot=slot,
        event=threading.Event(),
    )


def test_sam_preload_rejects_disabled_background_slot():
    original_capacity = api.predictor_manager.get_capacity()
    api.predictor_manager.set_capacity(1)
    try:
        payload = SamPreloadRequest(
            image_token="missing-token",
            image_name="next.jpg",
            slot="next",
            sam_variant="sam1",
        )
        with pytest.raises(HTTPException) as exc:
            api.sam_preload(payload)
        assert exc.value.status_code == 409
        assert "slot_disabled:next" in str(exc.value.detail)
    finally:
        api.predictor_manager.set_capacity(original_capacity)


def test_sam_preload_rejects_unknown_slot():
    payload = SamPreloadRequest(
        image_token="missing-token",
        image_name="next.jpg",
        slot="nxt",
        sam_variant="sam1",
    )

    with pytest.raises(HTTPException) as exc:
        api.sam_preload(payload)

    assert exc.value.status_code == 409
    assert "slot_invalid:nxt" in str(exc.value.detail)


def test_sam_preload_request_supersession_is_slot_scoped():
    manager = api.SamPreloadManager()
    try:
        current_job = _preload_job(10, slot="current")
        next_job = _preload_job(11, slot="next")
        with manager.lock:
            key = manager._request_key(next_job.slot, next_job.variant)
            manager.latest_request_id[key] = next_job.request_id

        assert manager._is_superseded(current_job) is False
    finally:
        manager.stop()


def test_sam_preload_request_supersedes_older_same_slot_variant():
    manager = api.SamPreloadManager()
    try:
        old_job = _preload_job(10, slot="next")
        latest_job = _preload_job(11, slot="next")
        with manager.lock:
            key = manager._request_key(latest_job.slot, latest_job.variant)
            manager.latest_request_id[key] = latest_job.request_id

        assert manager._is_superseded(old_job) is True
    finally:
        manager.stop()


def test_sam_preload_generation_supersession_remains_variant_wide():
    manager = api.SamPreloadManager()
    try:
        old_generation_job = _preload_job(10, slot="current", generation=3)
        with manager.lock:
            manager.latest_generation[old_generation_job.variant] = 4

        assert manager._is_superseded(old_generation_job) is True
    finally:
        manager.stop()


def test_sam1_backend_can_select_mlx_adapter(monkeypatch):
    class DummyPredictor:
        def __init__(self):
            self.image_shape = None

        def set_image(self, np_img):
            self.image_shape = np_img.shape

        def predict(self, **kwargs):
            return "masks", "scores", "logits"

    monkeypatch.setattr(api, "SAM1_BACKEND_PREF", "mlx")
    monkeypatch.setattr(api, "should_use_mlx_sam", lambda preference: True)
    monkeypatch.setattr(api, "build_mlx_sam_predictor", DummyPredictor)

    backend = api._Sam1Backend()

    backend.set_image(api.np.zeros((4, 5, 3), dtype=api.np.uint8))
    assert backend.backend == "mlx"
    assert backend.predict() == ("masks", "scores", "logits")
