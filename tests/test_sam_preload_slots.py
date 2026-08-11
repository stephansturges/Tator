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
            key = manager._request_key(next_job.slot)
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
            key = manager._request_key(latest_job.slot)
            manager.latest_request_id[key] = latest_job.request_id

        assert manager._is_superseded(old_job) is True
    finally:
        manager.stop()


def test_sam_preload_request_supersedes_older_cross_variant_same_slot():
    manager = api.SamPreloadManager()
    try:
        old_job = _preload_job(10, slot="current", variant="sam3")
        latest_job = _preload_job(11, slot="current", variant="sam1")
        with manager.lock:
            manager.latest_request_id[manager._request_key(latest_job.slot)] = latest_job.request_id

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


def test_sam3_backend_auto_selects_mlx_and_warms_decoder_once(monkeypatch):
    class DummyPredictor:
        runtime_notice = "mlx notice"

        def __init__(self):
            self.model = object()
            self.set_calls = 0
            self.predict_calls = 0

        def set_image(self, _np_img):
            self.set_calls += 1

        def predict(self, **_kwargs):
            self.predict_calls += 1
            return "masks", "scores", "logits"

    predictor = DummyPredictor()
    synchronizations = []
    monkeypatch.setattr(api, "SAM3_BACKEND_PREF", "auto")
    monkeypatch.setattr(api, "SAM_DECODER_WARMUP", True)
    monkeypatch.setattr(api, "_clear_missing_active_sam3_checkpoint", lambda: None)
    monkeypatch.setattr(api, "_active_sam3_model_supports_mlx", lambda: True)
    monkeypatch.setattr(api, "should_use_mlx_sam3", lambda _preference, _runtime="auto": True)
    monkeypatch.setattr(api, "build_mlx_sam3_predictor", lambda _runtime="auto": predictor)
    monkeypatch.setattr(
        api,
        "_synchronize_sam_accelerator",
        lambda backend, device: synchronizations.append((backend, device)),
    )

    backend = api._Sam3Backend()
    image = api.np.zeros((8, 12, 3), dtype=api.np.uint8)
    backend.set_image(image)
    backend.set_image(image)

    assert backend.backend == "mlx"
    assert backend.device == "metal"
    assert backend.runtime_notice == "mlx notice"
    assert predictor.set_calls == 2
    assert predictor.predict_calls == 1
    assert synchronizations == [("mlx", "metal"), ("mlx", "metal"), ("mlx", "metal")]


def test_sam3_checkpoint_identity_reaches_backend_and_slot_cache(monkeypatch):
    runtimes = []

    class DummyBackend:
        backend = "mlx"
        device = "metal"
        runtime_notice = "test"

        def __init__(self, runtime="auto"):
            runtimes.append(runtime)

        def set_image(self, _image):
            return None

        def unload(self):
            return None

    slot = api.PredictorSlot("current")
    monkeypatch.setattr(api, "_Sam3Backend", DummyBackend)
    image = api.np.zeros((4, 5, 3), dtype=api.np.uint8)

    slot.set_image(image, "token", "sam3@mlx-8bit", "image.jpg")
    slot.set_image(image, "token", "sam3@mlx-mxfp4", "image.jpg")

    assert runtimes == ["mlx-8bit", "mlx-mxfp4"]
    assert slot.variant == "sam3@mlx-mxfp4"
    assert tuple(slot.backends) == ("sam3@mlx-mxfp4",)


def test_default_variant_canonicalizes_sam3_runtime_identity():
    assert api._default_variant("sam3") == "sam3@auto"
    assert api._default_variant("sam3@8bit") == "sam3@mlx-8bit"
    assert api._base_sam_variant("sam3@mlx-mxfp4") == "sam3"


def test_explicit_mlx_rejects_custom_sam3_checkpoint(monkeypatch):
    monkeypatch.setattr(api, "SAM3_BACKEND_PREF", "mlx")
    monkeypatch.setattr(api, "_clear_missing_active_sam3_checkpoint", lambda: None)
    monkeypatch.setattr(api, "_active_sam3_model_supports_mlx", lambda: False)

    with pytest.raises(RuntimeError, match="mlx_sam3_custom_checkpoint_unsupported"):
        api._Sam3Backend()
