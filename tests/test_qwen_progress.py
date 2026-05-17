from fastapi import HTTPException
from PIL import Image

import localinferenceapi as api


def _reset_qwen_progress() -> None:
    with api.qwen_progress_lock:
        api.qwen_progress_state.clear()
        api.qwen_progress_state.update(
            {
                "run_id": None,
                "active": False,
                "kind": None,
                "phase": "idle",
                "phase_label": "Idle",
                "progress": 0.0,
                "message": "",
                "model_id": None,
                "platform": None,
                "local": None,
                "partial": None,
                "needs_download": None,
                "cache_path": None,
                "loaded": False,
                "input_tokens": None,
                "generated_tokens": 0,
                "max_new_tokens": None,
                "token_preview": "",
                "started_at": None,
                "updated_at": None,
                "completed_at": None,
                "error": None,
            }
        )


def test_qwen_infer_prompt_render_failure_marks_progress_error(monkeypatch):
    _reset_qwen_progress()

    def fake_resolve_image_payload(*args, **kwargs):
        pil_img = Image.new("RGB", (8, 8), color=(128, 128, 128))
        return pil_img, api.np.zeros((8, 8, 3), dtype=api.np.uint8), "token-1"

    def fake_render_prompt(*args, **kwargs):
        raise HTTPException(status_code=422, detail="prompt_config_broken")

    monkeypatch.setattr(api, "resolve_image_payload", fake_resolve_image_payload)
    monkeypatch.setattr(api, "_render_qwen_prompt_impl", fake_render_prompt)

    payload = api.QwenInferenceRequest(
        image_base64="stub",
        item_list="car",
        prompt_type="bbox",
    )

    try:
        api.qwen_infer(payload)
    except HTTPException as exc:
        assert exc.detail == "prompt_config_broken"
    else:
        raise AssertionError("qwen_infer should have raised")

    progress = api.qwen_progress()
    assert progress["active"] is False
    assert progress["phase"] == "error"
    assert progress["error"] == "Qwen prompt rendering failed"


def test_qwen_progress_token_updates_preview_and_count():
    _reset_qwen_progress()
    api._qwen_progress_start(
        kind="caption",
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        platform=api.QWEN_PLATFORM_TRANSFORMERS,
        message="test",
        max_new_tokens=10,
    )

    api._qwen_progress_token("A", generated_tokens=1, max_new_tokens=10)
    api._qwen_progress_token(" caption", generated_tokens=2, max_new_tokens=10)

    progress = api.qwen_progress()
    assert progress["active"] is True
    assert progress["phase"] == "generate"
    assert progress["generated_tokens"] == 2
    assert progress["token_preview"] == "A caption"
