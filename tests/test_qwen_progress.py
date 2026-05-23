import json

import pytest
from fastapi import HTTPException
from PIL import Image

import localinferenceapi as api


def _reset_qwen_progress() -> None:
    api.qwen_cancel_event.clear()
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
                "cancel_requested": False,
                "cancel_force": False,
                "step_id": None,
                "step_index": None,
                "step_total": None,
                "step_label": None,
                "step_detail": None,
                "step_region": None,
                "step_plan": [],
                "live_output": "",
                "log_lines": [],
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
    assert progress["live_output"] == "A caption"


def test_qwen_progress_step_plan_metadata_and_region():
    _reset_qwen_progress()
    api._qwen_progress_start(
        kind="caption",
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        platform=api.QWEN_PLATFORM_TRANSFORMERS,
        message="test",
        max_new_tokens=10,
    )
    plan = [
        {"id": "prepare", "label": "Prepare"},
        {"id": "load_model", "label": "Load model"},
        {"id": "window_1", "label": "Caption window 1/1"},
    ]
    api._qwen_progress_update(
        step_plan=plan,
        step_id="window_1",
        step_region={
            "x": 10,
            "y": 20,
            "width": 100,
            "height": 120,
            "image_name": "sample.jpg",
        },
        message="Captioning window",
    )

    progress = api.qwen_progress()
    assert progress["step_id"] == "window_1"
    assert progress["step_index"] == 3
    assert progress["step_total"] == 3
    assert progress["step_label"] == "Caption window 1/1"
    assert len(progress["step_plan"]) == 3
    assert progress["step_region"]["x"] == 10.0
    assert progress["step_region"]["image_name"] == "sample.jpg"


def test_qwen_caption_step_plan_describes_windowed_two_stage_flow():
    plan = api._build_qwen_caption_step_plan(
        caption_mode="windowed",
        total_windows=4,
        two_stage=True,
        is_thinking=True,
        force_unload=True,
        caption_model_id="caption-model",
        refinement_model_id="refine-model",
    )
    ids = [entry["id"] for entry in plan]
    assert ids[:2] == ["prepare", "load_model"]
    assert "window_4" in ids
    assert "draft_caption" in ids
    assert "refine_draft" in ids
    assert ids[-2:] == ["unload_model", "finalize"]


def test_qwen_caption_cancel_marks_progress_and_new_run_clears_event():
    _reset_qwen_progress()
    first_run = api._qwen_progress_start(
        kind="caption",
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        platform=api.QWEN_PLATFORM_TRANSFORMERS,
        message="test",
        max_new_tokens=10,
    )

    result = api.cancel_qwen_caption(force=False)

    assert result["cancelled"] is True
    assert result["run_id"] == first_run
    assert api.qwen_cancel_event.is_set()
    progress = api.qwen_progress()
    assert progress["active"] is True
    assert progress["phase"] == "cancelling"
    assert progress["cancel_requested"] is True
    assert progress["cancel_force"] is False

    second_run = api._qwen_progress_start(
        kind="caption",
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        platform=api.QWEN_PLATFORM_TRANSFORMERS,
        message="next",
        max_new_tokens=10,
    )

    assert second_run != first_run
    assert not api.qwen_cancel_event.is_set()
    assert api.qwen_progress()["cancel_requested"] is False


def test_qwen_prepass_progress_uses_caption_token_budget(monkeypatch):
    _reset_qwen_progress()
    monkeypatch.setattr(
        api,
        "_run_prepass_annotation",
        lambda payload: {"detections": [], "warnings": [], "image_token": payload.image_token},
    )

    api.qwen_prepass(
        api.QwenPrepassRequest(
            image_base64="stub",
            prepass_caption=True,
            prepass_caption_profile="deep",
            prepass_caption_max_tokens=99999,
        )
    )

    progress = api.qwen_progress()
    assert progress["max_new_tokens"] == 2000
    assert progress["phase"] == "complete"


def test_qwen_prepass_progress_uses_caption_profile_default(monkeypatch):
    _reset_qwen_progress()
    monkeypatch.setattr(
        api,
        "_run_prepass_annotation",
        lambda payload: {"detections": [], "warnings": [], "image_token": payload.image_token},
    )

    api.qwen_prepass(
        api.QwenPrepassRequest(
            image_base64="stub",
            prepass_caption=True,
            prepass_caption_profile="light",
        )
    )

    assert api.qwen_progress()["max_new_tokens"] == 512


def test_qwen_caption_io_input_does_not_fail_on_bad_token_count(monkeypatch, tmp_path):
    _reset_qwen_progress()
    monkeypatch.setattr(api, "LOG_ROOT", tmp_path)
    api._qwen_progress_start(
        kind="caption",
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        platform=api.QWEN_PLATFORM_TRANSFORMERS,
        message="test",
        max_new_tokens=10,
    )

    api._qwen_caption_io_input(
        call_id="call-1",
        source="test",
        model_id="model",
        max_new_tokens="not-an-int",
    )

    latest_jsonl = tmp_path / "qwen_caption_io_latest.jsonl"
    records = [json.loads(line) for line in latest_jsonl.read_text(encoding="utf-8").splitlines()]
    assert records[-1]["event"] == "input"
    assert records[-1]["max_new_tokens"] is None


def test_qwen_caption_io_readable_failure_does_not_break_request(monkeypatch, tmp_path):
    _reset_qwen_progress()
    monkeypatch.setattr(api, "LOG_ROOT", tmp_path)
    monkeypatch.setattr(
        api,
        "_qwen_caption_io_readable",
        lambda _record: (_ for _ in ()).throw(RuntimeError("format failed")),
    )
    api._qwen_progress_start(
        kind="caption",
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        platform=api.QWEN_PLATFORM_TRANSFORMERS,
        message="test",
        max_new_tokens=10,
    )

    api._qwen_caption_io_record({"event": "input", "call_id": "call-2"})

    latest_jsonl = tmp_path / "qwen_caption_io_latest.jsonl"
    records = [json.loads(line) for line in latest_jsonl.read_text(encoding="utf-8").splitlines()]
    assert records[-1]["event"] == "input"


def test_qwen_caption_io_stops_after_caption_progress_finishes(monkeypatch, tmp_path):
    _reset_qwen_progress()
    monkeypatch.setattr(api, "LOG_ROOT", tmp_path)
    api._qwen_progress_start(
        kind="caption",
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        platform=api.QWEN_PLATFORM_TRANSFORMERS,
        message="test",
        max_new_tokens=10,
    )
    latest_jsonl = tmp_path / "qwen_caption_io_latest.jsonl"
    before = latest_jsonl.read_text(encoding="utf-8")

    api._qwen_progress_finish("done", token_preview="caption")
    api._qwen_caption_io_output(
        call_id="stale-call",
        source="test",
        model_id="model",
        output_text="stale",
    )

    assert latest_jsonl.read_text(encoding="utf-8") == before


def test_qwen_caption_io_reset_replaces_symlinked_latest_logs(monkeypatch, tmp_path):
    _reset_qwen_progress()
    monkeypatch.setattr(api, "LOG_ROOT", tmp_path)
    outside_jsonl = tmp_path / "outside_latest.jsonl"
    outside_text = tmp_path / "outside_latest.log"
    outside_jsonl.write_text("external jsonl\n", encoding="utf-8")
    outside_text.write_text("external text\n", encoding="utf-8")
    latest_jsonl = tmp_path / "qwen_caption_io_latest.jsonl"
    latest_text = tmp_path / "qwen_caption_io_latest.log"
    try:
        latest_jsonl.symlink_to(outside_jsonl)
        latest_text.symlink_to(outside_text)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    api._qwen_progress_start(
        kind="caption",
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        platform=api.QWEN_PLATFORM_TRANSFORMERS,
        message="test",
        max_new_tokens=10,
    )

    assert outside_jsonl.read_text(encoding="utf-8") == "external jsonl\n"
    assert outside_text.read_text(encoding="utf-8") == "external text\n"
    assert not latest_jsonl.is_symlink()
    assert not latest_text.is_symlink()
    assert json.loads(latest_jsonl.read_text(encoding="utf-8").splitlines()[-1])["event"] == "run_start"


def test_qwen_caption_io_append_replaces_symlinked_run_log(monkeypatch, tmp_path):
    _reset_qwen_progress()
    monkeypatch.setattr(api, "LOG_ROOT", tmp_path)
    run_id = api._qwen_progress_start(
        kind="caption",
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        platform=api.QWEN_PLATFORM_TRANSFORMERS,
        message="test",
        max_new_tokens=10,
    )
    run_jsonl = tmp_path / "qwen_caption_io" / f"{run_id}.jsonl"
    outside_jsonl = tmp_path / "outside_run.jsonl"
    outside_jsonl.write_text("external run\n", encoding="utf-8")
    try:
        run_jsonl.unlink()
        run_jsonl.symlink_to(outside_jsonl)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    api._qwen_caption_io_record({"event": "input", "call_id": "call-linked"})

    assert outside_jsonl.read_text(encoding="utf-8") == "external run\n"
    assert not run_jsonl.is_symlink()
    records = [json.loads(line) for line in run_jsonl.read_text(encoding="utf-8").splitlines()]
    assert records[-1]["event"] == "input"
