import pytest

import localinferenceapi as api


class _NoThread:
    def __init__(self, *args, **kwargs):
        raise AssertionError("worker thread should not be created for invalid job input")


class _NoopThread:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass


class _FakeMpsBackend:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class _FakeTorch:
    def __init__(self, *, cuda=False, mps=False):
        self.cuda = type(
            "Cuda",
            (),
            {
                "is_available": staticmethod(lambda: cuda),
                "device_count": staticmethod(lambda: 1 if cuda else 0),
            },
        )()
        self.backends = type("Backends", (), {"mps": _FakeMpsBackend(mps)})()


def test_auto_label_missing_dataset_rejects_before_queue(monkeypatch):
    with api.AUTO_LABEL_JOBS_LOCK:
        api.AUTO_LABEL_JOBS.clear()

    def missing_dataset(_dataset_id):
        raise api.HTTPException(status_code=404, detail="dataset_not_found")

    monkeypatch.setattr(api, "_resolve_dataset_entry", missing_dataset)
    monkeypatch.setattr(api.threading, "Thread", _NoThread)

    with pytest.raises(api.HTTPException) as exc_info:
        api.start_auto_label_job(
            api.AutoLabelRequest(
                dataset_id="missing_dataset",
                enable_yolo=False,
                enable_rfdetr=False,
                enable_falcon=False,
                max_images=1,
            )
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "dataset_not_found"
    assert api.AUTO_LABEL_JOBS == {}


def test_prompt_helper_missing_dataset_rejects_before_queue(monkeypatch):
    with api.PROMPT_HELPER_JOBS_LOCK:
        api.PROMPT_HELPER_JOBS.clear()

    def missing_dataset(_dataset_id):
        raise api.HTTPException(status_code=404, detail="sam3_dataset_not_found")

    monkeypatch.setattr(api, "_resolve_sam3_or_qwen_dataset", missing_dataset)
    monkeypatch.setattr(api.threading, "Thread", _NoThread)

    requests = [
        lambda: api.start_prompt_helper_job(api.PromptHelperRequest(dataset_id="missing_dataset")),
        lambda: api.start_prompt_helper_search(
            api.PromptHelperSearchRequest(
                dataset_id="missing_dataset",
                prompts_by_class={0: ["object"]},
            )
        ),
        lambda: api.start_prompt_helper_recipe(
            api.PromptRecipeRequest(
                dataset_id="missing_dataset",
                class_id=0,
                prompts=[api.PromptRecipePrompt(prompt="object")],
            )
        ),
    ]
    for request_fn in requests:
        with pytest.raises(api.HTTPException) as exc_info:
            request_fn()
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "sam3_dataset_not_found"

    assert api.PROMPT_HELPER_JOBS == {}


def test_prompt_helper_preset_missing_dataset_rejects_before_save(monkeypatch):
    def missing_dataset(_dataset_id):
        raise api.HTTPException(status_code=404, detail="sam3_dataset_not_found")

    def should_not_save(*args, **kwargs):
        raise AssertionError("prompt helper preset should not be saved for a missing dataset")

    monkeypatch.setattr(api, "_resolve_sam3_or_qwen_dataset", missing_dataset)
    monkeypatch.setattr(api, "_save_prompt_helper_preset_impl", should_not_save)

    with pytest.raises(api.HTTPException) as exc_info:
        api.create_prompt_helper_preset(
            dataset_id="missing_dataset",
            label="bad preset",
            prompts_json='{"0":["object"]}',
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "sam3_dataset_not_found"


def test_calibration_missing_dataset_rejects_before_queue(monkeypatch):
    with api.CALIBRATION_JOBS_LOCK:
        api.CALIBRATION_JOBS.clear()

    def missing_labelmap(_dataset_id):
        raise api.HTTPException(status_code=404, detail="sam3_dataset_not_found")

    def should_not_queue(*args, **kwargs):
        raise AssertionError("calibration job should not be queued")

    monkeypatch.setattr(api, "_agent_load_labelmap_meta", missing_labelmap)
    monkeypatch.setattr(api, "_start_calibration_job_impl", should_not_queue)

    with pytest.raises(api.HTTPException) as exc_info:
        api.start_calibration_job(api.CalibrationRequest(dataset_id="missing_dataset"))

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "sam3_dataset_not_found"
    assert api.CALIBRATION_JOBS == {}


def test_calibration_empty_dataset_rejects_before_queue(monkeypatch):
    with api.CALIBRATION_JOBS_LOCK:
        api.CALIBRATION_JOBS.clear()

    def should_not_queue(*args, **kwargs):
        raise AssertionError("calibration job should not be queued")

    monkeypatch.setattr(api, "_agent_load_labelmap_meta", lambda _dataset_id: (["object"], ""))
    monkeypatch.setattr(api, "_calibration_list_images", lambda _dataset_id: [])
    monkeypatch.setattr(api, "_start_calibration_job_impl", should_not_queue)

    with pytest.raises(api.HTTPException) as exc_info:
        api.start_calibration_job(api.CalibrationRequest(dataset_id="empty_dataset"))

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "calibration_images_missing"
    assert api.CALIBRATION_JOBS == {}


def test_agent_mining_missing_dataset_rejects_before_queue(monkeypatch, tmp_path):
    with api.AGENT_MINING_JOBS_LOCK:
        api.AGENT_MINING_JOBS.clear()

    classifier_path = tmp_path / "classifier.pkl"
    classifier_path.write_text("not used", encoding="utf-8")

    def missing_dataset(_dataset_id):
        raise api.HTTPException(status_code=404, detail="sam3_dataset_not_found")

    monkeypatch.setattr(api, "_resolve_agent_clip_classifier_path_impl", lambda *args, **kwargs: classifier_path)
    monkeypatch.setattr(api, "_load_clip_head_from_classifier_impl", lambda *args, **kwargs: ({}, {}))
    monkeypatch.setattr(api, "_resolve_sam3_or_qwen_dataset", missing_dataset)
    monkeypatch.setattr(api.threading, "Thread", _NoThread)

    with pytest.raises(api.HTTPException) as exc_info:
        api.start_agent_mining_job(
            api.AgentMiningRequest(
                dataset_id="missing_dataset",
                clip_head_classifier_path=str(classifier_path),
            )
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "sam3_dataset_not_found"
    assert api.AGENT_MINING_JOBS == {}


def test_yolo_train_missing_dataset_rejects_before_run_dir(monkeypatch, tmp_path):
    with api.YOLO_TRAINING_JOBS_LOCK:
        api.YOLO_TRAINING_JOBS.clear()

    def missing_dataset(_payload):
        raise api.HTTPException(status_code=404, detail="sam3_dataset_not_found")

    def should_not_create_run_dir(*args, **kwargs):
        if kwargs.get("create"):
            raise AssertionError("YOLO run dir should not be created")
        return tmp_path / "unused"

    monkeypatch.setattr(api, "_resolve_yolo_training_dataset", missing_dataset)
    monkeypatch.setattr(api, "_yolo_run_dir_impl", should_not_create_run_dir)
    monkeypatch.setattr(api.threading, "Thread", _NoThread)

    with pytest.raises(api.HTTPException) as exc_info:
        api.create_yolo_training_job(
            api.YoloTrainRequest(dataset_id="missing_dataset", accept_tos=True)
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "sam3_dataset_not_found"
    assert api.YOLO_TRAINING_JOBS == {}


def test_yolo_train_records_mps_device_resolution(monkeypatch, tmp_path):
    with api.YOLO_TRAINING_JOBS_LOCK:
        api.YOLO_TRAINING_JOBS.clear()

    run_dir = tmp_path / "yolo_run"

    def run_dir_impl(*_args, **kwargs):
        if kwargs.get("create"):
            run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    monkeypatch.setattr(api, "_resolve_yolo_training_dataset", lambda _payload: {"yolo_ready": True, "task": "detect"})
    monkeypatch.setattr(api, "_yolo_run_dir_impl", run_dir_impl)
    monkeypatch.setattr(api, "torch", _FakeTorch(cuda=False, mps=True))
    monkeypatch.setattr(api.threading, "Thread", _NoopThread)

    out = api.create_yolo_training_job(
        api.YoloTrainRequest(dataset_id="dataset_1", accelerator="mps", accept_tos=True)
    )

    job = api.YOLO_TRAINING_JOBS[out["job_id"]]
    assert job.config["accelerator"] == "mps"
    assert job.config["device_resolution"]["resolved_accelerator"] == "mps"
    assert job.config["device_resolution"]["device_arg"] == "mps"


def test_yolo_train_rejects_unavailable_mps_before_queue(monkeypatch, tmp_path):
    with api.YOLO_TRAINING_JOBS_LOCK:
        api.YOLO_TRAINING_JOBS.clear()

    monkeypatch.setattr(api, "_resolve_yolo_training_dataset", lambda _payload: {"yolo_ready": True, "task": "detect"})
    monkeypatch.setattr(api, "_yolo_run_dir_impl", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("run dir should not be created")))
    monkeypatch.setattr(api, "torch", _FakeTorch(cuda=False, mps=False))
    monkeypatch.setattr(api.threading, "Thread", _NoThread)

    with pytest.raises(api.HTTPException) as exc_info:
        api.create_yolo_training_job(
            api.YoloTrainRequest(dataset_id="dataset_1", accelerator="mps", accept_tos=True)
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "yolo_mps_unavailable"
    assert api.YOLO_TRAINING_JOBS == {}


def test_rfdetr_train_missing_dataset_rejects_before_run_dir(monkeypatch, tmp_path):
    with api.RFDETR_TRAINING_JOBS_LOCK:
        api.RFDETR_TRAINING_JOBS.clear()

    def missing_dataset(_payload):
        raise api.HTTPException(status_code=404, detail="sam3_dataset_not_found")

    def should_not_create_run_dir(*args, **kwargs):
        if kwargs.get("create"):
            raise AssertionError("RF-DETR run dir should not be created")
        return tmp_path / "unused"

    monkeypatch.setattr(api, "_resolve_rfdetr_training_dataset", missing_dataset)
    monkeypatch.setattr(api, "_rfdetr_run_dir_impl", should_not_create_run_dir)
    monkeypatch.setattr(api.threading, "Thread", _NoThread)

    with pytest.raises(api.HTTPException) as exc_info:
        api.create_rfdetr_training_job(
            api.RfDetrTrainRequest(dataset_id="missing_dataset", accept_tos=True)
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "sam3_dataset_not_found"
    assert api.RFDETR_TRAINING_JOBS == {}


def test_sam3_train_missing_dataset_rejects_before_config(monkeypatch):
    with api.SAM3_TRAINING_JOBS_LOCK:
        api.SAM3_TRAINING_JOBS.clear()

    def missing_dataset(_dataset_id):
        raise api.HTTPException(status_code=404, detail="sam3_dataset_not_found")

    def should_not_build_config(*args, **kwargs):
        raise AssertionError("SAM3 config should not be built for a missing dataset")

    monkeypatch.setattr(api, "_resolve_sam3_dataset_meta", missing_dataset)
    monkeypatch.setattr(api, "_build_sam3_config", should_not_build_config)
    monkeypatch.setattr(api.threading, "Thread", _NoThread)

    with pytest.raises(api.HTTPException) as exc_info:
        api.create_sam3_training_job(api.Sam3TrainRequest(dataset_id="missing_dataset"))

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "sam3_dataset_not_found"
    assert api.SAM3_TRAINING_JOBS == {}


def test_yolo_head_graft_preflight_rejects_before_queue(monkeypatch):
    with api.YOLO_HEAD_GRAFT_JOBS_LOCK:
        api.YOLO_HEAD_GRAFT_JOBS.clear()

    monkeypatch.setattr(
        api,
        "yolo_head_graft_dry_run",
        lambda _payload: {"ok": False, "error": "yolo_base_missing_best"},
    )
    monkeypatch.setattr(api.threading, "Thread", _NoThread)

    with pytest.raises(api.HTTPException) as exc_info:
        api.create_yolo_head_graft_job(
            api.YoloHeadGraftRequest(
                base_run_id="missing_run",
                dataset_id="missing_dataset",
                accept_tos=True,
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "yolo_base_missing_best"
    assert api.YOLO_HEAD_GRAFT_JOBS == {}
