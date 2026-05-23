import pytest

import localinferenceapi as api
from services import calibration as calibration_service


class _NoThread:
    def __init__(self, *args, **kwargs):
        raise AssertionError("worker thread should not be created for invalid job input")


class _NoopThread:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass


class _FailStartThread:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        raise RuntimeError("thread start failed")


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


def test_prompt_helper_jobs_roll_back_when_thread_start_fails(monkeypatch):
    with api.PROMPT_HELPER_JOBS_LOCK:
        api.PROMPT_HELPER_JOBS.clear()

    monkeypatch.setattr(api, "_validate_prompt_helper_dataset_id", lambda _dataset_id: None)
    monkeypatch.setattr(api.threading, "Thread", _FailStartThread)

    requests = [
        lambda: api.start_prompt_helper_job(api.PromptHelperRequest(dataset_id="dataset_1")),
        lambda: api.start_prompt_helper_search(
            api.PromptHelperSearchRequest(
                dataset_id="dataset_1",
                prompts_by_class={0: ["object"]},
            )
        ),
        lambda: api.start_prompt_helper_recipe(
            api.PromptRecipeRequest(
                dataset_id="dataset_1",
                class_id=0,
                prompts=[api.PromptRecipePrompt(prompt="object")],
            )
        ),
    ]
    for request_fn in requests:
        with pytest.raises(RuntimeError, match="thread start failed"):
            request_fn()
        assert api.PROMPT_HELPER_JOBS == {}


def test_class_analysis_rolls_back_when_thread_start_fails(monkeypatch):
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()

    monkeypatch.setattr(api.threading, "Thread", _FailStartThread)

    with pytest.raises(RuntimeError, match="thread start failed"):
        api._enqueue_class_analysis_job({"source_mode": "backend_dataset"})

    assert api.CLASS_ANALYSIS_JOBS == {}


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


def test_agent_mining_rolls_back_when_thread_start_fails(monkeypatch, tmp_path):
    with api.AGENT_MINING_JOBS_LOCK:
        api.AGENT_MINING_JOBS.clear()

    classifier_path = tmp_path / "classifier.pkl"
    classifier_path.write_text("not used", encoding="utf-8")
    monkeypatch.setattr(
        api, "_resolve_agent_clip_classifier_path_impl", lambda *args, **kwargs: classifier_path
    )
    monkeypatch.setattr(
        api, "_load_clip_head_from_classifier_impl", lambda *args, **kwargs: ({}, {})
    )
    monkeypatch.setattr(api, "_resolve_sam3_or_qwen_dataset", lambda _dataset_id: None)
    monkeypatch.setattr(api.threading, "Thread", _FailStartThread)

    with pytest.raises(RuntimeError, match="thread start failed"):
        api.start_agent_mining_job(
            api.AgentMiningRequest(
                dataset_id="dataset_1",
                clip_head_classifier_path=str(classifier_path),
            )
        )

    assert api.AGENT_MINING_JOBS == {}


def test_auto_label_rolls_back_when_thread_start_fails(monkeypatch):
    with api.AUTO_LABEL_JOBS_LOCK:
        api.AUTO_LABEL_JOBS.clear()

    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: {"id": "dataset_1"})
    monkeypatch.setattr(api.threading, "Thread", _FailStartThread)

    with pytest.raises(RuntimeError, match="thread start failed"):
        api.start_auto_label_job(
            api.AutoLabelRequest(
                dataset_id="dataset_1",
                enable_yolo=False,
                enable_rfdetr=False,
                enable_falcon=False,
                max_images=1,
            )
        )

    assert api.AUTO_LABEL_JOBS == {}


def test_segmentation_build_rolls_back_when_thread_start_fails(monkeypatch, tmp_path):
    with api.SEGMENTATION_BUILD_JOBS_LOCK:
        api.SEGMENTATION_BUILD_JOBS.clear()

    monkeypatch.setattr(
        api,
        "_plan_segmentation_build",
        lambda _request: (
            {"id": "seg_out"},
            {"dataset_root": str(tmp_path / "seg_out"), "log_dir": str(tmp_path / "logs")},
        ),
    )
    monkeypatch.setattr(api.threading, "Thread", _FailStartThread)

    with pytest.raises(RuntimeError, match="thread start failed"):
        api._start_segmentation_build_job(
            api.SegmentationBuildRequest(source_dataset_id="dataset_1")
        )

    assert api.SEGMENTATION_BUILD_JOBS == {}


def test_clip_training_worker_rolls_back_when_thread_start_fails(monkeypatch, tmp_path):
    with api.TRAINING_JOBS_LOCK:
        api.TRAINING_JOBS.clear()

    temp_dir = tmp_path / "clip_train_staging"
    temp_dir.mkdir()
    job = api.ClipTrainingJob(job_id="clip_fail", temp_dir=str(temp_dir))
    monkeypatch.setattr(api.threading, "Thread", _FailStartThread)

    with pytest.raises(RuntimeError, match="thread start failed"):
        api._start_training_worker(
            job,
            images_dir=str(tmp_path / "images"),
            labels_dir=str(tmp_path / "labels"),
            labelmap_path=None,
            clip_name=api.DEFAULT_CLIP_MODEL,
            encoder_type="clip",
            encoder_model=None,
            output_dir=str(tmp_path / "classifiers"),
            labelmap_dir=str(tmp_path / "labelmaps"),
            model_filename="model.pkl",
            labelmap_filename="labelmap.pkl",
            test_size=0.2,
            random_seed=42,
            batch_size=64,
            max_iter=1000,
            min_per_class=2,
            class_weight="none",
            effective_beta=0.9999,
            C=1.0,
            device_override=None,
            solver="saga",
            classifier_type="logreg",
            mlp_hidden_sizes="256",
            mlp_dropout=0.1,
            mlp_epochs=50,
            mlp_lr=1e-3,
            mlp_weight_decay=1e-4,
            mlp_label_smoothing=0.05,
            mlp_loss_type="ce",
            mlp_focal_gamma=2.0,
            mlp_focal_alpha=None,
            mlp_sampler="balanced",
            mlp_mixup_alpha=0.1,
            mlp_normalize_embeddings=True,
            mlp_patience=6,
            mlp_activation="relu",
            mlp_layer_norm=False,
            mlp_hard_mining_epochs=5,
            logit_adjustment_mode="none",
            logit_adjustment_inference=None,
            arcface_enabled=False,
            arcface_margin=0.2,
            arcface_scale=30.0,
            supcon_weight=0.0,
            supcon_temperature=0.07,
            supcon_projection_dim=128,
            supcon_projection_hidden=0,
            embedding_center=False,
            embedding_standardize=False,
            preprocess_mode="canonical",
            canonical_size=336,
            embedding_crop_mode="padded_square",
            embedding_crop_padding_ratio=0.08,
            background_mode="full_crop",
            embedding_view_mode="single",
            embedding_adjustment="remove_size_bias",
            dinov3_pooling="pooler",
            cradio_pooling="summary",
            embedding_aggregation="pooled",
            embedding_salad_head_id="",
            calibration_mode="none",
            calibration_max_iters=50,
            calibration_min_temp=0.5,
            calibration_max_temp=5.0,
            reuse_embeddings=False,
            hard_example_mining=False,
            hard_mining_misclassified_weight=3.0,
            hard_mining_low_conf_weight=2.0,
            hard_mining_low_conf_threshold=0.65,
            hard_mining_margin_threshold=0.15,
            convergence_tol=1e-4,
            bg_class_count=2,
            cancel_event=job.cancel_event,
        )

    assert api.TRAINING_JOBS == {}
    assert not temp_dir.exists()


def test_calibration_rolls_back_persisted_state_when_thread_start_fails(monkeypatch, tmp_path):
    jobs = {}
    jobs_lock = calibration_service.threading.Lock()
    monkeypatch.setattr(calibration_service.threading, "Thread", _FailStartThread)

    with pytest.raises(RuntimeError, match="thread start failed"):
        calibration_service._start_calibration_job(
            api.CalibrationRequest(dataset_id="dataset_1"),
            job_cls=calibration_service.CalibrationJob,
            jobs=jobs,
            jobs_lock=jobs_lock,
            run_job_fn=lambda *_args, **_kwargs: None,
            calibration_root=tmp_path,
        )

    assert jobs == {}
    assert not any(tmp_path.iterdir())


def test_calibration_start_rejects_symlinked_root_without_target_write_and_rollback(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    calibration_root = tmp_path / "calibration_jobs"
    try:
        calibration_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    jobs = {}
    jobs_lock = calibration_service.threading.Lock()

    with pytest.raises(ValueError, match="calibration_root_symlink"):
        calibration_service._start_calibration_job(
            api.CalibrationRequest(dataset_id="dataset_1"),
            job_cls=calibration_service.CalibrationJob,
            jobs=jobs,
            jobs_lock=jobs_lock,
            run_job_fn=lambda *_args, **_kwargs: None,
            calibration_root=calibration_root,
        )

    assert jobs == {}
    assert list(outside.iterdir()) == []


def test_calibration_report_bundle_rejects_symlinked_root_without_target_read(
    monkeypatch,
    tmp_path,
):
    outside = tmp_path / "outside"
    report_dir = outside / "cal_report"
    report_dir.mkdir(parents=True)
    (report_dir / "report_bundle.json").write_text('{"escaped": true}', encoding="utf-8")
    calibration_root = tmp_path / "calibration_jobs"
    try:
        calibration_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "CALIBRATION_ROOT", calibration_root)
    with api.CALIBRATION_JOBS_LOCK:
        api.CALIBRATION_JOBS.clear()

    with pytest.raises(api.HTTPException) as exc_info:
        api.get_calibration_report_bundle("cal_report")

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "calibration_job_not_found"


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


def test_yolo_train_cleans_run_dir_when_worker_start_fails(monkeypatch, tmp_path):
    yolo_root = tmp_path / "yolo_runs"
    monkeypatch.setattr(api, "YOLO_JOB_ROOT", yolo_root)
    monkeypatch.setattr(
        api,
        "_resolve_yolo_training_dataset",
        lambda _payload: {"yolo_ready": True, "task": "detect"},
    )
    monkeypatch.setattr(
        api,
        "_start_yolo_training_worker",
        lambda _job: (_ for _ in ()).throw(RuntimeError("thread start failed")),
    )
    with api.YOLO_TRAINING_JOBS_LOCK:
        api.YOLO_TRAINING_JOBS.clear()

    with pytest.raises(RuntimeError, match="thread start failed"):
        api.create_yolo_training_job(
            api.YoloTrainRequest(dataset_id="dataset_1", accept_tos=True)
        )

    assert api.YOLO_TRAINING_JOBS == {}
    assert not yolo_root.exists() or list(yolo_root.iterdir()) == []


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


def test_rfdetr_train_cleans_run_dir_when_worker_start_fails(monkeypatch, tmp_path):
    rfdetr_root = tmp_path / "rfdetr_runs"
    monkeypatch.setattr(api, "RFDETR_JOB_ROOT", rfdetr_root)
    monkeypatch.setattr(
        api,
        "_resolve_rfdetr_training_dataset",
        lambda _payload: {"dataset_root": str(tmp_path / "dataset"), "task": "detect"},
    )
    monkeypatch.setattr(
        api,
        "_start_rfdetr_training_worker",
        lambda _job: (_ for _ in ()).throw(RuntimeError("thread start failed")),
    )
    with api.RFDETR_TRAINING_JOBS_LOCK:
        api.RFDETR_TRAINING_JOBS.clear()

    with pytest.raises(RuntimeError, match="thread start failed"):
        api.create_rfdetr_training_job(
            api.RfDetrTrainRequest(dataset_id="dataset_1", accept_tos=True)
        )

    assert api.RFDETR_TRAINING_JOBS == {}
    assert not rfdetr_root.exists() or list(rfdetr_root.iterdir()) == []


def test_yolo_head_graft_cleans_run_dir_when_worker_start_fails(monkeypatch, tmp_path):
    yolo_root = tmp_path / "yolo_runs"
    monkeypatch.setattr(api, "YOLO_JOB_ROOT", yolo_root)
    monkeypatch.setattr(api, "_preflight_yolo_head_graft_create", lambda _payload: {"ok": True})
    monkeypatch.setattr(
        api,
        "_start_yolo_head_graft_worker",
        lambda _job: (_ for _ in ()).throw(RuntimeError("thread start failed")),
    )
    with api.YOLO_HEAD_GRAFT_JOBS_LOCK:
        api.YOLO_HEAD_GRAFT_JOBS.clear()

    with pytest.raises(RuntimeError, match="thread start failed"):
        api.create_yolo_head_graft_job(
            api.YoloHeadGraftRequest(
                base_run_id="base_run",
                dataset_id="dataset_1",
                accept_tos=True,
            )
        )

    assert api.YOLO_HEAD_GRAFT_JOBS == {}
    assert not yolo_root.exists() or list(yolo_root.iterdir()) == []


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
