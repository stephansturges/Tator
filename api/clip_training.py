"""APIRouter for CLIP classifier training endpoints."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Type

from fastapi import APIRouter, File, Form, UploadFile


def build_clip_training_router(
    *,
    start_fn: Callable[..., Any],
    list_fn: Callable[[], Any],
    get_fn: Callable[[str], Any],
    cancel_fn: Callable[[str], Any],
) -> APIRouter:
    router = APIRouter()

    @router.post("/clip/train")
    async def start_clip_training(
        images: Optional[List[UploadFile]] = File(None),
        labels: Optional[List[UploadFile]] = File(None),
        labelmap: Optional[UploadFile] = File(None),
        clip_model_name: str = Form(""),
        encoder_type: str = Form("clip"),
        encoder_model: Optional[str] = Form(None),
        output_dir: str = Form("."),
        model_filename: str = Form("my_logreg_model.pkl"),
        labelmap_filename: str = Form("my_label_list.pkl"),
        test_size: float = Form(0.2),
        random_seed: int = Form(42),
        batch_size: int = Form(64),
        max_iter: int = Form(1000),
        min_per_class: int = Form(2),
        class_weight: str = Form("balanced"),
        effective_beta: float = Form(0.9999),
        C: float = Form(1.0),
        device_override: Optional[str] = Form(None),
        images_path_native: Optional[str] = Form(None),
        labels_path_native: Optional[str] = Form(None),
        labelmap_path_native: Optional[str] = Form(None),
        solver: str = Form("saga"),
        classifier_type: str = Form("logreg"),
        mlp_hidden_sizes: str = Form("256"),
        mlp_dropout: float = Form(0.1),
        mlp_epochs: int = Form(50),
        mlp_lr: float = Form(1e-3),
        mlp_weight_decay: float = Form(1e-4),
        mlp_label_smoothing: float = Form(0.05),
        mlp_loss_type: str = Form("ce"),
        mlp_focal_gamma: float = Form(2.0),
        mlp_focal_alpha: float = Form(-1.0),
        mlp_sampler: str = Form("balanced"),
        mlp_mixup_alpha: float = Form(0.1),
        mlp_normalize_embeddings: Optional[str] = Form("true"),
        mlp_patience: int = Form(6),
        mlp_activation: str = Form("relu"),
        mlp_layer_norm: Optional[str] = Form("false"),
        mlp_hard_mining_epochs: int = Form(5),
        logit_adjustment_mode: str = Form("none"),
        logit_adjustment_inference: Optional[str] = Form(None),
        arcface_enabled: Optional[str] = Form("false"),
        arcface_margin: float = Form(0.2),
        arcface_scale: float = Form(30.0),
        supcon_weight: float = Form(0.0),
        supcon_temperature: float = Form(0.07),
        supcon_projection_dim: int = Form(128),
        supcon_projection_hidden: int = Form(0),
        embedding_center: Optional[str] = Form("false"),
        embedding_standardize: Optional[str] = Form("false"),
        calibration_mode: str = Form("none"),
        calibration_max_iters: int = Form(50),
        calibration_min_temp: float = Form(0.5),
        calibration_max_temp: float = Form(5.0),
        reuse_embeddings: Optional[str] = Form(None),
        hard_example_mining: Optional[str] = Form(None),
        hard_mis_weight: float = Form(3.0),
        hard_low_conf_weight: float = Form(2.0),
        hard_low_conf_threshold: float = Form(0.65),
        hard_margin_threshold: float = Form(0.15),
        convergence_tol: float = Form(1e-4),
        bg_class_count: int = Form(2),
        staged_temp_dir: Optional[str] = Form(None),
    ):
        return await start_fn(
            images=images,
            labels=labels,
            labelmap=labelmap,
            clip_model_name=clip_model_name,
            encoder_type=encoder_type,
            encoder_model=encoder_model,
            output_dir=output_dir,
            model_filename=model_filename,
            labelmap_filename=labelmap_filename,
            test_size=test_size,
            random_seed=random_seed,
            batch_size=batch_size,
            max_iter=max_iter,
            min_per_class=min_per_class,
            class_weight=class_weight,
            effective_beta=effective_beta,
            C=C,
            device_override=device_override,
            images_path_native=images_path_native,
            labels_path_native=labels_path_native,
            labelmap_path_native=labelmap_path_native,
            solver=solver,
            classifier_type=classifier_type,
            mlp_hidden_sizes=mlp_hidden_sizes,
            mlp_dropout=mlp_dropout,
            mlp_epochs=mlp_epochs,
            mlp_lr=mlp_lr,
            mlp_weight_decay=mlp_weight_decay,
            mlp_label_smoothing=mlp_label_smoothing,
            mlp_loss_type=mlp_loss_type,
            mlp_focal_gamma=mlp_focal_gamma,
            mlp_focal_alpha=mlp_focal_alpha,
            mlp_sampler=mlp_sampler,
            mlp_mixup_alpha=mlp_mixup_alpha,
            mlp_normalize_embeddings=mlp_normalize_embeddings,
            mlp_patience=mlp_patience,
            mlp_activation=mlp_activation,
            mlp_layer_norm=mlp_layer_norm,
            mlp_hard_mining_epochs=mlp_hard_mining_epochs,
            logit_adjustment_mode=logit_adjustment_mode,
            logit_adjustment_inference=logit_adjustment_inference,
            arcface_enabled=arcface_enabled,
            arcface_margin=arcface_margin,
            arcface_scale=arcface_scale,
            supcon_weight=supcon_weight,
            supcon_temperature=supcon_temperature,
            supcon_projection_dim=supcon_projection_dim,
            supcon_projection_hidden=supcon_projection_hidden,
            embedding_center=embedding_center,
            embedding_standardize=embedding_standardize,
            calibration_mode=calibration_mode,
            calibration_max_iters=calibration_max_iters,
            calibration_min_temp=calibration_min_temp,
            calibration_max_temp=calibration_max_temp,
            reuse_embeddings=reuse_embeddings,
            hard_example_mining=hard_example_mining,
            hard_mis_weight=hard_mis_weight,
            hard_low_conf_weight=hard_low_conf_weight,
            hard_low_conf_threshold=hard_low_conf_threshold,
            hard_margin_threshold=hard_margin_threshold,
            convergence_tol=convergence_tol,
            bg_class_count=bg_class_count,
            staged_temp_dir=staged_temp_dir,
        )

    @router.get("/clip/train")
    def list_training_jobs():
        return list_fn()

    @router.get("/clip/train/{job_id}")
    def get_training_job(job_id: str):
        return get_fn(job_id)

    @router.post("/clip/train/{job_id}/cancel")
    def cancel_training_job(job_id: str):
        return cancel_fn(job_id)

    return router
