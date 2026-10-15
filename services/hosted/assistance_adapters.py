"""User-private trained classifier inference without application globals.

Recipe loading/crops reuse the existing classifier services. Probability and
feature composition preserve the first-pane trained-head inference contract.
Only the supervised worker instantiates a backbone; importing this module does
not load Torch, transformers, model weights, app, or localinferenceapi.
"""
from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
import tempfile

import numpy as np

from services.hosted.artifacts import open_regular, validate_digest
from utils.classifier_utils import _clip_head_background_indices, _clip_head_classes, _is_background_class_name
from utils.embedding_recipe import (
    apply_size_bias_residualizer, covariates_from_records, make_embedding_crop_views,
    normalize_background_mode, normalize_dinov3_pooling, normalize_embedding_aggregation,
    normalize_embedding_view_mode, normalize_preprocess_mode, normalize_rows,
)


def _verified_copy(source, target, digest, *, limit):
    expected = validate_digest(digest)
    actual, count = hashlib.sha256(), 0
    with open_regular(Path(source)) as reader, target.open("xb") as writer:
        while block := reader.read(1024 * 1024):
            count += len(block)
            if count > limit:
                raise ValueError("hosted_classifier_artifact_too_large")
            actual.update(block)
            writer.write(block)
    if actual.hexdigest() != expected:
        raise ValueError("hosted_classifier_artifact_changed")
    target.chmod(0o400)


def load_trained_head(config):
    """Load an admin-registered head and its mandatory, exact saved metadata.

    A private copy gives the existing loader its original adjacent-file naming
    convention even when the durable files are stored by content digest.
    Pickle/joblib is executable input and is never accepted from tool payloads.
    """
    import joblib
    from fastapi import HTTPException
    from services.classifier import (
        _infer_clip_model_from_embedding_dim_impl, _load_clip_head_from_classifier_impl,
        _resolve_head_normalize_embeddings_impl,
    )
    with tempfile.TemporaryDirectory(prefix="hosted-classifier-") as directory:
        checkpoint, metadata = Path(directory) / "head.pkl", Path(directory) / "head.meta.pkl"
        _verified_copy(config["checkpoint"], checkpoint, config["checkpoint_sha256"], limit=256 * 1024 * 1024)
        _verified_copy(config["metadata_path"], metadata, config["metadata_sha256"], limit=64 * 1024 * 1024)
        meta = joblib.load(metadata)
        if not isinstance(meta, dict) or not (meta.get("encoder_model") or meta.get("clip_model")):
            raise ValueError("hosted_classifier_recipe_required")
        # Preload metadata once: malformed metadata must never become the
        # legacy loader's optional-metadata fallback.
        head = _load_clip_head_from_classifier_impl(
            checkpoint, joblib_load_fn=lambda path: meta if Path(path) == metadata else joblib.load(path),
            http_exception_cls=HTTPException, clip_head_background_indices_fn=_clip_head_background_indices,
            resolve_head_normalize_embeddings_fn=lambda value, default: _resolve_head_normalize_embeddings_impl(value, default=default),
            infer_clip_model_fn=_infer_clip_model_from_embedding_dim_impl,
            active_clip_model_name=None, default_clip_model="ViT-B/32", logger=logging.getLogger(__name__),
        )
    if not isinstance(head, dict):
        raise ValueError("hosted_classifier_head_invalid")
    if normalize_embedding_aggregation(head.get("embedding_aggregation")) != "pooled":
        raise ValueError("hosted_classifier_aggregation_requires_registered_runtime_head")
    if str(head.get("encoder_type")).lower() not in {"clip", "dinov3", "cradio"}:
        raise ValueError("hosted_classifier_encoder_unsupported")
    labels = config.get("labels")
    if (not isinstance(labels, list) or not labels or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("hosted_classifier_registered_labels_required")
    classes = _clip_head_classes(head)
    if not classes or any(label not in labels and not _is_background_class_name(label) for label in classes):
        raise ValueError("hosted_classifier_registered_labels_changed")
    return head


def _build_encoder(head, device):
    """Instantiate a separate native backbone for this user worker."""
    kind = str(head["encoder_type"]).lower()
    name = head.get("encoder_model") or head.get("clip_model")
    if kind == "clip":
        import clip
        model, processor = clip.load(name, device=device)
        model.eval()
        return model, processor
    if kind == "dinov3":
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
        model.eval().to(device)
        return model, processor
    from utils.cradio_embedding import load_cradio_backbone
    model, processor, _, loaded_device = load_cradio_backbone(name, device, backend="torch")
    if str(loaded_device) != str(device):
        raise ValueError("hosted_classifier_device_changed")
    return model, processor


def _encode_views(model, processor, views, head, device):
    import torch
    kind = str(head["encoder_type"]).lower()
    if kind == "cradio":
        from utils.cradio_embedding import encode_cradio_images, normalize_cradio_pooling
        return np.asarray(encode_cradio_images(model, processor, device, views,
            pooling=normalize_cradio_pooling(head.get("cradio_pooling")), normalize=False), dtype=np.float32)
    with torch.no_grad():
        if kind == "clip":
            inputs = torch.stack([processor(view) for view in views]).to(device)
            features = model.encode_image(inputs).float()
        else:
            inputs = {key: value.to(device) for key, value in processor(images=views, return_tensors="pt").items()}
            output = model(**inputs)
            hidden = getattr(output, "last_hidden_state", None)
            start = 1 + max(0, int(getattr(getattr(model, "config", None), "num_register_tokens", 0) or 0))
            spatial = hidden[:, start:, :] if hidden is not None and hidden.ndim == 3 and hidden.shape[1] > start else None
            pooling = normalize_dinov3_pooling(head.get("dinov3_pooling"))
            if pooling == "pooler" and getattr(output, "pooler_output", None) is not None:
                features = output.pooler_output
            elif pooling == "patch_mean" and spatial is not None:
                features = spatial.mean(dim=1)
            elif pooling == "cls_patch_concat" and spatial is not None:
                features = torch.cat([torch.nn.functional.normalize(hidden[:, 0, :], dim=-1),
                                      torch.nn.functional.normalize(spatial.mean(dim=1), dim=-1)], dim=-1)
            elif hidden is not None:
                features = hidden[:, 0, :]
            else:
                features = output[0]
    return features.detach().float().cpu().numpy()


def _postprocess(features, head, records=None):
    features = np.asarray(features, dtype=np.float32)
    normalize = bool(head.get("normalize_embeddings"))
    if normalize:
        features = normalize_rows(features)
    transform = head.get("embedding_adjustment_transform")
    if transform and records is not None:
        covariates, _ = covariates_from_records(records)
        # The shared helper is permissive for historical analyses. An explicit
        # trained runtime must not silently discard a malformed fitted recipe.
        keep = np.asarray(transform.get("keep_mask"), dtype=bool)
        beta = np.asarray(transform.get("beta"), dtype=np.float32)
        if (keep.shape != (covariates.shape[1],) or beta.shape != (int(keep.sum()) + 1, features.shape[1])
                or np.asarray(transform.get("mean")).size != int(keep.sum())
                or np.asarray(transform.get("std")).size != int(keep.sum())):
            raise ValueError("hosted_classifier_adjustment_shape_invalid")
        features = apply_size_bias_residualizer(features, covariates, transform, normalize=True)
    if normalize and head.get("classifier_type") == "mlp":
        features = normalize_rows(features)
    for key, divide in (("embedding_center_values", False), ("embedding_std_values", True)):
        if head.get(key) is not None:
            values = np.asarray(head[key], dtype=np.float32).reshape(1, -1)
            if values.shape[1] != features.shape[1]:
                raise ValueError("hosted_classifier_standardization_shape_invalid")
            features = features / np.where(values == 0, 1.0, values) if divide else features - values
    if not np.isfinite(features).all():
        raise ValueError("hosted_classifier_features_invalid")
    return features.astype(np.float32, copy=False)


def predict_probabilities(features, head):
    """Existing linear/MLP, ArcFace, calibration and logit-adjustment math."""
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("hosted_classifier_features_invalid")
    if head.get("classifier_type") == "mlp":
        layers = head.get("layers") or []
        if not layers:
            raise ValueError("hosted_classifier_head_invalid")
        for index, layer in enumerate(layers):
            weight, bias = np.asarray(layer["weight"], dtype=np.float32), np.asarray(layer["bias"], dtype=np.float32)
            if head.get("arcface") and index == len(layers) - 1:
                scale = max(1e-6, float(head.get("arcface_scale") or 1.0))
                x_norm = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
                weight_norm = weight / np.maximum(np.linalg.norm(weight, axis=1, keepdims=True), 1e-12)
                x = x_norm @ weight_norm.T * scale
                continue
            x = x @ weight.T + bias
            if layer.get("layer_norm_weight") is not None:
                x = (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + float(layer.get("layer_norm_eps") or 1e-5))
                x = x * np.asarray(layer["layer_norm_weight"], dtype=np.float32)
                if layer.get("layer_norm_bias") is not None:
                    x = x + np.asarray(layer["layer_norm_bias"], dtype=np.float32)
            activation = str(layer.get("activation") or "").lower()
            if activation == "relu":
                x = np.maximum(0.0, x)
            elif activation == "gelu":
                x = 0.5 * x * (1.0 + np.vectorize(math.erf, otypes=[np.float32])(x / math.sqrt(2.0)))
        logits = x
    else:
        logits = x @ np.asarray(head["coef"], dtype=np.float32).T + np.asarray(head["intercept"], dtype=np.float32).reshape(1, -1)
    mode = str(head.get("proba_mode") or "softmax").lower()
    count = len(_clip_head_classes(head))
    if (mode == "binary" and (count != 2 or logits.shape[1] != 1)) or (mode != "binary" and logits.shape[1] != count):
        raise ValueError("hosted_classifier_output_shape_invalid")
    if head.get("logit_adjustment_inference") and head.get("logit_adjustment") is not None:
        adjustment = np.asarray(head["logit_adjustment"], dtype=np.float32).reshape(1, -1)
        if mode == "binary" and adjustment.shape[1] == 2:
            adjustment = adjustment[:, 1:2] - adjustment[:, 0:1]
        if adjustment.shape[1] != logits.shape[1]:
            raise ValueError("hosted_classifier_calibration_shape_invalid")
        logits = logits + adjustment
    if head.get("temperature") is not None:
        temperature = float(head["temperature"])
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("hosted_classifier_temperature_invalid")
        logits = logits / temperature
    if not np.isfinite(logits).all():
        raise ValueError("hosted_classifier_logits_invalid")
    if mode in {"binary", "ovr"}:
        with np.errstate(over="ignore"):
            probabilities = 1.0 / (1.0 + np.exp(-logits))
        probabilities = np.concatenate([1.0 - probabilities, probabilities], axis=1) if mode == "binary" else probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    else:
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32, copy=False)


class ClassifierRuntime:
    def __init__(self, config, device):
        self.head = load_trained_head(config)
        self.device = device
        self.model, self.processor = _build_encoder(self.head, device)

    def encode_crops(self, images, *, head=None, geometry_records=None):
        """Exact existing already-cropped feature seam used by EDR scoring."""
        if self.model is None:
            raise ValueError("hosted_classifier_closed")
        if not images:
            return None
        head = self.head if head is None else head
        raw = _encode_views(self.model, self.processor, images, head, self.device)
        return _postprocess(raw, head, geometry_records)

    def encode_items(self, items, *, head=None, geometry_records=None):
        """Compose saved crop views before the fitted head transform."""
        head = self.head if head is None else head
        rows = []
        for index, item in enumerate(items):
            views = list(item) if isinstance(item, (tuple, list)) else [item]
            raw_head = {key: value for key, value in head.items() if key not in {
                "embedding_adjustment_transform", "embedding_center_values", "embedding_std_values"}}
            if len(views) > 1:
                raw = self.encode_crops(views, head=raw_head)
                raw = normalize_rows(normalize_rows(raw).reshape(1, -1))
                records = [geometry_records[index]] if geometry_records is not None else None
                row = _postprocess(raw, head, records)
            else:
                records = [geometry_records[index]] if geometry_records is not None else None
                row = self.encode_crops(views, head=head, geometry_records=records)
            if row.shape != (1, int(head["embedding_dim"])):
                raise ValueError("hosted_classifier_embedding_shape_invalid")
            rows.append(row)
        return np.concatenate(rows, axis=0) if rows else None

    def predict(self, image, xyxy, background_guard=True):
        if self.model is None:
            raise ValueError("hosted_classifier_closed")
        bounds = np.asarray(xyxy, dtype=np.float64)
        if bounds.shape != (4,) or not np.isfinite(bounds).all():
            raise ValueError("hosted_classifier_bbox_invalid")
        x1, y1, x2, y2 = bounds.tolist()
        if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            raise ValueError("hosted_classifier_bbox_invalid")
        head = self.head
        padding = float(head.get("embedding_crop_padding_ratio") if head.get("embedding_crop_padding_ratio") is not None else 0.08)
        views, crop, metadata = make_embedding_crop_views(image, bounds,
            crop_mode=str(head.get("embedding_crop_mode") or "padded_square"),
            padding_ratio=max(0, min(1, padding)), preprocess_mode=normalize_preprocess_mode(head.get("preprocess_mode")),
            canonical_size=max(64, min(1024, int(head.get("canonical_size") or 336))),
            background_mode=normalize_background_mode(head.get("background_mode")),
            view_mode=normalize_embedding_view_mode(head.get("embedding_view_mode")))
        record = {"bbox_xyxy": bounds.tolist(), "crop_xyxy": list(crop), "width": int(round(max(1, x2-x1))),
                  "height": int(round(max(1, y2-y1))), "embedding_views": metadata}
        try:
            raw = np.asarray(_encode_views(self.model, self.processor, views, head, self.device), dtype=np.float32)
        finally:
            for view in views:
                view.close()
        if raw.ndim != 2 or raw.shape[0] != len(views):
            raise ValueError("hosted_classifier_embedding_shape_invalid")
        if len(views) > 1:
            raw_head = {key: value for key, value in head.items() if key not in {"embedding_adjustment_transform", "embedding_center_values", "embedding_std_values"}}
            raw = _postprocess(raw, raw_head)
            raw = normalize_rows(normalize_rows(raw).reshape(1, -1))
        features = _postprocess(raw, head, [record])
        if features.shape[1] != int(head["embedding_dim"]):
            raise ValueError("hosted_classifier_embedding_shape_invalid")
        row = predict_probabilities(features, head)[0]
        classes = _clip_head_classes(head)
        order = sorted(range(len(classes)), key=lambda index: float(row[index]), reverse=True)
        backgrounds = set(_clip_head_background_indices(classes))
        best = order[0]
        if background_guard and best in backgrounds:
            best = next((index for index in order if index not in backgrounds), best)
        # Preserve the existing first-pane second-label/margin selection rule.
        second = next((index for index in order[1:] if index != best), None)
        return {"label": classes[best], "proba": float(row[best]),
                "second_label": classes[second] if second is not None else None,
                "second_proba": float(row[second]) if second is not None else None,
                "margin": float(row[best] - row[second]) if second is not None else None, "error": None}

    def close(self):
        self.model = self.processor = None


def _sam3_processor(network, device):
    from sam3.model.sam3_image_processor import Sam3Processor
    from services.sam3_runtime import _sam3_clear_device_pinned_caches_impl
    _sam3_clear_device_pinned_caches_impl(network)
    return Sam3Processor(network, device=str(device))


def _finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("hosted_sam3_numeric_input_invalid")
    return value


def _numpy(value):
    return value.detach().float().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _sam3_masks(output, state, threshold, image):
    """Normalize SAM3 logits/probabilities to aligned image-size binary masks."""
    from PIL import Image
    raw = next((source[key] for source in (output, state) for key in ("masks_logits", "masks")
                if source.get(key) is not None), None)
    if raw is None:
        return None
    masks = _numpy(raw)
    if masks.size == 0:
        return []
    if masks.ndim == 2:
        masks = masks[None, ...]
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]
    if masks.ndim != 3 or not np.isfinite(masks).all():
        raise ValueError("hosted_sam3_masks_invalid")
    if masks.min() < 0 or masks.max() > 1:
        masks = 1 / (1 + np.exp(-np.clip(masks, -50, 50)))
    result = []
    for mask in masks:
        binary = (mask > threshold).astype(np.uint8)
        if binary.shape != (image.height, image.width):
            with Image.fromarray(binary) as source:
                with source.resize(image.size, Image.Resampling.NEAREST) as resized:
                    binary = np.array(resized, dtype=np.uint8)
        result.append(binary)
    return result


def _sam3_candidates(image, output, state, *, threshold, mask_threshold, min_size, limit):
    from utils.coords import _mask_to_bounding_box
    boxes = _numpy(output["boxes"]).reshape(-1, 4) if output.get("boxes") is not None else np.empty((0, 4))
    scores = _numpy(output["scores"]).reshape(-1) if output.get("scores") is not None else np.empty(0)
    masks = _sam3_masks(output, state, mask_threshold, image)
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("hosted_sam3_output_invalid")
    count = len(boxes) if len(boxes) else len(masks or [])
    order = sorted(range(count), key=lambda index: float(scores[index]) if index < len(scores) else 0, reverse=True)
    result = []
    for index in order:
        score = float(scores[index]) if index < len(scores) else None
        if score is not None and score < threshold:
            continue
        mask = masks[index] if masks is not None and index < len(masks) else None
        box = boxes[index].tolist() if len(boxes) else list(_mask_to_bounding_box(mask))
        box = [max(0, min(image.width, box[0])), max(0, min(image.height, box[1])),
               max(0, min(image.width, box[2])), max(0, min(image.height, box[3]))]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        area = float(np.count_nonzero(mask)) if mask is not None else (box[2]-box[0]) * (box[3]-box[1])
        if area < min_size:
            continue
        result.append({"xyxy": box, "score": score, "mask": mask})
        if limit and len(result) >= limit:
            break
    return result


def _iou_xyxy(first, second):
    from services.detector_merge import _iou_xywh
    return _iou_xywh([first[0], first[1], first[2]-first[0], first[3]-first[1]],
                     [second[0], second[1], second[2]-second[0], second[3]-second[1]])


class Sam3PromptRuntime:
    """Text/visual prompts on a user worker's already loaded SAM3 network.

    A fresh processor image state is used on every request/tile. Prompts never
    carry over between images, people, or text and visual requests. Window masks
    spill packed bits to private temporary files until score-ordered fusion, so
    dense windows do not retain one full-size mask per candidate in RAM.
    """
    def __init__(self, network, device):
        self.processor = _sam3_processor(network, device)

    def _infer(self, image, text, boxes, labels, *, threshold, mask_threshold, min_size, limit):
        import torch
        processor = self.processor
        with torch.inference_mode():
            processor.set_confidence_threshold(threshold)
            state = processor.set_image(image)
            output = state
            if text:
                try:
                    output = processor.set_text_prompt(state=state, prompt=text)
                    if boxes and isinstance(output, dict):
                        state = output
                except KeyError:
                    # Existing box-only SAM3 checkpoint path: no segmentation
                    # head is invented. Decode the model's actual grounding.
                    raw = processor.model.forward_grounding(backbone_out=state.get("backbone_out", {}),
                        find_input=processor.find_stage, find_target=None,
                        geometric_prompt=state.get("geometric_prompt", processor.model._get_dummy_prompt()))
                    corners = raw.get("pred_boxes_xyxy")
                    if corners is None:
                        center = _numpy(raw["pred_boxes"]).reshape(-1, 4)
                        corners = np.concatenate([center[:, :2] - center[:, 2:] / 2,
                                                  center[:, :2] + center[:, 2:] / 2], axis=1)
                    corners = _numpy(corners).reshape(-1, 4) * [image.width, image.height, image.width, image.height]
                    output = {"boxes": corners, "scores": torch.sigmoid(raw["pred_logits"]).reshape(-1)}
            for box, label in zip(boxes, labels):
                x, y, width, height = box
                output = processor.add_geometric_prompt(
                    [(x + width/2)/image.width, (y + height/2)/image.height, width/image.width, height/image.height],
                    label, state=state)
            if not isinstance(output, dict):
                raise ValueError("hosted_sam3_output_invalid")
            return _sam3_candidates(image, output, state, threshold=threshold, mask_threshold=mask_threshold,
                                    min_size=min_size, limit=limit)

    def predict(self, image, payload, *, visual=False, allow_partial_seeds=False):
        if self.processor is None:
            raise ValueError("hosted_sam3_closed")
        from PIL import Image
        from services.detector_params import _clamp_slice_params
        from utils.coco import _encode_binary_mask_impl
        from utils.coords import _xyxy_to_yolo_norm_list
        text = str(payload.get("text_prompt") or (payload.get("prompt") if visual else "") or "").strip()
        if len(text) > 8192 or (not visual and not text):
            raise ValueError("hosted_sam3_text_prompt_invalid")
        threshold = max(0, min(1, _finite(payload.get("threshold", .55 if visual else .5))))
        mask_threshold = max(0, min(1, _finite(payload.get("mask_threshold", .2 if visual else .5))))
        min_size = max(0, _finite(payload.get("min_size") or 0))
        simplify = payload.get("simplify_epsilon")
        if simplify is not None:
            simplify = max(0, _finite(simplify))
        limit = int(_finite(payload.get("max_results") or 0))
        limit = limit if limit > 0 else None
        if limit and limit > 10000:
            raise ValueError("hosted_sam3_result_limit_exceeded")
        boxes, labels = [], []
        if visual:
            raw = payload.get("bboxes") or [[payload["bbox_left"], payload["bbox_top"], payload["bbox_width"], payload["bbox_height"]]]
            if not isinstance(raw, list) or not raw or len(raw) > 1024:
                raise ValueError("hosted_sam3_visual_boxes_invalid")
            for item in raw:
                if not isinstance(item, (list, tuple)) or len(item) != 4:
                    raise ValueError("hosted_sam3_visual_boxes_invalid")
                x, y, width, height = [_finite(value) for value in item]
                valid = (0 <= x < image.width and 0 <= y < image.height and width > 0 and height > 0
                         and x+width <= image.width and y+height <= image.height)
                if allow_partial_seeds:
                    # Saved EDR windows deliberately retain the full exemplar
                    # geometry when its box intersects a crop. This is a
                    # server-only call option, never a browser payload field.
                    valid = width > 0 and height > 0 and x < image.width and y < image.height and x+width > 0 and y+height > 0
                if not valid:
                    raise ValueError("hosted_sam3_visual_boxes_invalid")
                boxes.append([x, y, width, height])
            labels = payload.get("bbox_labels")
            if labels is None:
                labels = [True] * len(boxes)
            if not isinstance(labels, list) or len(labels) != len(boxes) or any(type(label) is not bool for label in labels):
                raise ValueError("hosted_sam3_visual_labels_invalid")
        windowed = bool(payload.get("windowed"))
        if visual and windowed:
            raise ValueError("hosted_sam3_visual_windowing_unsupported")
        warnings, candidates, seeds = [], [], [[x, y, x+w, y+h] for x, y, w, h in boxes]
        tiles = [(image, (0, 0))]
        slices = None
        merge_iou = 1.0
        if windowed:
            from utils.image import _slice_image_sahi
            size, overlap, merge_iou = _clamp_slice_params(int(_finite(payload.get("window_size") or 672)),
                _finite(payload.get("window_overlap") if payload.get("window_overlap") is not None else .1),
                _finite(payload.get("merge_iou") if payload.get("merge_iou") is not None else .5),
                image.width, image.height, warnings)
            stride = max(1, size - int(overlap * size))
            if (math.ceil(max(0, image.width-size)/stride)+1) * (math.ceil(max(0, image.height-size)/stride)+1) > 256:
                raise ValueError("hosted_sam3_window_limit_exceeded")
            slices, starts = _slice_image_sahi(image, size, overlap)
            tiles = zip(slices, starts)
            warnings.extend([f"windowed_tiles:{len(slices)}", f"windowed_slice_size:{size}", f"windowed_overlap:{overlap:.2f}"])
        per_tile_limit = max(limit, 50) if limit and windowed else limit
        mask_bytes = 0
        with tempfile.TemporaryDirectory(prefix="hosted-sam3-prompts-") as directory:
            for tile, (offset_x, offset_y) in tiles:
                crop = Image.fromarray(tile).convert("RGB") if windowed else tile
                try:
                    results = self._infer(crop, text, boxes, labels, threshold=threshold, mask_threshold=mask_threshold,
                                          min_size=min_size, limit=per_tile_limit)
                    for result in results:
                        result["xyxy"] = [result["xyxy"][0]+offset_x, result["xyxy"][1]+offset_y,
                                          result["xyxy"][2]+offset_x, result["xyxy"][3]+offset_y]
                        if visual and any(_iou_xyxy(result["xyxy"], seed) > .9 for seed in seeds):
                            continue
                        mask = result.pop("mask")
                        result.update(mask_path=None, offset=(offset_x, offset_y), shape=(crop.height, crop.width))
                        if mask is not None:
                            packed = np.packbits(mask.reshape(-1)).tobytes()
                            mask_bytes += len(packed)
                            if mask_bytes > 512 * 1024 * 1024:
                                raise ValueError("hosted_sam3_scratch_limit_exceeded")
                            path = Path(directory) / str(len(candidates))
                            path.write_bytes(packed)
                            result["mask_path"] = path
                        candidates.append(result)
                        if len(candidates) > 10000:
                            raise ValueError("hosted_sam3_candidate_limit_exceeded")
                finally:
                    if windowed:
                        crop.close()
            candidates.sort(key=lambda result: result["score"] or 0, reverse=True)
            kept = []
            for result in candidates:
                if windowed and any(_iou_xyxy(result["xyxy"], other["xyxy"]) > merge_iou for other in kept):
                    continue
                kept.append(result)
                if limit and len(kept) >= limit:
                    break
            detections, encoded_masks, encoded_bytes = [], [], 0
            for result in kept:
                encoded = None
                if result["mask_path"] is not None:
                    height, width = result["shape"]
                    bits = np.frombuffer(result["mask_path"].read_bytes(), dtype=np.uint8)
                    mask = np.unpackbits(bits, count=height*width).reshape(height, width)
                    if windowed:
                        full = np.zeros((image.height, image.width), dtype=np.uint8)
                        x, y = result["offset"]
                        full[y:y+height, x:x+width] = mask
                        mask = full
                    encoded = _encode_binary_mask_impl(mask, max_bytes=16 * 1024 * 1024)
                    if encoded is None:
                        raise ValueError("hosted_sam3_mask_encoding_failed")
                    # Existing packed masks appear both in detections and the
                    # aligned masks list. Bound their complete wire response.
                    import json
                    encoded_bytes += len(json.dumps(encoded))
                    if encoded_bytes > 32 * 1024 * 1024:
                        raise ValueError("hosted_sam3_response_limit_exceeded")
                encoded_masks.append(encoded)
                detections.append({"bbox": _xyxy_to_yolo_norm_list(image.width, image.height, *result["xyxy"]),
                    "qwen_label": "visual" if visual else text, "source": "sam3_text", "score": result["score"],
                    "mask": encoded, "simplify_epsilon": simplify})
        if windowed:
            warnings.extend([f"windowed_per_window_limit:{per_tile_limit if per_tile_limit else 'unlimited'}",
                             f"windowed_final_limit:{limit if limit else 'unlimited'}",
                             f"windowed_raw_detections:{len(candidates)}", f"windowed_final_detections:{len(detections)}",
                             f"windowed_fused_detections:{len(candidates)-len(detections)}", f"windowed_merge_iou:{merge_iou:.2f}"])
        if not detections:
            warnings.append("no_results")
        return {"detections": detections, "warnings": warnings,
                "masks": encoded_masks if any(mask is not None for mask in encoded_masks) else None}

    def close(self):
        self.processor = None
