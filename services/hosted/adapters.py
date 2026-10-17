"""Native hosted SAM and detector workers, independent of API globals.

Model imports are lazy and occur only inside the supervised child factory.
These adapters return proposals; they never write annotations or select models
from client-supplied paths.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import uuid

from services.hosted.config import RFDETR_CLASSES, verify_checkpoint


def _number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("hosted_numeric_input_invalid")
    return number


def _box(payload, width, height):
    left, top = _number(payload["bbox_left"]), _number(payload["bbox_top"])
    right = left + _number(payload["bbox_width"])
    bottom = top + _number(payload["bbox_height"])
    box = [max(0, min(width, left)), max(0, min(height, top)), max(0, min(width, right)), max(0, min(height, bottom))]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("hosted_bbox_invalid")
    return box


def _image(payload, config):
    from .media import MAX_MEDIA_BYTES, decode_encoded, validate_working_view
    from .errors import DomainError
    root = Path(config["media_root"]).resolve(strict=True)
    candidate = Path(payload["media_path"])
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("hosted_media_path_invalid")
    for item in (candidate, *candidate.parents):
        if item.resolve(strict=True) == root:
            break
        if item.is_symlink():
            raise ValueError("hosted_media_path_invalid")
    # Read one immutable byte snapshot: verification and decoding must describe
    # the same file, even if an external process replaces its path.
    with resolved.open("rb") as stream:
        raw = stream.read(MAX_MEDIA_BYTES + 1)
    if len(raw) > MAX_MEDIA_BYTES:
        raise ValueError("hosted_image_bytes_exceeded")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != payload.get("media_sha256"):
        raise ValueError("hosted_media_identity_changed")
    image, info = decode_encoded(raw)
    try:
        if payload.get('working_media'):
            validate_working_view(raw, payload['working_media'],
                source_sha256=payload.get('source_media_hash'), require_current_compatible=True)
        elif info.needs_working_view:
            raise DomainError('media_policy_upgrade_required', 409)
    except BaseException:
        image.close()
        raise
    return image, digest


def _build_sam(model, device):
    if model.get("backend") == "mlx":
        if model["kind"] != "sam3" or device != "mps":
            raise ValueError("hosted_sam3_mlx_requires_apple_silicon")
        from services.hosted.mlx_sam3 import build_predictor
        predictor = build_predictor(model)
        return predictor, predictor.model
    verify_checkpoint(model)
    import torch
    if model["kind"] == "sam1":
        from segment_anything import SamPredictor, sam_model_registry
        network = sam_model_registry[model["model_type"]](checkpoint=model["checkpoint"])
        network.to(device=device)
        network.eval()
        return SamPredictor(network), network
    from sam3.model_builder import build_sam3_image_model
    # Build on CPU, then explicitly move to the assigned device. Do not allocate
    # on CUDA:0 before relocating a user's stack to another GPU.
    network = build_sam3_image_model(device="cpu", checkpoint_path=model["checkpoint"],
        load_from_HF=False, enable_inst_interactivity=True,
        enable_segmentation=model.get("enable_segmentation", True), bpe_path=model["bpe_path"])
    network.to(device=torch.device(device))
    network.eval()
    predictor = getattr(network, "inst_interactive_predictor", None)
    if predictor is None or getattr(predictor, "model", None) is None:
        raise RuntimeError("sam3_interactive_predictor_missing")
    if getattr(predictor.model, "backbone", None) is None:
        predictor.model.backbone = network.backbone
    return predictor, network


def _stabilize_bbox_mask(mask, box):
    """Preserve the existing Magic Tweak local-component selection rule."""
    import cv2
    import numpy as np
    left, top, right, bottom = box
    height, width = mask.shape
    margin_x, margin_y = max(16, (right - left) * 0.5), max(16, (bottom - top) * 0.5)
    x1, y1 = max(0, math.floor(left - margin_x)), max(0, math.floor(top - margin_y))
    x2, y2 = min(width, math.ceil(right + margin_x)), min(height, math.ceil(bottom + margin_y))
    window = np.ascontiguousarray(mask[y1:y2, x1:x2])
    count, components, _, _ = cv2.connectedComponentsWithStats(window, connectivity=8)
    result = np.zeros_like(mask)
    if count <= 1:
        return result
    support = components[max(y1, math.floor(top))-y1:min(y2, math.ceil(bottom))-y1,
                         max(x1, math.floor(left))-x1:min(x2, math.ceil(right))-x1]
    counts = np.bincount(support.reshape(-1), minlength=count)
    counts[0] = 0
    selected = int(np.argmax(counts))
    if counts[selected] > 0:
        result[y1:y2, x1:x2] = (components == selected).astype(np.uint8)
    return result


class UserToolAdapter:
    def __init__(self, *, config, device):
        self.config, self.device = config, device
        self.capacity = config["capacity"]
        self.slots = OrderedDict()
        self.slots["current"] = {"backend": _build_sam(config["model"], device), "key": None}
        self.classifier = None
        self.sam3_prompts = None
        self.qwen = None
        self.sam3_chain = None
        if config.get("classifier"):
            from services.hosted.assistance_adapters import ClassifierRuntime
            self.classifier = ClassifierRuntime(config["classifier"], device)

    def _classify(self, image, bbox):
        if self.classifier is None:
            raise ValueError("hosted_classifier_not_configured")
        details = self.classifier.predict(image, bbox, background_guard=True)
        result = {key: details.get(key) for key in ("proba", "second_proba", "margin", "error")}
        bindings = self.config.get("classifier_bindings", {})
        label = details.get("label")
        if label not in bindings:
            if not details.get("error"):
                raise ValueError("hosted_classifier_label_unmapped")
            result.update(prediction="unknown", second_label=None)
        else:
            target = bindings[label]
            result.update(prediction=target["name"], class_id=str(target["position"]), stable_class_id=target["id"])
            second = bindings.get(details.get("second_label"))
            result["second_label"] = second["name"] if second else None
        return result

    def _slot(self, name):
        allowed = ("current", "next", "previous")[:self.capacity]
        if name not in allowed:
            raise ValueError("hosted_predictor_slot_disabled")
        if name not in self.slots:
            if self.config["model"].get("backend") == "mlx":
                # Reuse immutable weights within this user's worker; each slot
                # still owns its own image features and prompt state.
                from services.mlx_sam3 import MlxSam3PredictorAdapter
                current = self.slots["current"]["backend"][0]
                predictor = MlxSam3PredictorAdapter(current.model, current.processor,
                    model_path=current.model_path, runtime_notice=current.runtime_notice)
                backend = (predictor, current.model)
            else:
                backend = _build_sam(self.config["model"], self.device)
            self.slots[name] = {"backend": backend, "key": None}
        return self.slots[name]

    def execute(self, operation, payload, context):
        if context["actor_id"] != self.config["actor_id"]:
            raise ValueError("hosted_runtime_owner_mismatch")
        if operation == "sam_state":
            # Inspect only this user's actual cache; no image embedding is
            # produced or moved by a status request. Restrict to this dataset.
            entries = []
            for name in ("current", "next", "previous")[:self.capacity]:
                slot = self.slots.get(name)
                key = slot.get("key") if slot else None
                visible = key is not None and key[0] == context["dataset_id"]
                entries.append({"slot": name, "enabled": True, "loaded": slot is not None,
                    "variant": self.config["model"]["kind"], "image_id": key[1] if visible else None,
                    "token": hashlib.sha256(json.dumps(key).encode()).hexdigest() if visible else None})
            return {"slots": entries, "capacity": self.capacity}
        auto = operation in {"sam_point_auto", "sam_point_multi_auto", "sam_bbox_auto", "sam3_text_auto"}
        if auto:
            operation = operation.removesuffix("_auto")
        if operation not in {"sam_preload", "sam_point", "sam_point_multi", "sam_bbox", "classify_bbox", "sam3_text", "sam3_visual", "sam3_recipe_chain", "qwen_infer", "qwen_caption", "qwen_caption_preview", "qwen_caption_case"}:
            raise ValueError("hosted_tool_not_implemented")
        if (auto or operation == "classify_bbox") and self.classifier is None:
            raise ValueError("hosted_classifier_not_configured")
        import numpy as np
        from utils.coco import _encode_binary_mask_impl
        from utils.coords import _mask_to_bounding_box, _xyxy_to_yolo_norm_list
        image, digest = _image(payload, self.config)
        try:
            # Pixel features are reusable across saves, tab leases and logins.
            # Full immutable authority is checked by the supervisor on every
            # admission/dispatch/result; this actor-private cache grants none.
            key = (context["dataset_id"], context["image_id"], digest)
            if operation == "sam_preload":
                slot = self._slot(payload.get("slot", "current"))
            token = hashlib.sha256(json.dumps(key).encode()).hexdigest()
            if operation == "sam3_recipe_chain":
                if not self.config.get("sam3_recipes"):
                    raise ValueError("hosted_sam3_recipe_not_configured")
                if self.sam3_chain is None:
                    from services.hosted.sam3_chain import Sam3ChainRuntime
                    self.sam3_chain = Sam3ChainRuntime(self.config["sam3_recipes"], self.device)
                result = self.sam3_chain.execute(image, payload, context)
                result["image_token"] = token
                return result
            if operation in {"qwen_infer", "qwen_caption", "qwen_caption_preview", "qwen_caption_case"}:
                if not self.config.get("qwen"):
                    raise ValueError("hosted_qwen_not_configured")
                if self.qwen is None:
                    from services.hosted.qwen_adapters import QwenRuntime
                    def sam_predict(source, **options):
                        # Qwen refinement can use a view/crop. Its exact input
                        # differs from a navigation preload, so reset the cache
                        # identity instead of claiming it remains the full frame.
                        current = self._slot("current")
                        current["backend"][0].set_image(np.ascontiguousarray(source))
                        current["key"] = None
                        return current["backend"][0].predict(**options)
                    self.qwen = QwenRuntime(self.config["qwen"], self.config["qwen"]["device"], sam_predict=sam_predict)
                request_payload = dict(payload)
                if operation == "qwen_caption_case":
                    from services.hosted.caption_case import execute_caption_case
                    result = execute_caption_case(self.qwen, image, request_payload, context)
                    result["image_token"] = token
                    return result
                if operation in {"qwen_caption", "qwen_caption_preview"}:
                    request_payload.update(unload_others=False, force_unload=False)
                result = self.qwen.execute(operation, image, request_payload, context)
                result["image_token"] = token
                return result
            if operation in {"sam3_text", "sam3_visual"}:
                if self.config["model"]["kind"] != "sam3":
                    raise ValueError("hosted_sam3_required")
                if self.config["model"].get("backend") == "mlx":
                    raise ValueError("hosted_sam3_mlx_grounding_not_supported")
                if self.sam3_prompts is None:
                    from services.hosted.assistance_adapters import Sam3PromptRuntime
                    self.sam3_prompts = Sam3PromptRuntime(self._slot("current")["backend"][1], self.device)
                result = self.sam3_prompts.predict(image, payload, visual=operation == "sam3_visual")
                result["image_token"] = token
                if auto:
                    from utils.coco import _decode_binary_mask_impl
                    from utils.coords import _yolo_to_xyxy_int
                    for detection in result["detections"]:
                        box = _yolo_to_xyxy_int(image.width, image.height, detection["bbox"])
                        if detection.get("mask"):
                            box = _mask_to_bounding_box(_decode_binary_mask_impl(detection["mask"]))
                        detection.pop("qwen_label", None)
                        detection.update(uuid=uuid.uuid4().hex, image_token=token)
                        if box[2] <= box[0] or box[3] <= box[1]:
                            detection.update(prediction="unknown", error="empty_mask")
                        else:
                            detection.update(self._classify(image, box))
                return result
            if operation == "classify_bbox":
                bbox = _box(payload, image.width, image.height)
                return {**self._classify(image, bbox), "bbox": _xyxy_to_yolo_norm_list(image.width, image.height, *bbox),
                        "uuid": payload.get("uuid"), "image_token": token}
            if operation == "sam_preload":
                if slot["key"] != key:
                    slot["backend"][0].set_image(np.ascontiguousarray(image))
                    slot["key"] = key
                return {"status": "ready", "token": token, "image_token": token, "width": image.width, "height": image.height,
                        "backend": self.config["model"]["kind"], "device": self.device}
            options = {"multimask_output": False}
            box = None
            if operation == "sam_bbox":
                box = _box(payload, image.width, image.height)
                options["box"] = np.asarray(box, dtype=np.float32)
            else:
                positive = [[payload["point_x"], payload["point_y"]]] if operation == "sam_point" else payload.get("positive_points", [])
                negative = [] if operation == "sam_point" else payload.get("negative_points", [])
                if not positive or len(positive) + len(negative) > 1024:
                    raise ValueError("hosted_sam_points_invalid")
                points = np.asarray(positive + negative, dtype=np.float32)
                if points.shape != (len(positive) + len(negative), 2) or not np.isfinite(points).all():
                    raise ValueError("hosted_sam_points_invalid")
                if np.any(points < 0) or np.any(points[:, 0] >= image.width) or np.any(points[:, 1] >= image.height):
                    raise ValueError("hosted_sam_points_outside_image")
                options.update(point_coords=points, point_labels=np.asarray([1] * len(positive) + [0] * len(negative), dtype=np.int64))
            # Every request revalidates bytes; cache lookup never authorizes
            # access. Invalid prompts cannot displace an active cached image.
            slot = next((slot for slot in self.slots.values() if slot["key"] == key), self._slot("current"))
            predictor = slot["backend"][0]
            if slot["key"] != key:
                predictor.set_image(np.ascontiguousarray(image))
                slot["key"] = key
            masks, _, _ = predictor.predict(**options)
            mask = (np.asarray(masks[0]) > 0).astype(np.uint8)
            if mask.shape != (image.height, image.width):
                raise ValueError("hosted_sam_mask_shape_invalid")
            if box is not None and payload.get("local_refine"):
                mask = _stabilize_bbox_mask(mask, box)
                if not mask.any():
                    raise ValueError("sam_local_refine_missed_prompt")
            bbox = _mask_to_bounding_box(mask)
            result = {"class_id": "0", "bbox": _xyxy_to_yolo_norm_list(image.width, image.height, *bbox),
                    "uuid": payload.get("uuid"), "image_token": token,
                    "mask": _encode_binary_mask_impl(mask, max_bytes=16 * 1024 * 1024), "simplify_epsilon": None}
            if auto:
                result.pop("class_id")
                if not mask.any():
                    result.update(prediction="unknown", error="empty_mask")
                else:
                    result.update(self._classify(image, bbox))
            return result
        finally:
            image.close()

    def close(self):
        if self.sam3_chain is not None:
            self.sam3_chain.close()
            self.sam3_chain = None
        if self.qwen is not None:
            self.qwen.close()
            self.qwen = None
        if self.sam3_prompts is not None:
            self.sam3_prompts.close()
            self.sam3_prompts = None
        self.slots.clear()
        if self.classifier is not None:
            self.classifier.close()
            self.classifier = None


def _build_detector(definition, device):
    verify_checkpoint(definition)
    if definition["kind"] == "yolo":
        from ultralytics import YOLO
        model = YOLO(definition["checkpoint"])
        if getattr(model, "task", None) != "detect":
            raise ValueError("hosted_detector_requires_bbox_model")
        model.to(device)
        return model
    import rfdetr
    model_class = getattr(rfdetr, RFDETR_CLASSES[definition["variant"]])
    model = model_class(pretrain_weights=definition["checkpoint"], device=device)
    model.model.class_names = definition["labels"]
    return model


class DatasetDetectorAdapter:
    def __init__(self, *, config, device):
        self.config, self.device = config, device
        self.model = _build_detector(config["model"], device)

    def execute(self, operation, payload, context):
        if context["dataset_id"] != self.config["dataset_id"]:
            raise ValueError("hosted_runtime_owner_mismatch")
        if operation not in {"detect_full", "detect_region", "detect_windowed"}:
            raise ValueError("hosted_tool_not_implemented")
        from PIL import Image
        from services.detector_merge import _merge_detections_nms
        from services.detector_params import _clamp_conf_value, _clamp_iou_value, _clamp_max_det_value, _clamp_slice_params
        from services.detectors import _yolo_extract_detections_impl, _rfdetr_extract_detections_impl
        from utils.image import _slice_image_sahi
        image, _ = _image(payload, self.config)
        try:
            warnings = []
            conf = _clamp_conf_value(_number(payload.get("conf", 0.25)), warnings)
            iou = _clamp_iou_value(_number(payload.get("iou", 0.45)), warnings)
            max_det = _clamp_max_det_value(int(payload.get("max_det", 300)), warnings)
            merge_iou = 0
            if operation == "detect_region":
                left, top, right, bottom = _box(payload, image.width, image.height)
                region = (math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom))
                crops = [(image.crop(region), (region[0], region[1]))]
            elif operation == "detect_windowed":
                size, overlap, merge_iou = _clamp_slice_params(int(payload.get("slice_size", 640)),
                    _number(payload.get("overlap", 0.2)), _number(payload.get("merge_iou", 0.5)), image.width, image.height, warnings)
                stride = max(1, size - int(overlap * size))
                columns = max(1, math.ceil(max(0, image.width - size) / stride) + 1)
                rows = max(1, math.ceil(max(0, image.height - size) / stride) + 1)
                if columns * rows > 1024:
                    raise ValueError("hosted_detector_tile_budget_exceeded")
                tiles, starts = _slice_image_sahi(image, size, overlap)
                crops = [(Image.fromarray(tile), start) for tile, start in zip(tiles, starts, strict=True)]
            else:
                crops = [(image, (0, 0))]
            labels, raw = list(self.config["model"]["labels"]), []
            try:
                for index, (crop, (x, y)) in enumerate(crops):
                    if self.config["model"]["kind"] == "yolo":
                        result = self.model.predict(crop, conf=conf, iou=iou, max_det=max_det, verbose=False, device=self.device)
                        extracted = _yolo_extract_detections_impl(result, labels, x, y, image.width, image.height)
                    else:
                        result = self.model.predict(crop, threshold=conf)
                        class_ids = self.config["model"].get("class_ids")
                        if class_ids is not None and result is not None and getattr(result, "class_id", None) is not None:
                            import numpy as np
                            lookup = {native: index for index, native in enumerate(class_ids)}
                            if any(int(value) not in lookup for value in result.class_id):
                                raise ValueError("hosted_detector_label_unmapped")
                            result.class_id = np.asarray([lookup[int(value)] for value in result.class_id])
                        extracted, shifted = _rfdetr_extract_detections_impl(result, labels, x, y, image.width, image.height)
                        if shifted:
                            warnings.append("labelmap_shifted")
                    raw.extend(extracted)
                    # Preserve the native detector's full-frame fallback when
                    # windowed inference finds no objects in any tile.
                    if operation == "detect_windowed" and index == len(crops) - 1 and not raw and crop is not image:
                        crops.append((image, (0, 0)))
                        warnings.append("sahi_empty_full_frame_fallback")
            finally:
                for crop, _ in crops:
                    if crop is not image:
                        crop.close()
            bindings = self.config.get("class_bindings")
            if bindings is not None:
                for detection in raw:
                    source = detection.get("class_name")
                    if source not in bindings:
                        raise ValueError("hosted_detector_label_unmapped")
                    target = bindings[source]
                    detection.update(class_id=target["position"], stable_class_id=target["id"], class_name=target["name"])
                labels = list(self.config["dataset_labelmap"])
            detections = _merge_detections_nms(raw, merge_iou, max_det)
            return {"detections": detections, "labelmap": labels, "warnings": sorted(set(warnings)) or None}
        finally:
            image.close()

    def close(self):
        self.model = None


class DatasetPipelineAdapter:
    """One stable sequential owner for direct detection and saved EDR recipes."""
    def __init__(self, *, config, device):
        self.config, self.device = config, device
        self.detector = self.edr = None

    def execute(self, operation, payload, context):
        if context["dataset_id"] != self.config["dataset_id"]:
            raise ValueError("hosted_runtime_owner_mismatch")
        if operation == "edr_prepass":
            if "edr" not in self.config:
                raise ValueError("hosted_edr_not_configured")
            if self.edr is None:
                from services.hosted.edr_adapters import DatasetEdrAdapter
                self.edr = DatasetEdrAdapter(config=self.config["edr"], device=self.device)
            return self.edr.execute(operation, payload, context)
        if operation not in {"detect_full", "detect_region", "detect_windowed"} or "detector" not in self.config:
            raise ValueError("hosted_detector_not_configured")
        if self.detector is None:
            self.detector = DatasetDetectorAdapter(config=self.config["detector"], device=self.device)
        return self.detector.execute(operation, payload, context)

    def close(self):
        for adapter in (self.detector, self.edr):
            if adapter:
                adapter.close()
        self.detector = self.edr = None
