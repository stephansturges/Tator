import numpy as np
import pytest

import localinferenceapi as api
from models.schemas import QwenDetection, Sam3TextPrompt
from utils.coords import _xyxy_to_yolo_norm_list


def test_sam3_window_mask_to_full_offsets_crop_mask():
    crop_mask = np.ones((2, 3), dtype=np.uint8)

    full_mask = api._sam3_window_mask_to_full(
        crop_mask,
        full_w=10,
        full_h=8,
        crop_w=3,
        crop_h=2,
        offset_x=4,
        offset_y=5,
    )

    assert full_mask is not None
    assert full_mask.shape == (8, 10)
    assert int(full_mask.sum()) == 6
    assert np.all(full_mask[5:7, 4:7] == 1)


def test_sam3_window_mask_to_full_clips_negative_offsets():
    crop_mask = np.ones((3, 4), dtype=np.uint8)

    full_mask = api._sam3_window_mask_to_full(
        crop_mask,
        full_w=6,
        full_h=5,
        crop_w=4,
        crop_h=3,
        offset_x=-2,
        offset_y=-1,
    )

    assert full_mask is not None
    assert full_mask.shape == (5, 6)
    assert int(full_mask.sum()) == 4
    assert np.all(full_mask[0:2, 0:2] == 1)


def test_sam3_windowed_fusion_keeps_best_overlapping_detection():
    high_score = QwenDetection(
        bbox=_xyxy_to_yolo_norm_list(100, 100, 10, 10, 30, 30),
        qwen_label="vehicle",
        source="sam3_text",
        score=0.9,
    )
    lower_score_overlap = QwenDetection(
        bbox=_xyxy_to_yolo_norm_list(100, 100, 12, 12, 32, 32),
        qwen_label="vehicle",
        source="sam3_text",
        score=0.7,
    )
    far_detection = QwenDetection(
        bbox=_xyxy_to_yolo_norm_list(100, 100, 70, 70, 90, 90),
        qwen_label="vehicle",
        source="sam3_text",
        score=0.6,
    )

    detections, masks = api._fuse_sam3_windowed_detections(
        [
            (lower_score_overlap, np.ones((100, 100), dtype=np.uint8)),
            (high_score, np.ones((100, 100), dtype=np.uint8)),
            (far_detection, np.ones((100, 100), dtype=np.uint8)),
        ],
        img_w=100,
        img_h=100,
        merge_iou=0.5,
        limit=10,
    )

    assert [det.score for det in detections] == [0.9, 0.6]
    assert masks is not None
    assert len(masks) == 2


def test_sam3_text_prompt_accepts_windowed_options():
    payload = Sam3TextPrompt(
        image_base64="abcd",
        text_prompt="red vehicle",
        windowed=True,
        window_size="640",
        window_overlap="0.2",
        merge_iou="0.55",
    )

    assert payload.windowed is True
    assert payload.window_size == 640
    assert payload.window_overlap == 0.2
    assert payload.merge_iou == 0.55


def test_labelmap_payload_rejects_multiline_class_names():
    with pytest.raises(api.HTTPException) as exc_info:
        api._normalize_labelmap_payload(["vehicle", "bad\nclass"])

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "labelmap_invalid_class"
