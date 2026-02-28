from __future__ import annotations

from models.schemas import QwenDetection

import localinferenceapi


def test_dedupe_qwen_detections_iou_keeps_top_scoring_overlap() -> None:
    det_high = QwenDetection(
        bbox=[0.1, 0.1, 0.2, 0.2],
        source="bbox",
        score=0.9,
        class_name="obj",
    )
    det_low = QwenDetection(
        bbox=[0.1, 0.1, 0.2, 0.2],
        source="bbox",
        score=0.2,
        class_name="obj",
    )

    kept = localinferenceapi._dedupe_qwen_detections_iou(
        [det_low, det_high],
        img_w=100,
        img_h=100,
        iou_thresh=0.5,
    )

    assert kept == [det_high]


def test_dedupe_qwen_detections_iou_keeps_distinct_boxes() -> None:
    det_a = QwenDetection(
        bbox=[0.1, 0.1, 0.2, 0.2],
        source="bbox",
        score=0.4,
        class_name="obj",
    )
    det_b = QwenDetection(
        bbox=[0.7, 0.7, 0.2, 0.2],
        source="bbox",
        score=0.3,
        class_name="obj",
    )

    kept = localinferenceapi._dedupe_qwen_detections_iou(
        [det_a, det_b],
        img_w=100,
        img_h=100,
        iou_thresh=0.5,
    )

    assert kept == [det_a, det_b]
