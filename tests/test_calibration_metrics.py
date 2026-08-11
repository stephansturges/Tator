import pytest

from services.calibration_metrics import _evaluate_prompt_candidate_impl


def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def test_prompt_candidate_counts_predictions_once():
    def _run_sam3(*_args, **_kwargs):
        raise AssertionError("run_sam3_text_inference_fn should not be used with cached detections")

    metrics = _evaluate_prompt_candidate_impl(
        prompt="test",
        threshold=0.0,
        cat_id=0,
        image_ids=[1],
        gt_index={1: [("1:0", (0.0, 0.0, 10.0, 10.0))]},
        other_gt_index=None,
        images={1: {"path": "unused.jpg", "width": 100, "height": 100}},
        iou_threshold=0.5,
        max_dets=10,
        image_cache={},
        cached_detections={1: [(0.0, 0.0, 10.0, 10.0, 0.9), (20.0, 20.0, 30.0, 30.0, 0.8)]},
        run_sam3_text_inference_fn=_run_sam3,
        yolo_to_xyxy_fn=lambda _w, _h, bbox: bbox,
        iou_fn=_iou_xyxy,
    )

    assert metrics["preds"] == 2
    assert metrics["matches"] == 1
    assert metrics["fps"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(2.0 * 0.5 * 1.0 / (0.5 + 1.0 + 1e-8))
