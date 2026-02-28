from utils.coords import _normalize_window_xyxy


def test_normalize_window_xyxy_supports_bbox_2d_dict():
    window = {"bbox_2d": [0, 0, 1000, 1000]}
    assert _normalize_window_xyxy(window, 100, 80) == (0.0, 0.0, 100.0, 80.0)


def test_normalize_window_xyxy_supports_xyxy_dict():
    window = {"x1": 10, "y1": 20, "x2": 90, "y2": 70}
    assert _normalize_window_xyxy(window, 100, 80) == (10.0, 20.0, 90.0, 70.0)

