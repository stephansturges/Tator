from types import SimpleNamespace

from PIL import Image

from services.prepass_grid import _agent_grid_cell_xyxy
from services.prepass_windows import _agent_sam3_text_windows, _agent_similarity_windows


def _grid_payload(**overrides):
    base = {
        "grid_cols": 2,
        "grid_rows": 2,
        "grid_overlap_ratio": None,
        "similarity_window_mode": "grid",
        "sam3_text_window_mode": "grid",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_grid_cell_parser_supports_standard_labels():
    grid = {
        "cols": 2,
        "rows": 2,
        "cell_w": 50.0,
        "cell_h": 50.0,
        "col_labels": ["A", "B"],
        "img_w": 100,
        "img_h": 100,
    }
    assert _agent_grid_cell_xyxy(grid, "A1", overlap_ratio=0.0) == (0.0, 0.0, 50.0, 50.0)
    assert _agent_grid_cell_xyxy(grid, "1B", overlap_ratio=0.0) == (50.0, 0.0, 100, 50.0)


def test_grid_windows_exist_and_respect_explicit_zero_overlap():
    img = Image.new("RGB", (100, 100), "white")
    payload_zero = _grid_payload(grid_overlap_ratio=0.0)
    payload_default = _grid_payload(grid_overlap_ratio=None)

    zero_windows = _agent_similarity_windows(payload_zero, pil_img=img, grid_overlap_ratio_default=0.2)
    default_windows = _agent_similarity_windows(
        payload_default,
        pil_img=img,
        grid_overlap_ratio_default=0.2,
    )
    assert len(zero_windows) == 4
    assert len(default_windows) == 4
    assert zero_windows[0]["bbox_xyxy_px"] == [0.0, 0.0, 50.0, 50.0]
    # Default overlap expands the first cell beyond exact half-width.
    assert default_windows[0]["bbox_xyxy_px"][2] > 50.0


def test_sam3_text_grid_windows_exist():
    img = Image.new("RGB", (100, 100), "white")
    payload = _grid_payload(grid_overlap_ratio=0.0)
    windows = _agent_sam3_text_windows(payload, pil_img=img, grid_overlap_ratio_default=0.2)
    assert len(windows) == 4
    assert windows[0]["grid_cell"] == "A1"


def test_similarity_sahi_windows_sanitize_invalid_slice_and_overlap(monkeypatch):
    captured = {}

    def _fake_slice(_img, slice_size, overlap):
        captured["slice_size"] = slice_size
        captured["overlap"] = overlap
        return [], [(0, 0)]

    monkeypatch.setattr("services.prepass_windows._slice_image_sahi", _fake_slice)
    img = Image.new("RGB", (100, 100), "white")
    payload = _grid_payload(
        similarity_window_mode="sahi",
        similarity_window_size=-1,
        similarity_window_overlap=1.7,
        sahi_window_size=640,
        sahi_overlap_ratio=0.2,
    )
    windows = _agent_similarity_windows(payload, pil_img=img, grid_overlap_ratio_default=0.2)
    assert captured == {"slice_size": 640, "overlap": 0.2}
    assert windows


def test_sam3_text_sahi_windows_sanitize_invalid_slice_and_overlap(monkeypatch):
    captured = {}

    def _fake_slice(_img, slice_size, overlap):
        captured["slice_size"] = slice_size
        captured["overlap"] = overlap
        return [], [(0, 0)]

    monkeypatch.setattr("services.prepass_windows._slice_image_sahi", _fake_slice)
    img = Image.new("RGB", (100, 100), "white")
    payload = _grid_payload(
        sam3_text_window_mode="sahi",
        sam3_text_window_size=0,
        sam3_text_window_overlap=float("nan"),
        sahi_window_size=640,
        sahi_overlap_ratio=0.2,
    )
    windows = _agent_sam3_text_windows(payload, pil_img=img, grid_overlap_ratio_default=0.2)
    assert captured == {"slice_size": 640, "overlap": 0.2}
    assert windows
