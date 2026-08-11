from types import SimpleNamespace

from services.calibration import (
    _canonical_sahi_settings,
    _first_float_with_default,
    _float_with_default,
    _canonical_sam3_text_window_settings,
    _canonical_cross_class_dedupe_settings,
    _canonical_similarity_settings,
    _canonical_similarity_window_settings,
)


def _payload(**kwargs):
    return SimpleNamespace(**kwargs)


def test_similarity_canonicalization_ignores_diverse_only_knobs_for_top_strategy():
    base = _payload(
        similarity_exemplar_strategy="top",
        similarity_exemplar_count=4,
        similarity_exemplar_seed=11,
        similarity_exemplar_fraction=0.15,
        similarity_exemplar_min=2,
        similarity_exemplar_max=9,
        similarity_exemplar_source_quota=1,
    )
    variant = _payload(
        similarity_exemplar_strategy="top",
        similarity_exemplar_count=4,
        similarity_exemplar_seed=999,
        similarity_exemplar_fraction=0.75,
        similarity_exemplar_min=8,
        similarity_exemplar_max=64,
        similarity_exemplar_source_quota=5,
    )
    assert _canonical_similarity_settings(base) == _canonical_similarity_settings(variant)


def test_similarity_canonicalization_ignores_top_count_for_diverse_strategy():
    base = _payload(
        similarity_exemplar_strategy="diverse",
        similarity_exemplar_count=3,
        similarity_exemplar_seed=42,
        similarity_exemplar_fraction=0.25,
        similarity_exemplar_min=4,
        similarity_exemplar_max=10,
        similarity_exemplar_source_quota=2,
    )
    variant = _payload(
        similarity_exemplar_strategy="diverse",
        similarity_exemplar_count=999,
        similarity_exemplar_seed=42,
        similarity_exemplar_fraction=0.25,
        similarity_exemplar_min=4,
        similarity_exemplar_max=10,
        similarity_exemplar_source_quota=2,
    )
    assert _canonical_similarity_settings(base) == _canonical_similarity_settings(variant)


def test_similarity_canonicalization_preserves_zero_source_quota_for_diverse_strategy():
    payload = _payload(
        similarity_exemplar_strategy="diverse",
        similarity_exemplar_count=3,
        similarity_exemplar_seed=42,
        similarity_exemplar_fraction=0.25,
        similarity_exemplar_min=4,
        similarity_exemplar_max=10,
        similarity_exemplar_source_quota=0,
    )
    assert _canonical_similarity_settings(payload)["source_quota"] == 0


def test_cross_class_dedupe_canonicalization_ignores_iou_when_disabled():
    first = _payload(cross_class_dedupe_enabled=False, cross_class_dedupe_iou=0.2)
    second = _payload(cross_class_dedupe_enabled=False, cross_class_dedupe_iou=0.95)
    assert _canonical_cross_class_dedupe_settings(first) == {
        "enabled": False,
        "iou": None,
    }
    assert _canonical_cross_class_dedupe_settings(first) == _canonical_cross_class_dedupe_settings(second)


def test_cross_class_dedupe_canonicalization_clamps_iou_when_enabled():
    first = _payload(cross_class_dedupe_enabled=True, cross_class_dedupe_iou=1.7)
    second = _payload(cross_class_dedupe_enabled=True, cross_class_dedupe_iou=1.0)
    assert _canonical_cross_class_dedupe_settings(first) == {"enabled": True, "iou": 1.0}
    assert _canonical_cross_class_dedupe_settings(first) == _canonical_cross_class_dedupe_settings(second)


def test_cross_class_dedupe_canonicalization_disables_when_iou_is_zero():
    payload = _payload(cross_class_dedupe_enabled=True, cross_class_dedupe_iou=0.0)
    assert _canonical_cross_class_dedupe_settings(payload) == {"enabled": False, "iou": None}
    disabled = _payload(cross_class_dedupe_enabled=False, cross_class_dedupe_iou=0.95)
    assert _canonical_cross_class_dedupe_settings(payload) == _canonical_cross_class_dedupe_settings(disabled)


def test_float_helpers_preserve_explicit_zero_values():
    assert _float_with_default(0.0, 0.5) == 0.0
    assert _float_with_default(None, 0.5) == 0.5
    assert _first_float_with_default(0.0, 0.7, default=0.2) == 0.0
    assert _first_float_with_default(None, 0.7, default=0.2) == 0.7


def test_window_canonicalization_ignores_window_params_when_windowing_disabled():
    sam3_first = _payload(
        sam3_text_window_extension=False,
        sam3_text_window_mode="sahi",
        sam3_text_window_size=896,
        sam3_text_window_overlap=0.45,
    )
    sam3_second = _payload(
        sam3_text_window_extension=False,
        sam3_text_window_mode="grid",
        sam3_text_window_size=320,
        sam3_text_window_overlap=0.05,
    )
    assert _canonical_sam3_text_window_settings(sam3_first) == {
        "enabled": False,
        "mode": None,
        "size": None,
        "overlap": None,
    }
    assert _canonical_sam3_text_window_settings(sam3_first) == _canonical_sam3_text_window_settings(
        sam3_second
    )

    sim_first = _payload(
        similarity_window_extension=False,
        similarity_window_mode="sahi",
        similarity_window_size=1024,
        similarity_window_overlap=0.4,
    )
    sim_second = _payload(
        similarity_window_extension=False,
        similarity_window_mode="grid",
        similarity_window_size=256,
        similarity_window_overlap=0.1,
    )
    assert _canonical_similarity_window_settings(sim_first) == {
        "enabled": False,
        "mode": None,
        "size": None,
        "overlap": None,
    }
    assert _canonical_similarity_window_settings(sim_first) == _canonical_similarity_window_settings(
        sim_second
    )


def test_window_canonicalization_ignores_sahi_size_overlap_in_grid_mode():
    sam3_first = _payload(
        sam3_text_window_extension=True,
        sam3_text_window_mode="grid",
        sam3_text_window_size=896,
        sam3_text_window_overlap=0.45,
    )
    sam3_second = _payload(
        sam3_text_window_extension=True,
        sam3_text_window_mode="grid",
        sam3_text_window_size=320,
        sam3_text_window_overlap=0.05,
    )
    assert _canonical_sam3_text_window_settings(sam3_first) == {
        "enabled": True,
        "mode": "grid",
        "size": None,
        "overlap": None,
    }
    assert _canonical_sam3_text_window_settings(sam3_first) == _canonical_sam3_text_window_settings(
        sam3_second
    )

    sim_first = _payload(
        similarity_window_extension=True,
        similarity_window_mode="grid",
        similarity_window_size=1024,
        similarity_window_overlap=0.4,
    )
    sim_second = _payload(
        similarity_window_extension=True,
        similarity_window_mode="grid",
        similarity_window_size=256,
        similarity_window_overlap=0.1,
    )
    assert _canonical_similarity_window_settings(sim_first) == {
        "enabled": True,
        "mode": "grid",
        "size": None,
        "overlap": None,
    }
    assert _canonical_similarity_window_settings(sim_first) == _canonical_similarity_window_settings(
        sim_second
    )


def test_window_canonicalization_sahi_sanitizes_size_and_overlap():
    sam3_payload = _payload(
        sam3_text_window_extension=True,
        sam3_text_window_mode="sahi",
        sam3_text_window_size=-512,
        sam3_text_window_overlap=1.7,
    )
    sim_payload = _payload(
        similarity_window_extension=True,
        similarity_window_mode="sahi",
        similarity_window_size="bad",
        similarity_window_overlap=float("nan"),
    )
    assert _canonical_sam3_text_window_settings(sam3_payload) == {
        "enabled": True,
        "mode": "sahi",
        "size": 640,
        "overlap": 0.2,
    }
    assert _canonical_similarity_window_settings(sim_payload) == {
        "enabled": True,
        "mode": "sahi",
        "size": 640,
        "overlap": 0.2,
    }


def test_sahi_canonicalization_treats_zero_values_as_default():
    zero_payload = _payload(sahi_window_size=0, sahi_overlap_ratio=0.0)
    negative_payload = _payload(sahi_window_size=-512, sahi_overlap_ratio=-0.1)
    large_overlap_payload = _payload(sahi_window_size=640, sahi_overlap_ratio=1.2)
    non_finite_payload = _payload(sahi_window_size="bad", sahi_overlap_ratio=float("nan"))
    default_payload = _payload(sahi_window_size=None, sahi_overlap_ratio=None)
    assert _canonical_sahi_settings(zero_payload) == {"size": 640, "overlap": 0.2}
    assert _canonical_sahi_settings(negative_payload) == {"size": 640, "overlap": 0.2}
    assert _canonical_sahi_settings(large_overlap_payload) == {"size": 640, "overlap": 0.2}
    assert _canonical_sahi_settings(non_finite_payload) == {"size": 640, "overlap": 0.2}
    assert _canonical_sahi_settings(zero_payload) == _canonical_sahi_settings(default_payload)
