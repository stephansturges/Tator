"""Object-aware Incoming diversity: the scoring and selection rules."""
import numpy as np
import pytest

from services.data_ingestion import (
    nearest_distances,
    object_aware_ranking,
    object_reference_spacing,
    reference_spacing,
)


def unit(angle_degrees, dims=4, axes=(0, 1)):
    """A unit vector at an angle in the plane of two axes; cosine distance is 1 - cos(angle difference)."""
    vector = np.zeros(dims, dtype=np.float32)
    vector[axes[0]], vector[axes[1]] = np.cos(np.radians(angle_degrees)), np.sin(np.radians(angle_degrees))
    return vector


def distance(angle_degrees):
    return 1.0 - float(np.cos(np.radians(angle_degrees)))


def reference():
    """Confirmed scenes 10 degrees apart; confirmed boxes 20 degrees apart across images."""
    scenes = np.stack([unit(angle) for angle in (0, 10, 20, 30)])
    # Two boxes per image, 1 degree apart inside an image, 20 degrees between images.
    boxes = np.stack([unit(angle) for angle in (0, 1, 20, 21, 40, 41, 60, 61)])
    return scenes, boxes, [0, 0, 1, 1, 2, 2, 3, 3]


def test_object_spacing_skips_boxes_of_the_same_image():
    scenes, boxes, owner = reference()
    assert reference_spacing(scenes) == pytest.approx(distance(10), rel=1e-4)
    # Nearest box in another image is 19 or 20 degrees away, never the 1-degree sibling.
    assert object_reference_spacing(boxes, owner) == pytest.approx(distance(19), rel=1e-3)
    assert object_reference_spacing(boxes, [0] * len(owner)) is None
    assert nearest_distances(np.stack([unit(5)]), scenes)[0] == pytest.approx(distance(5), rel=1e-4)


def test_scores_are_in_typical_spacing_and_mix_scene_with_the_three_newest_boxes():
    scenes, boxes, owner = reference()
    candidate_scenes = np.stack([unit(0), unit(0), unit(50)])
    # Image 0: four boxes, one familiar and three new. Image 1: no detections. Image 2: one familiar box.
    candidate_boxes = np.stack([unit(0), unit(90), unit(100), unit(110), unit(20)])
    ranked = object_aware_ranking(candidate_scenes, scenes, boxes, owner, candidate_boxes, [0, 0, 0, 0, 2], object_weight=0.5)
    scene_spacing, object_spacing = distance(10), distance(19)
    assert ranked["scene_spacing"] == pytest.approx(scene_spacing, rel=1e-3)
    assert ranked["novelty_cutoff"] == 1.0 and ranked["novelty_cutoff_mode"] == "combined_spacing"
    top_three = np.mean([distance(90 - 61), distance(100 - 61), distance(110 - 61)]) / object_spacing
    scene_0, object_0, combined_0 = ranked["initial"][0]
    assert scene_0 == pytest.approx(0.0, abs=1e-4) and object_0 == pytest.approx(top_three, rel=1e-3)
    assert combined_0 == pytest.approx(0.5 * top_three, rel=1e-3)
    assert list(ranked["box_top"][:4]) == [False, True, True, True]
    # No detections: not treated as empty of objects, scored on the scene alone.
    assert ranked["initial"][1] == (pytest.approx(0.0, abs=1e-4), None, pytest.approx(0.0, abs=1e-4))
    # A new scene with only a familiar object: half the scene score.
    scene_2, object_2, combined_2 = ranked["initial"][2]
    assert scene_2 == pytest.approx(distance(20) / scene_spacing, rel=1e-3) and object_2 == pytest.approx(0.0, abs=1e-3)
    assert combined_2 == pytest.approx(0.5 * scene_2, rel=1e-3)
    assert ranked["selected"] == [0, 2] and ranked["reasons"] == ["objects", None, "scene"]
    # A wrong predicted class cannot matter: the search has no class input at all.
    scene_only = object_aware_ranking(candidate_scenes, scenes, boxes, owner, candidate_boxes, [0, 0, 0, 0, 2], object_weight=0.0)
    assert scene_only["selected"] == [2]


def test_selected_image_covers_its_scene_and_boxes_for_the_rest():
    scenes, boxes, owner = reference()
    # Five familiar scenes that all show the same new object.
    candidate_scenes = np.stack([unit(0)] * 5)
    candidate_boxes = np.stack([unit(90, axes=(2, 3))] * 5)
    ranked = object_aware_ranking(candidate_scenes, scenes, boxes, owner, candidate_boxes, list(range(5)), object_weight=0.5)
    assert ranked["selected"] == [0]
    assert all(ranked["initial"][index][2] > 1.0 for index in range(5))
    assert all(ranked["final"][index][2] == pytest.approx(0.0, abs=1e-4) for index in range(1, 5))
    # Without a cutoff the keep fraction sets the count.
    share = object_aware_ranking(candidate_scenes, scenes, boxes, owner, candidate_boxes, list(range(5)), novelty_cutoff="off", keep_fraction=0.4)
    assert len(share["selected"]) == 2 and share["novelty_cutoff_mode"] == "keep_fraction"
    with pytest.raises(ValueError, match="novelty_cutoff_invalid"):
        object_aware_ranking(candidate_scenes, scenes, boxes, owner, candidate_boxes, list(range(5)), novelty_cutoff=1000)


def test_object_mode_refuses_without_a_usable_reference():
    scenes, boxes, owner = reference()
    with pytest.raises(ValueError, match="object_spacing_unavailable"):
        object_aware_ranking(scenes[:1], scenes, boxes[:2], [0, 0], boxes[:1], [0])
    with pytest.raises(ValueError, match="object_scene_spacing_unavailable"):
        object_aware_ranking(scenes[:1], scenes[:1], boxes, owner, boxes[:1], [0])
