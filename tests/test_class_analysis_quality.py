import unittest
from tempfile import TemporaryDirectory

import numpy as np

from utils.class_analysis_quality import (
    THOROUGH_QUALITY_RECIPE,
    apply_quality_rows,
    available_quality_memory_mb,
    build_adaptive_review_ranking,
    compute_size_evidence,
    merge_quality_features,
    plan_quality_execution,
    resolve_quality_recipe,
    score_quality_records,
)


class ClassAnalysisQualityTests(unittest.TestCase):
    def _records(self, count=80):
        rows = []
        for index in range(count):
            label = "transit_object" if index % 2 == 0 else "pole"
            rows.append(
                {
                    "point_id": f"point-{index:03d}",
                    "class_id": label,
                    "source_image_id": f"image-{index // 4:03d}",
                    "bbox": [0, 0, 12 if index < 8 else 80, 18 if index < 8 else 90],
                    "source_width": 1000,
                    "source_height": 1000,
                }
            )
        return rows

    def _features(self, records, dimensions=12):
        rng = np.random.default_rng(7)
        values = rng.normal(0.0, 0.08, (len(records), dimensions)).astype(np.float32)
        for index, record in enumerate(records):
            values[index, 0] += -1.0 if record["class_id"] == "transit_object" else 1.0
        values[11, 0] *= -1.0
        return values

    def test_recipe_aliases_keep_thorough_as_default(self):
        self.assertEqual(resolve_quality_recipe(None).recipe_id, THOROUGH_QUALITY_RECIPE)
        self.assertEqual(resolve_quality_recipe("balanced").recipe_id, THOROUGH_QUALITY_RECIPE)
        self.assertTrue(resolve_quality_recipe("thorough_quality_v1").use_cradio)

    def test_custom_recipe_honors_normalized_multibackbone_weights(self):
        recipe = resolve_quality_recipe(
            "custom",
            {
                "quality_use_cradio": True,
                "quality_use_el2n": True,
                "quality_compact_weight": 3.0,
                "quality_cradio_weight": 1.0,
                "quality_late_compact_weight": 6.0,
                "quality_late_cradio_weight": 4.0,
                "quality_late_weight": 7.0,
                "quality_el2n_weight": 3.0,
            },
        )
        self.assertEqual(recipe.recipe_id, "custom")
        self.assertTrue(recipe.use_cradio)
        self.assertTrue(recipe.use_el2n)
        self.assertAlmostEqual(recipe.compact_weight, 0.5)
        self.assertAlmostEqual(recipe.cradio_weight, 0.5)
        self.assertAlmostEqual(recipe.late_compact_weight, 0.5)
        self.assertAlmostEqual(recipe.late_cradio_weight, 0.5)
        self.assertAlmostEqual(recipe.late_weight, 0.5)
        self.assertAlmostEqual(recipe.el2n_weight, 0.5)

    def test_feature_merge_is_aligned_and_normalized(self):
        compact = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        cradio = np.asarray([[0.0, 1.0], [3.0, 0.0]], dtype=np.float32)
        merged = merge_quality_features(compact, cradio)
        self.assertEqual(merged.shape, (2, 4))
        np.testing.assert_allclose(np.linalg.norm(merged, axis=1), np.ones(2), atol=1e-6)

    def test_tiny_evidence_uses_source_pixels_and_global_quintile(self):
        evidence = compute_size_evidence(self._records())
        self.assertTrue(all(row["tiny_object"] for row in evidence[:8]))
        self.assertTrue(all(row["low_source_detail"] for row in evidence[:8]))
        self.assertTrue(any(not row["tiny_object"] for row in evidence[20:]))

    def test_equal_large_boxes_are_neutral_not_low_detail(self):
        records = [
            {
                "point_id": f"equal-{index}",
                "class_id": "transit_object",
                "bbox_xyxy": [10, 10, 210, 210],
                "source_width": 1000,
                "source_height": 1000,
            }
            for index in range(20)
        ]
        evidence = compute_size_evidence(records)
        self.assertTrue(all(row["bbox_global_percentile"] == 0.5 for row in evidence))
        self.assertTrue(all(not row["tiny_object"] for row in evidence))
        self.assertTrue(all(not row["relative_small_object"] for row in evidence))

    def test_missing_source_dimensions_do_not_mix_pixel_and_fraction_units(self):
        records = [
            {
                "point_id": "known",
                "class_id": "transit_object",
                "bbox_xyxy": [0, 0, 100, 100],
                "source_width": 1000,
                "source_height": 1000,
            },
            {
                "point_id": "unknown",
                "class_id": "transit_object",
                "bbox_xyxy": [0, 0, 100, 100],
            },
        ]
        evidence = compute_size_evidence(records)
        self.assertEqual(evidence[0]["bbox_source_area"], 0.01)
        self.assertIsNone(evidence[1]["bbox_source_area"])
        self.assertEqual(evidence[0]["bbox_global_percentile"], 0.5)
        self.assertEqual(evidence[1]["bbox_global_percentile"], 0.5)

    def test_thorough_scoring_has_global_budget_and_real_proposals(self):
        records = self._records()
        compact = self._features(records)
        cradio = self._features(records, dimensions=10)[:, ::-1]
        merged, rows, review_ids = score_quality_records(
            records,
            compact,
            cradio,
            logistic_fit_limit=80,
            rbf_fit_limit=80,
            neighbour_reference_limit=80,
        )
        self.assertEqual(merged.shape[0], len(records))
        self.assertEqual(len(review_ids), 4)
        self.assertEqual(set(review_ids), {row["point_id"] for row in rows if row["quality_review_candidate"]})
        self.assertTrue(all(row["proposed_class"] in {"transit_object", "pole"} for row in rows))
        result = {"points": [dict(record) for record in records], "wrong_class_candidates": [], "summary": {}}
        apply_quality_rows(result, rows, review_ids)
        self.assertEqual(result["quality_review_queue"]["priority_ids"], review_ids)
        self.assertEqual(
            [row["point_id"] for row in result["wrong_class_candidates"]],
            review_ids,
        )
        self.assertTrue(result["quality_review_queue"]["tiny_ids"])
        self.assertEqual(
            result["quality_review_queue"]["tiny_ids"],
            result["quality_review_queue"]["low_detail_ids"],
        )
        self.assertEqual(
            result["quality_review_queue"]["schema"],
            "class-analysis-quality-review-queue-v2",
        )
        self.assertEqual(
            result["summary"]["size_evidence"]["tiny_object_semantics"],
            "legacy_alias_of_low_source_detail",
        )
        self.assertEqual(result["summary"]["quality_recipe"], THOROUGH_QUALITY_RECIPE)

    def test_low_detail_filter_does_not_duplicate_points_into_wrong_class_queue(self):
        records = self._records(40)
        result = {
            "points": [dict(record) for record in records],
            "wrong_class_candidates": [],
            "summary": {},
        }
        rows = [
            {
                "point_id": record["point_id"],
                "quality_score": 0.0,
                "low_source_detail": True,
                "relative_small_object": True,
            }
            for record in records
        ]
        apply_quality_rows(result, rows, [records[0]["point_id"]])
        self.assertEqual(
            [row["point_id"] for row in result["wrong_class_candidates"]],
            [records[0]["point_id"]],
        )
        self.assertEqual(
            result["quality_review_queue"]["low_detail_ids"],
            [record["point_id"] for record in records],
        )
        self.assertEqual(
            result["quality_review_queue"]["all_flagged_ids"],
            [record["point_id"] for record in records],
        )

    def test_quality_application_preserves_legacy_candidates(self):
        records = self._records()
        compact = self._features(records)
        cradio = self._features(records, dimensions=10)[:, ::-1]
        _, rows, review_ids = score_quality_records(
            records,
            compact,
            cradio,
            logistic_fit_limit=80,
            rbf_fit_limit=80,
            neighbour_reference_limit=80,
        )
        legacy = dict(records[-1])
        result = {
            "points": [dict(record) for record in records],
            "wrong_class_candidates": [legacy],
            "summary": {},
        }
        apply_quality_rows(result, rows, review_ids)
        ids = [row["point_id"] for row in result["wrong_class_candidates"]]
        self.assertIn(legacy["point_id"], ids)
        self.assertEqual(len(ids), len(set(ids)))
        preserved = next(row for row in result["wrong_class_candidates"] if row["point_id"] == legacy["point_id"])
        self.assertIn("legacy_heuristic", preserved["quality_flag_sources"])

    def test_all_equal_single_class_scores_abstain_without_fake_priority(self):
        records = self._records(40)
        for record in records:
            record["class_id"] = "transit_object"
        features = np.ones((40, 8), dtype=np.float32)
        _, rows, review_ids = score_quality_records(
            records,
            features,
            recipe_id="precise_compact_v1",
            logistic_fit_limit=40,
            neighbour_reference_limit=40,
        )
        self.assertEqual(review_ids, [])
        self.assertTrue(all(row["quality_score"] == 0.0 for row in rows))
        self.assertTrue(all(not row["quality_signal_available"] for row in rows))
        self.assertTrue(all(row["quality_abstention_reason"] == "no_discriminating_quality_signal" for row in rows))

    def test_class_absent_from_oof_training_abstains_from_proposal(self):
        records = self._records(48)
        for index, record in enumerate(records):
            record["class_id"] = "rare" if index < 4 else ("transit_object" if index % 2 == 0 else "pole")
            record["source_image_id"] = "rare-only-source" if index < 4 else f"source-{index // 4}"
        features = self._features(records)
        _, rows, _ = score_quality_records(
            records,
            features,
            recipe_id="precise_compact_v1",
            logistic_fit_limit=48,
            rbf_fit_limit=48,
            neighbour_reference_limit=48,
        )
        rare_rows = [row for row in rows if row["point_id"] in {record["point_id"] for record in records[:4]}]
        self.assertTrue(rare_rows)
        self.assertTrue(all(row["proposed_class"] is None for row in rare_rows))

    def test_memory_planner_keeps_full_and_budgeted_paths_explicit(self):
        available = available_quality_memory_mb()
        full = plan_quality_execution(
            policy="full",
            budget_mb=None,
            record_count=1_000_000,
            compact_dimensions=2944,
            cradio_dimensions=4608,
            available_mb=available,
        )
        bounded = plan_quality_execution(
            policy="budgeted",
            budget_mb=2048,
            record_count=1_000_000,
            compact_dimensions=2944,
            cradio_dimensions=4608,
            available_mb=available,
        )
        self.assertEqual(full["resolved_policy"], "full")
        self.assertIsNone(full["budget_mb"])
        self.assertEqual(bounded["resolved_policy"], "budgeted")
        self.assertEqual(bounded["budget_mb"], 2048)
        self.assertTrue(bounded["never_aborts_for_configured_budget"])

    def test_budgeted_scoring_uses_memmap_and_bounded_algorithms(self):
        records = self._records()
        metadata = {}
        with TemporaryDirectory() as scratch:
            merged, rows, review_ids = score_quality_records(
                records,
                self._features(records),
                self._features(records, dimensions=10)[:, ::-1],
                memory_policy="budgeted",
                memory_budget_mb=1024,
                scratch_dir=scratch,
                execution_metadata=metadata,
                logistic_fit_limit=80,
                rbf_fit_limit=80,
                neighbour_reference_limit=80,
            )
            self.assertIsInstance(merged, np.memmap)
            self.assertEqual(len(rows), len(records))
            self.assertTrue(review_ids)
            self.assertEqual(metadata["resolved_policy"], "budgeted")
            self.assertEqual(metadata["fusion_storage"], "temporary_memmap")
            self.assertEqual(metadata["neighbour_algorithm"], "pynndescent")
            self.assertEqual(metadata["proposal_algorithm"], "exact_fit_bounded_rbf_svc")

    def test_adaptive_ranking_inserts_only_real_lower_ranked_audits(self):
        points = []
        for index in range(70):
            disposition = ""
            sampling_source = ""
            if index < 30:
                disposition = "reassign" if index < 6 else "confirm"
                sampling_source = "audit" if index < 3 else "priority"
            points.append(
                {
                    "point_id": f"real-{index:03d}",
                    "class_id": "transit_object" if index % 2 == 0 else "pole",
                    "review_disposition": disposition,
                    "review_sampling_source": sampling_source,
                    "human_review_revision": index + 1,
                    "quality_score": 1.0 - index / 100.0,
                    "review_priority_score": 1.0 - index / 100.0,
                    "compact_logistic_disagreement": (index % 7) / 7.0,
                    "compact_neighbor_disagreement": (index % 5) / 5.0,
                    "proposed_class_differs": index % 3 == 0,
                    "quality_review_candidate": 30 <= index < 55,
                }
            )
        ranking = build_adaptive_review_ranking(points)
        self.assertTrue(ranking["ready"])
        real_ids = {point["point_id"] for point in points}
        self.assertTrue(set(ranking["ordered_point_ids"]).issubset(real_ids))
        self.assertTrue(set(ranking["audit_point_ids"]).issubset(real_ids))
        self.assertTrue(ranking["audit_point_ids"])
        first_audit = ranking["ordered_point_ids"].index(ranking["audit_point_ids"][0])
        self.assertEqual(first_audit, 9)

    def test_bootstrap_ranking_collects_real_audits_before_model_is_ready(self):
        points = [
            {
                "point_id": f"pending-{index:03d}",
                "class_name": "transit_object" if index % 2 == 0 else "pole",
                "human_review_disposition": "",
                "quality_score": 1.0 - index / 100.0,
                "review_priority_score": 1.0 - index / 100.0,
            }
            for index in range(40)
        ]
        ranking = build_adaptive_review_ranking(points)
        self.assertFalse(ranking["ready"])
        self.assertEqual(ranking["model"]["type"], "bootstrap_baseline")
        real_ids = {point["point_id"] for point in points}
        self.assertTrue(set(ranking["ordered_point_ids"]).issubset(real_ids))
        self.assertEqual(len(ranking["ordered_point_ids"]), 3)
        self.assertTrue(set(ranking["audit_point_ids"]).issubset(real_ids))
        self.assertEqual(ranking["ordered_point_ids"], ranking["audit_point_ids"])

    def test_adaptive_priority_preserves_every_strong_detector_source(self):
        points = [
            {"point_id": "legacy", "class_name": "transit_object", "is_wrong_class_candidate": True, "quality_score": 0.1},
            {"point_id": "overlap", "class_name": "transit_object", "is_dual_bbox_conflict": True, "quality_score": 0.2},
            {"point_id": "deep", "class_name": "pole", "refined_outlier": {"actionable": True}, "quality_score": 0.3},
            {"point_id": "quality", "class_name": "pole", "quality_review_candidate": True, "quality_score": 0.4},
            {"point_id": "audit", "class_name": "pole", "quality_score": 0.0},
        ]
        ranking = build_adaptive_review_ranking(points)
        ordered = ranking["ordered_point_ids"]
        self.assertTrue({"legacy", "overlap", "deep", "quality"}.issubset(set(ordered)))
        self.assertEqual(set(ranking["audit_point_ids"]), {"audit"})

    def test_single_outcome_keeps_bootstrap_queue(self):
        points = []
        for index in range(40):
            points.append(
                {
                    "point_id": f"single-{index}",
                    "class_name": "transit_object" if index % 2 else "pole",
                    "human_review_disposition": "reassign_class" if index < 30 else "",
                    "human_review_revision": index if index < 30 else "",
                    "review_sampling_source": "audit" if index < 3 else "priority",
                    "quality_review_candidate": index >= 30,
                    "quality_score": index / 40,
                }
            )
        ranking = build_adaptive_review_ranking(points)
        self.assertFalse(ranking["ready"])
        self.assertTrue(ranking["ordered_point_ids"])
        self.assertEqual(ranking["model"]["type"], "bootstrap_baseline")

    def test_production_model_roles_remain_separate(self):
        source = (
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "localinferenceapi.py"
        ).read_text(encoding="utf-8")
        salad_start = source.index("def _class_analysis_train_salad_head")
        salad_end = source.index("\ndef ", salad_start + 5)
        salad_source = source[salad_start:salad_end]
        self.assertIn('payload.get("encoder_model")', salad_source)
        self.assertNotIn("deep_evidence_encoder_model", salad_source)
        self.assertNotIn("class_analysis_refinement_requires_dinov3\"", source)
        self.assertGreaterEqual(source.count('request.get("deep_evidence_encoder_model")'), 3)
        self.assertIn('request_payload["quality_memory_policy"]', source)
        self.assertIn('job.request.get("quality_review_fraction")', source)


if __name__ == "__main__":
    unittest.main()
