import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import localinferenceapi as api


class ClassAnalysisQualityApiTests(unittest.TestCase):
    def _result(self):
        points = []
        for index in range(70):
            disposition = ""
            sampling_source = ""
            if index < 30:
                disposition = "reassign_class" if index < 6 else "confirm_current"
                sampling_source = "audit" if index < 3 else "priority"
            points.append(
                {
                    "point_id": f"point-{index:03d}",
                    "class_name": "transit_object" if index % 2 == 0 else "pole",
                    "human_review_disposition": disposition,
                    "human_review_revision": f"rdr1_{index:032x}",
                    "review_sampling_source": sampling_source,
                    "quality_review_candidate": 30 <= index < 55,
                    "quality_score": 1.0 - index / 100.0,
                    "review_priority_score": 1.0 - index / 100.0,
                    "compact_logistic_disagreement": (index % 7) / 7.0,
                    "compact_neighbor_disagreement": (index % 5) / 5.0,
                    "proposed_class_differs": index % 3 == 0,
                }
            )
        return {
            "summary": {
                "analysis_input_digest": "a" * 64,
                "analysis_job_id": "ca_quality_api",
            },
            "points": points,
            "wrong_class_candidates": [dict(point) for point in points],
            "refinement_candidates": [],
            "vignette_candidates": [],
        }

    def test_ranking_sidecar_is_analysis_and_revision_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_path = Path(temporary) / "result.json"
            job = api.ClassAnalysisJob(
                job_id="ca_quality_api",
                status="completed",
                request={},
                result_path=str(result_path),
            )
            result = self._result()
            with mock.patch.object(api, "_get_class_analysis_job", return_value=job), mock.patch.object(
                api,
                "get_class_analysis_result",
                return_value=copy.deepcopy(result),
            ):
                ranking = api.recalibrate_class_analysis_review_ranking(
                    job.job_id,
                    {"analysis_digest": "a" * 64, "seed": 11},
                )
                self.assertTrue(ranking["enabled"])
                self.assertTrue(ranking["ready"])
                self.assertTrue(ranking["audit_point_ids"])
                real_ids = {point["point_id"] for point in result["points"]}
                self.assertTrue(set(ranking["ordered_point_ids"]).issubset(real_ids))
                self.assertTrue((Path(temporary) / "review_ranking.json").is_file())

                overlaid = api._class_analysis_apply_adaptive_review_ranking(
                    copy.deepcopy(result),
                    job=job,
                )
                ranked_points = [
                    point for point in overlaid["points"]
                    if point.get("adaptive_review_rank") is not None
                ]
                self.assertTrue(ranked_points)
                self.assertNotIn(
                    "adaptive_review_rank",
                    overlaid["wrong_class_candidates"][0],
                )
                self.assertEqual(
                    overlaid["adaptive_review_ranking"]["review_state_digest"],
                    ranking["review_state_digest"],
                )
                with self.assertRaises(api.HTTPException) as stale:
                    api.reset_class_analysis_review_ranking(
                        job.job_id,
                        {
                            "ranking_revision": "dqr1_" + "0" * 32,
                            "analysis_digest": "a" * 64,
                        },
                    )
                self.assertEqual(stale.exception.status_code, 409)
                reset = api.reset_class_analysis_review_ranking(
                    job.job_id,
                    {
                        "ranking_revision": ranking["ranking_revision"],
                        "analysis_digest": "a" * 64,
                    },
                )
                self.assertTrue(reset["removed"])
                self.assertFalse((Path(temporary) / "review_ranking.json").exists())

    def test_router_exposes_explicit_ranking_operations(self):
        paths = {route.path for route in api.app.routes}
        self.assertIn(
            "/class_analysis/jobs/{job_id}/review-ranking/recalibrate",
            paths,
        )
        self.assertIn(
            "/class_analysis/jobs/{job_id}/review-ranking",
            paths,
        )
        self.assertIn("/class_analysis/preflight", paths)

    def test_full_preflight_reports_confirmation_without_blocking_memory_limited(self):
        with mock.patch.object(api, "available_quality_memory_mb", return_value=1024):
            full = api._class_analysis_quality_preflight(
                {"record_count": 100000, "quality_memory_policy": "full"}
            )
            bounded = api._class_analysis_quality_preflight(
                {
                    "record_count": 100000,
                    "quality_memory_policy": "budgeted",
                    "quality_memory_budget_mb": 1024,
                }
            )
        self.assertTrue(full["requires_confirmation"])
        self.assertEqual(full["schema"], "class-analysis-preflight-v2")
        self.assertFalse(bounded["requires_confirmation"])
        self.assertEqual(
            bounded["algorithm_plan"]["neighbors"],
            "low_memory_ann_with_exact_fallback",
        )

    def test_auto_preflight_uses_budgeted_plan_for_quadratic_runs(self):
        with mock.patch.object(api, "available_quality_memory_mb", return_value=1_000_000):
            preflight = api._class_analysis_quality_preflight(
                {
                    "record_count": api.CLASS_ANALYSIS_EXACT_NEIGHBOR_LIMIT + 1,
                    "quality_memory_policy": "auto",
                    "quality_memory_budget_mb": api.QUALITY_MEMORY_MAX_MB,
                }
            )
        self.assertEqual(preflight["resolved_policy"], "budgeted")
        self.assertNotEqual(preflight["runtime_risk"], "normal")
        self.assertFalse(preflight["requires_confirmation"])

    def test_preflight_uses_server_source_count_when_available(self):
        source = {
            "manifest": {
                "images": [
                    {
                        "label_lines": [
                            "0 0.5 0.5 0.1 0.1",
                            "1 0.5 0.5 0.1 0.1",
                        ]
                    }
                ]
            },
            "labelmap": ["transit_object", "pole"],
        }
        with mock.patch.object(api, "_class_analysis_source", return_value=source):
            preflight = api._class_analysis_quality_preflight(
                {
                    "analysis_scope": "selected_class",
                    "class_name": "transit_object",
                    "record_count": 999,
                    "quality_memory_policy": "budgeted",
                    "quality_memory_budget_mb": 1024,
                }
            )
        self.assertEqual(preflight["record_count"], 1)
        self.assertEqual(preflight["record_count_source"], "server_source")

    def test_preflight_confirmation_digest_ignores_count_provenance(self):
        payload = {
            "record_count": 2,
            "quality_memory_policy": "full",
        }
        source = {
            "manifest": {
                "images": [
                    {
                        "label_lines": [
                            "0 0.5 0.5 0.1 0.1",
                            "0 0.4 0.4 0.1 0.1",
                        ]
                    }
                ]
            },
            "labelmap": ["transit_object"],
        }
        with mock.patch.object(api, "available_quality_memory_mb", return_value=4096):
            with mock.patch.object(
                api,
                "_class_analysis_source",
                side_effect=RuntimeError("source not uploaded"),
            ):
                client = api._class_analysis_quality_preflight(payload)
            with mock.patch.object(api, "_class_analysis_source", return_value=source):
                server = api._class_analysis_quality_preflight(payload)
        self.assertEqual(client["record_count"], server["record_count"])
        self.assertNotEqual(client["record_count_source"], server["record_count_source"])
        self.assertEqual(client["preflight_digest"], server["preflight_digest"])

    def test_run_fingerprint_ignores_preflight_telemetry_and_browser_count(self):
        source = {
            "source_mode": "linked",
            "source_id": "dataset",
            "dataset_root": Path("/unused"),
            "labelmap": ["transit_object"],
            "manifest": {
                "images": [
                    {
                        "split": "train",
                        "image_relpath": "frame.jpg",
                        "image_sha256": "b" * 64,
                        "label_lines": ["0 0.5 0.5 0.1 0.1"],
                    }
                ]
            },
        }
        stable = {
            "quality_memory_policy": "budgeted",
            "quality_memory_budget_mb": 2048,
            "quality_execution_plan_id": "plan-v2",
            "quality_resolved_memory_policy": "budgeted",
            "quality_resolved_memory_budget_mb": 2048,
        }
        with mock.patch.object(api, "_class_analysis_source", return_value=source):
            first = api._class_analysis_run_fingerprint(
                {
                    **stable,
                    "record_count": 1,
                    "quality_preflight": {"available_memory_mb": 1000},
                }
            )
            second = api._class_analysis_run_fingerprint(
                {
                    **stable,
                    "record_count": 999,
                    "quality_preflight": {"available_memory_mb": 2000},
                    "quality_preflight_digest": "volatile",
                    "quality_full_warning_acknowledged": True,
                }
            )
        self.assertEqual(first, second)

    def test_sqlite_point_store_round_trips_and_streams_public_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            result_path = out_dir / "result.json"
            points = [
                {
                    "point_id": f"point-{index}",
                    "class_name": "transit_object",
                    "projection": [index, index + 0.5],
                    "quality_score": index / 10,
                }
                for index in range(3)
            ]
            descriptor = api._class_analysis_write_points_store(
                out_dir / api.CLASS_ANALYSIS_POINTS_DB_FILENAME,
                out_dir,
                points,
            )
            with mock.patch.object(
                api,
                "_class_analysis_binary_artifact_snapshot",
                wraps=api._class_analysis_binary_artifact_snapshot,
            ) as snapshot:
                loaded = api._class_analysis_load_points_store(
                    result_path,
                    descriptor,
                )
                self.assertEqual(snapshot.call_count, 1)
            self.assertEqual([point["point_id"] for point in loaded], ["point-0", "point-1", "point-2"])
            with mock.patch.object(
                api,
                "_class_analysis_binary_artifact_snapshot",
                wraps=api._class_analysis_binary_artifact_snapshot,
            ) as snapshot:
                encoded = "".join(
                    api._class_analysis_stream_result_json(
                        {"summary": {"object_count": 3}, "points_storage": descriptor},
                        result_path=result_path,
                        descriptor=descriptor,
                        overlays={"point-1": {"human_review_disposition": "confirm_current"}},
                        ranking_rows={"point-2": {"adaptive_review_score": 0.9}},
                        ranking_order={"point-2": 0},
                    )
                )
                self.assertEqual(snapshot.call_count, 1)
            public = json.loads(encoded)
            self.assertEqual(len(public["points"]), 3)
            self.assertNotIn("points_storage", public)
            self.assertEqual(public["points"][1]["human_review_disposition"], "confirm_current")
            self.assertEqual(public["points"][2]["adaptive_review_rank"], 0)

            store_path, connection, identity = (
                api._class_analysis_open_validated_points_store(
                    result_path,
                    descriptor,
                )
            )
            early_stream = api._class_analysis_stream_result_json(
                {"summary": {"object_count": 3}, "points_storage": descriptor},
                result_path=result_path,
                descriptor=descriptor,
                overlays={},
                ranking_rows={},
                ranking_order={},
                connection=connection,
                store_path=store_path,
                store_identity=identity,
                close_connection=True,
            )
            self.assertEqual(next(early_stream), "{")
            early_stream.close()
            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")

            with (out_dir / api.CLASS_ANALYSIS_POINTS_DB_FILENAME).open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaises(api.HTTPException) as changed:
                api._class_analysis_load_points_store(result_path, descriptor)
            self.assertEqual(changed.exception.status_code, 409)

    def test_non_candidate_receipt_is_discovered_and_overlaid(self):
        review_key = "cro_" + "a" * 64
        revision = "rdr1_" + "b" * 32
        point = {
            "point_id": "ordinary-point",
            "class_name": "transit_object",
            "review_object_key": review_key,
        }
        entry = {
            "disposition": "skip",
            "updated_at": 123.0,
            "entry_revision": revision,
            "origin": "desktop",
        }
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            result_path = out_dir / "result.json"
            descriptor = api._class_analysis_write_points_store(
                out_dir / api.CLASS_ANALYSIS_POINTS_DB_FILENAME,
                out_dir,
                [point],
            )
            store_path, connection, identity = (
                api._class_analysis_open_validated_points_store(
                    result_path,
                    descriptor,
                )
            )
            try:
                with (
                    mock.patch.object(
                        api,
                        "_class_analysis_review_disposition_root",
                        return_value=mock.sentinel.ledger_root,
                    ),
                    mock.patch.object(
                        api,
                        "_class_analysis_row_has_review_disposition",
                        return_value=True,
                    ),
                    mock.patch.object(
                        api,
                        "_class_analysis_lookup_review_dispositions",
                        return_value={review_key: entry},
                    ),
                ):
                    reviewed_ids = api._class_analysis_reviewed_point_ids_from_connection(
                        connection,
                        "",
                    )
                    loaded_points = api._class_analysis_load_points_from_connection(
                        connection,
                        descriptor,
                        point_ids=reviewed_ids,
                    )
                    overlaid = api._class_analysis_apply_review_dispositions(
                        {
                            "summary": {},
                            "points": loaded_points,
                            "wrong_class_candidates": [],
                            "refinement_candidates": [],
                            "vignette_candidates": [],
                            "within_class_outlier_candidates": [],
                        }
                    )
                    public = api._class_analysis_public_result(overlaid)
            finally:
                connection.close()
            api._class_analysis_assert_points_store_identity(store_path, identity)
        self.assertEqual(reviewed_ids, [point["point_id"]])
        self.assertEqual(
            overlaid["points"][0]["human_review_disposition"],
            "skip",
        )
        self.assertEqual(overlaid["points"][0]["human_review_revision"], revision)
        self.assertNotIn("review_object_key", public["points"][0])
        self.assertEqual(public["points"][0]["human_review_disposition"], "skip")

    def test_legacy_points_store_receipt_discovery_streams_bound_metadata(self):
        point = {
            "point_id": "legacy-point",
            "class_name": "transit_object",
            "review_object_key": "cro_" + "c" * 64,
        }
        connection = sqlite3.connect(":memory:")
        connection.execute(
            "CREATE TABLE points (ordinal INTEGER, point_id TEXT, payload TEXT)"
        )
        connection.execute(
            "INSERT INTO points VALUES (?, ?, ?)",
            (0, point["point_id"], json.dumps({"point_id": point["point_id"]})),
        )
        try:
            with (
                mock.patch.object(
                    api,
                    "_class_analysis_review_disposition_root",
                    return_value=mock.sentinel.ledger_root,
                ),
                mock.patch.object(
                    api,
                    "_class_analysis_iter_bound_metadata_rows",
                    return_value=iter([(0, point)]),
                ) as metadata_rows,
                mock.patch.object(
                    api,
                    "_class_analysis_row_has_review_disposition",
                    return_value=True,
                ) as has_receipt,
            ):
                reviewed_ids = api._class_analysis_reviewed_point_ids_from_connection(
                    connection,
                    "legacy-job",
                    job=mock.sentinel.job,
                )
        finally:
            connection.close()
        self.assertEqual(reviewed_ids, [point["point_id"]])
        metadata_rows.assert_called_once_with(mock.sentinel.job)
        has_receipt.assert_called_once()

    def test_reviewed_point_scan_returns_before_reading_rows_without_ledger(self):
        connection = sqlite3.connect(":memory:")
        connection.execute(
            "CREATE TABLE points ("
            "ordinal INTEGER, point_id TEXT, payload TEXT, "
            "review_object_key TEXT, pair_review_key TEXT)"
        )
        connection.execute(
            "INSERT INTO points VALUES (?, ?, ?, ?, ?)",
            (0, "p0", json.dumps({"point_id": "p0"}), "cro_" + "d" * 64, ""),
        )
        try:
            with (
                mock.patch.object(
                    api,
                    "_class_analysis_review_disposition_root",
                    return_value=None,
                ),
                mock.patch.object(
                    api,
                    "_class_analysis_row_has_review_disposition",
                ) as has_receipt,
            ):
                reviewed_ids = api._class_analysis_reviewed_point_ids_from_connection(
                    connection,
                    "job-without-ledger",
                )
        finally:
            connection.close()
        self.assertEqual(reviewed_ids, [])
        has_receipt.assert_not_called()

    def test_data_quality_explorer_assets_are_served_by_production_routes(self):
        for asset_name in (
            "class_analysis_cache_admin.js",
            "class_split_controls.css",
            "class_split_graph_view.js",
        ):
            response = api.serve_tator_ui_asset(asset_name)
            asset_path = Path(response.path)
            self.assertEqual(asset_path.name, asset_name)
            self.assertTrue(asset_path.is_file())


if __name__ == "__main__":
    unittest.main()
