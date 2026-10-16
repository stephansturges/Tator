"""Current read authority for immutable command results, including old receipts.

This registry deliberately denies unknown operations. Adding a mutation requires
choosing its disclosure policy; receipts themselves are never rewritten.
"""
import json

from .errors import DomainError


ADMIN_OPERATIONS = frozenset({
    "create_dataset", "set_dataset_settings", "patch_dataset_settings", "import_image", "import_yolo_image",
    "resolve_import_conflict", "assign", "set_splits", "approve", "approve_from_review_queue", "return",
    "override", "recall_images", "reassign", "override_assignment", "add_class", "rename_class",
    "resolve_class_request", "create_snapshot", "register_artifact", "archive_image", "archive_dataset",
    "restore_archive", "purge_archive", "purge_orphans", "export_canonical", "recovery_backup", "start_migration",
    "migration_image_done", "finish_migration", "install_tool_model", "request_heavy_tool",
    "capture_ingestion_candidates", "capture_analysis", "analysis_terminal", "publish_analysis_result",
    "publish_native_analysis", "apply_analysis_annotations", "dqe_apply_objects", "dqe_disposition",
    "dqe_create_batch", "dqe_resume_batch", "dqe_cancel_batch", "salad_profiles", "register_salad_head",
    "configure_provider", "reconcile_provider_call", "review_caption_instructions", "caption_batch_start",
    "apply_caption_proposal", "caption_batch_cancel", "caption_batch_resume",
})
IMAGE_OPERATIONS = frozenset({"save", "release_editor", "submit", "paid_batch_apply"})
CAPTION_OPERATIONS = frozenset({"start_caption_job", "cancel_caption_job", "resume_caption_job"})
PAID_OPERATIONS = frozenset({"paid_batch_start", "paid_batch_cancel", "paid_batch_archive", "paid_batch_resume", "paid_batch_recover"})


def can_disclose(store, con, actor, operation, result):
    def image(identifier):
        store._can_view(actor, store._image(con, identifier))

    def dataset(identifier):
        store._dataset(con, identifier)
        if actor.role != "admin" and not con.execute(
            "SELECT 1 FROM hosted_images i WHERE dataset_id=? AND assignee_id=? "
            "AND state IN ('assigned','returned') AND NOT EXISTS "
            "(SELECT 1 FROM hosted_archived_images a WHERE a.image_id=i.id) LIMIT 1",
            (identifier, actor.user_id),
        ).fetchone():
            raise DomainError("dataset_not_found", 404)

    def job(table):
        if not con.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone():
            return None
        # Table names come only from the closed operation registry below.
        return con.execute(f"SELECT * FROM {table} WHERE id=? AND actor_id=?", (result.get("job_id"), actor.user_id)).fetchone()

    try:
        if operation in ADMIN_OPERATIONS:
            return actor.role == "admin"
        if not isinstance(result, dict):
            return False
        if operation in IMAGE_OPERATIONS:
            image(result.get("id"))
        elif operation == "request_class":
            dataset(result.get("dataset_id"))
        elif operation == "select_user_sam":
            # User-private preference. The receipt lookup already checks owner.
            return True
        elif operation in {"save_tool_cascade", "delete_tool_cascade"}:
            if not con.execute("SELECT 1 FROM sqlite_master WHERE name='hosted_tool_cascades'").fetchone():
                return False
            row = con.execute("SELECT dataset_id FROM hosted_tool_cascades WHERE id=? AND actor_id=?", (result.get("id"), actor.user_id)).fetchone()
            if row is None:
                return False
            dataset(row["dataset_id"])
        elif operation in CAPTION_OPERATIONS:
            row = job("hosted_caption_jobs")
            if row is None:
                return False
            request = json.loads(row["request_json"])
            dataset(request.get("dataset_id"))
            image(request.get("image_id"))
        elif operation in PAID_OPERATIONS:
            row = job("hosted_paid_batches")
            if row is None:
                return False
            manifest = json.loads(row["manifest_json"])
            dataset(manifest.get("dataset_id"))
            for member in manifest["images"]:
                image(member["id"])
        else:
            return False
        return True
    except (DomainError, KeyError, TypeError, ValueError):
        # Old or malformed receipts cannot grant new resource authority.
        return False
