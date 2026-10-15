"""Native analysis read models over retained, immutable artifacts.

Reads reuse the actual normalized DQE session store and pruning pair store.
No legacy app, folder-derived membership or legacy mutation endpoint is used.
"""
from contextlib import contextmanager
import io
import json
from pathlib import Path
import tempfile

from .artifacts import ArtifactRef
from .errors import DomainError


class AnalysisViews:
    def __init__(self, analysis):
        self.analysis = analysis
        self.store, self.artifacts = analysis.store, analysis.artifacts
        self.session_provider = None

    def metadata(self, actor, job_id):
        status = self.analysis.status(actor, job_id)
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            row = con.execute("SELECT payload_json FROM hosted_analysis_views WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise DomainError("analysis_view_unavailable", 409)
        return {**json.loads(row[0]), "job_id": job_id, "stale": status["stale"]}

    def _reference(self, actor, job_id, relative):
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            row = con.execute("SELECT sha256,size_bytes FROM hosted_analysis_artifacts WHERE job_id=? AND relative_path=?", (job_id, relative)).fetchone()
            if row is None:
                raise DomainError("analysis_artifact_unavailable", 404)
            return ArtifactRef(row[0], row[1])

    @contextmanager
    def _file(self, actor, job_id, relative):
        if self.session_provider is not None and relative.endswith('/session.sqlite3'):
            with self.session_provider(actor, job_id, relative) as path:
                yield path
            return
        reference = self._reference(actor, job_id, relative)
        # A private immutable pin supplies a filename to existing read-only SQL
        # helpers. The original native directory can already have disappeared.
        with tempfile.TemporaryDirectory(prefix="hosted-analysis-read-", dir=self.analysis.root) as directory:
            self.artifacts.pin(reference, directory)
            yield Path(directory) / reference.sha256

    def _session_path(self, actor, job_id):
        meta = self.metadata(actor, job_id)
        if meta["consumer"] != "dqe" or not meta.get("native_job_id"):
            raise DomainError("analysis_consumer_mismatch", 409)
        return f"class_analysis/{meta['native_job_id']}/session.sqlite3"

    def _call_session(self, actor, job_id, function, *args, **kwargs):
        from services.class_analysis_session_store import SessionStoreError
        try:
            with self._file(actor, job_id, self._session_path(actor, job_id)) as path:
                result = function(path, *args, **kwargs)
        except SessionStoreError as error:
            raise DomainError(error.detail, error.status_code) from error
        return {**result, "hosted_job_id": job_id, "stale": self.analysis.status(actor, job_id)["stale"]}

    def graph(self, actor, job_id, **filters):
        from services.class_analysis_session_store import get_class_analysis_graph_payload
        return self._call_session(actor, job_id, get_class_analysis_graph_payload, **filters)

    def queue(self, actor, job_id, **pagination):
        from services.class_analysis_session_store import get_class_analysis_review_queue_payload
        return self._call_session(actor, job_id, get_class_analysis_review_queue_payload, **pagination)

    def point(self, actor, job_id, point_id):
        from services.class_analysis_session_store import get_class_analysis_point_detail_payload
        result = self._call_session(actor, job_id, get_class_analysis_point_detail_payload, point_id)
        point = result["point"]
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            _, manifest, _ = self.analysis._job(con, job_id)
        # The frozen export uses globally unique hosted IDs as basenames.
        identifier = Path(point.get("image_relpath", "")).stem
        image = next((row for row in manifest["images"] if row["id"] == identifier), None)
        ordinal = point.get("label_line_index")
        if image is None or type(ordinal) is not int or not 0 <= ordinal < len(image["annotations"]["objects"]):
            raise DomainError("analysis_point_identity_unavailable", 409)
        obj = image["annotations"]["objects"][ordinal]
        result["hosted_identity"] = {"dataset_id": manifest["dataset_id"], "image_id": identifier,
            "version_id": image["version_id"], "object_id": obj["id"], "class_id": obj["class_id"], "bbox": obj["bbox"]}
        return result

    def evidence(self, actor, job_id, point_id):
        from services.class_analysis_session_store import get_class_analysis_point_evidence_payload
        return self._call_session(actor, job_id, get_class_analysis_point_evidence_payload, point_id)

    def thumbnail(self, actor, job_id, point_id, *, context=False, size=256):
        from PIL import Image
        if type(size) is not int or not 32 <= size <= 1024:
            raise DomainError("analysis_thumbnail_size_invalid", 422)
        identity = self.point(actor, job_id, point_id)["hosted_identity"]
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            _, manifest, _ = self.analysis._job(con, job_id)
        image = next(row for row in manifest["images"] if row["id"] == identity["image_id"])
        with self.artifacts.open(ArtifactRef(image["media_hash"], image["size_bytes"])) as stream, Image.open(stream) as decoded:
            view = decoded.convert("RGB")
            if not context:
                x, y, w, h = identity["bbox"]
                view = view.crop((x, y, x + w, y + h))
            view.thumbnail((size, size))
            result = io.BytesIO()
            view.save(result, format="JPEG", quality=88)
            return result.getvalue()

    def _pruning_file(self, actor, job_id, filename):
        meta = self.metadata(actor, job_id)
        if meta["consumer"] != "pruning" or not meta.get("native_job_id"):
            raise DomainError("analysis_consumer_mismatch", 409)
        return f"class_analysis/pruning/{meta['native_job_id']}/{filename}"

    def pruning_families(self, actor, job_id):
        from services.dataset_pruning_evidence import PairStore
        from services.dataset_pruning_review import family_inventory
        with self._file(actor, job_id, self._pruning_file(actor, job_id, "images.json")) as path:
            images = json.loads(path.read_text())
        with self._file(actor, job_id, self._pruning_file(actor, job_id, "evidence.sqlite3")) as path:
            families = family_inventory(images, PairStore(path))
        return {"families": families, "images": images, "stale": self.analysis.status(actor, job_id)["stale"]}

    def pruning_pair(self, actor, job_id, left, right):
        from services.dataset_pruning_evidence import PairStore
        if type(left) is not int or type(right) is not int or min(left, right) < 0 or left == right:
            raise DomainError("analysis_pair_invalid", 422)
        with self._file(actor, job_id, self._pruning_file(actor, job_id, "evidence.sqlite3")) as path:
            result = PairStore(path).pair(left, right)
        if result is None:
            raise DomainError("analysis_pair_unavailable", 404)
        return {"pair": result, "stale": self.analysis.status(actor, job_id)["stale"]}
