import json
import subprocess
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.class_analysis import build_class_analysis_router
from services.class_analysis_cache_admin import (
    CacheBusyError,
    CacheRootUnsafeError,
    cache_inventory,
    purge_cache,
)


def test_cache_inventory_and_targeted_purge_preserve_non_regenerable_data(
    tmp_path: Path,
):
    cache_root = tmp_path / "cache"
    (cache_root / "image_packs").mkdir(parents=True)
    (cache_root / "resume_embeddings").mkdir()
    (cache_root / "patch_reference_banks").mkdir()
    (cache_root / "image_packs" / "crop.bin").write_bytes(b"crop")
    (cache_root / "resume_embeddings" / "resume.bin").write_bytes(
        b"embedding"
    )
    (cache_root / "patch_reference_banks" / "bank.bin").write_bytes(
        b"bank"
    )

    before = cache_inventory(cache_root, max_bytes=1024)
    assert before["total_bytes"] == 4 + 9 + 4
    assert before["managed_bytes"] == 4 + 9
    assert before["purgeable_bytes"] == 4 + 9
    assert before["protected_bytes"] == 4
    assert before["over_budget_bytes"] == 0
    assert before["usage_fraction"] == pytest.approx(13 / 1024)
    assert before["budget_scope"] == [
        "image_packs",
        "resume_embeddings",
    ]
    assert (
        before["categories"]["patch_reference_banks"]["purgeable"]
        is False
    )

    result = purge_cache(cache_root, max_bytes=1024)

    assert result["bytes_reclaimed"] == 13
    assert list((cache_root / "image_packs").iterdir()) == []
    assert list((cache_root / "resume_embeddings").iterdir()) == []
    assert (
        cache_root / "patch_reference_banks" / "bank.bin"
    ).read_bytes() == b"bank"


def test_cache_purge_refuses_active_jobs_and_category_symlinks(
    tmp_path: Path,
):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.bin").write_bytes(b"keep")

    with pytest.raises(CacheBusyError) as busy:
        purge_cache(
            cache_root,
            active_users_fn=lambda: [
                {
                    "kind": "analysis",
                    "job_id": "ca_busy",
                    "status": "running",
                }
            ],
        )
    assert busy.value.active_users[0]["job_id"] == "ca_busy"

    (cache_root / "image_packs").symlink_to(
        outside,
        target_is_directory=True,
    )
    with pytest.raises(CacheRootUnsafeError):
        purge_cache(cache_root, categories=["image_packs"])
    assert (outside / "keep.bin").read_bytes() == b"keep"


def _router(cache_status_fn, purge_cache_fn):
    def unused(*_args, **_kwargs):
        return {}

    return build_class_analysis_router(
        capabilities_fn=unused,
        create_job_fn=unused,
        create_active_workspace_job_fn=unused,
        start_active_workspace_upload_fn=unused,
        batch_active_workspace_upload_fn=unused,
        finalize_active_workspace_upload_fn=unused,
        create_active_workspace_snapshot_job_fn=unused,
        cancel_active_workspace_upload_fn=unused,
        get_job_fn=unused,
        get_result_fn=unused,
        get_projection_fn=unused,
        get_thumbnail_fn=unused,
        record_review_disposition_fn=unused,
        authorize_training_capture_request_fn=unused,
        record_training_action_fn=unused,
        training_action_status_fn=unused,
        export_training_actions_fn=unused,
        create_cluster_search_fn=unused,
        get_cluster_search_fn=unused,
        cancel_cluster_search_fn=unused,
        cancel_job_fn=unused,
        create_qwen_review_fn=unused,
        list_qwen_reviews_fn=unused,
        get_qwen_review_fn=unused,
        cancel_qwen_review_fn=unused,
        get_qwen_review_evidence_fn=unused,
        cache_status_fn=cache_status_fn,
        purge_cache_fn=purge_cache_fn,
    )


def test_cache_api_routes_forward_status_and_explicit_categories():
    calls = []
    app = FastAPI()
    app.include_router(
        _router(
            lambda: {"total_bytes": 17},
            lambda payload: calls.append(payload)
            or {"status": "cleared"},
        )
    )
    client = TestClient(app)

    assert client.get("/class_analysis/cache").json() == {
        "total_bytes": 17
    }
    response = client.post(
        "/class_analysis/cache/purge",
        json={
            "categories": [
                "image_packs",
                "resume_embeddings",
            ]
        },
    )
    assert response.json() == {"status": "cleared"}
    assert calls == [
        {
            "categories": [
                "image_packs",
                "resume_embeddings",
            ]
        }
    ]


def test_cache_ui_refresh_and_clear_report_budget_scope_and_success():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "ybat-master"
        / "class_analysis_cache_admin.js"
    )
    node_script = f"""
const fs = require("fs");
const listeners = {{}};
const elements = new Map();
function element(id) {{
  if (!elements.has(id)) {{
    elements.set(id, {{
      id,
      textContent: "",
      disabled: false,
      dataset: {{}},
      addEventListener(event, handler) {{ listeners[`${{id}}:${{event}}`] = handler; }},
    }});
  }}
  return elements.get(id);
}}
globalThis.document = {{
  readyState: "complete",
  getElementById: element,
}};
globalThis.location = {{ origin: "http://127.0.0.1:8000" }};
globalThis.getTatorApiRoot = () => "http://127.0.0.1:8000";
globalThis.confirm = () => true;
const initial = {{
  total_bytes: 17,
  managed_bytes: 13,
  purgeable_bytes: 13,
  protected_bytes: 4,
  max_bytes: 1024,
  over_budget_bytes: 0,
  active_users: [],
  categories: {{ image_packs: {{ bytes: 4 }}, resume_embeddings: {{ bytes: 9 }} }},
}};
const after = {{
  total_bytes: 4,
  managed_bytes: 0,
  purgeable_bytes: 0,
  protected_bytes: 4,
  max_bytes: 1024,
  over_budget_bytes: 0,
  active_users: [],
  categories: {{ image_packs: {{ bytes: 0 }}, resume_embeddings: {{ bytes: 0 }} }},
}};
let calls = 0;
globalThis.fetch = async (_url, options = {{}}) => {{
  calls += 1;
  const payload = options.method === "POST"
    ? {{ bytes_reclaimed: 13, after }}
    : initial;
  return {{ ok: true, status: 200, text: async () => JSON.stringify(payload) }};
}};
function waitFor(predicate) {{
  return new Promise((resolve, reject) => {{
    let attempts = 0;
    const tick = () => {{
      if (predicate()) return resolve();
      if (++attempts > 100) return reject(new Error("timeout"));
      setTimeout(tick, 1);
    }};
    tick();
  }});
}}
eval(fs.readFileSync({json.dumps(str(script_path))}, "utf8"));
(async () => {{
  await waitFor(() => calls === 1 && element("classAnalysisCacheStatus").textContent.includes("regenerable"));
  if (!element("classAnalysisCacheStatus").textContent.includes("4 B protected")) throw new Error("protected bytes missing");
  await listeners["classAnalysisCacheClear:click"]();
  if (calls !== 2) throw new Error("purge request missing");
  if (!element("classAnalysisCacheStatus").textContent.startsWith("Cleared 13 B.")) throw new Error("success message lost");
  if (!element("classAnalysisCacheStatus").textContent.includes("4 B total")) throw new Error("post-purge total missing");
  if (!element("classAnalysisCacheClear").disabled) throw new Error("empty cache clear must disable");
}})().catch((error) => {{ console.error(error); process.exitCode = 1; }});
"""
    completed = subprocess.run(
        ["node", "-e", node_script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_cache_ui_failure_message_is_not_replaced_by_stale_status():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "ybat-master"
        / "class_analysis_cache_admin.js"
    )
    node_script = f"""
const fs = require("fs");
const listeners = {{}};
const elements = new Map();
function element(id) {{
  if (!elements.has(id)) elements.set(id, {{
    textContent: "", disabled: false, dataset: {{}},
    addEventListener(event, handler) {{ listeners[`${{id}}:${{event}}`] = handler; }},
  }});
  return elements.get(id);
}}
globalThis.document = {{ readyState: "complete", getElementById: element }};
globalThis.location = {{ origin: "http://127.0.0.1:8000" }};
globalThis.getTatorApiRoot = () => "http://127.0.0.1:8000";
globalThis.confirm = () => true;
let calls = 0;
globalThis.fetch = async (_url, options = {{}}) => {{
  calls += 1;
  if (calls === 1) return {{
    ok: true, status: 200,
    text: async () => JSON.stringify({{
      total_bytes: 17, managed_bytes: 13, purgeable_bytes: 13,
      protected_bytes: 4, active_users: [], categories: {{}},
    }}),
  }};
  return {{
    ok: false, status: 503,
    text: async () => JSON.stringify({{ detail: {{ code: "cache_busy" }} }}),
  }};
}};
function waitFor(predicate) {{
  return new Promise((resolve, reject) => {{
    let attempts = 0;
    const tick = () => predicate() ? resolve() : ++attempts > 100
      ? reject(new Error("timeout")) : setTimeout(tick, 1);
    tick();
  }});
}}
eval(fs.readFileSync({json.dumps(str(script_path))}, "utf8"));
(async () => {{
  await waitFor(() => calls === 1 && element("classAnalysisCacheStatus").textContent.includes("regenerable"));
  await listeners["classAnalysisCacheClear:click"]();
  const status = element("classAnalysisCacheStatus");
  if (!status.textContent.includes("Cache clear blocked: cache_busy")) throw new Error(status.textContent);
  if (status.dataset.tone !== "error") throw new Error("error tone missing");
}})().catch((error) => {{ console.error(error); process.exitCode = 1; }});
"""
    completed = subprocess.run(
        ["node", "-e", node_script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
