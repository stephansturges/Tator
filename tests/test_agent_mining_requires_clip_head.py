import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import HTTPException  # noqa: E402

from localinferenceapi import AgentMiningRequest, _start_agent_mining_job  # noqa: E402


def test_agent_mining_start_requires_clip_head():
    payload = AgentMiningRequest(dataset_id="ds")
    try:
        _start_agent_mining_job(payload)
    except HTTPException as exc:
        assert exc.status_code == 400
        assert exc.detail == "agent_mining_clip_head_required"
        return
    raise AssertionError("Expected agent mining to require a pretrained CLIP head.")

