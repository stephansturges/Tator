from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    api = sys.modules.get("localinferenceapi")
    if api is None:
        return
    for manager_name in ("sam_preload_manager", "predictor_manager"):
        manager = getattr(api, manager_name, None)
        stop = getattr(manager, "stop", None)
        if callable(stop):
            stop()
