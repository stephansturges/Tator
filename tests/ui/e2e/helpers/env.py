import os
import socket
from urllib.parse import urlparse


def env(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default) or "").strip()


def require_ui_env() -> tuple[str, str]:
    page_url = env("UI_PAGE_URL")
    dataset_path = env("UI_DATASET_PATH")
    if not page_url:
        raise RuntimeError("Set UI_PAGE_URL to the hosted ybat UI URL.")
    if not dataset_path:
        raise RuntimeError("Set UI_DATASET_PATH to a valid server-local dataset path.")
    return page_url, dataset_path


def api_root() -> str:
    return env("UI_API_ROOT", "http://127.0.0.1:8000").rstrip("/")


def backend_reachable() -> bool:
    health_url = env("UI_HEALTH_URL", "http://127.0.0.1:8000/system/health_summary")
    parsed = urlparse(health_url)
    host = parsed.hostname or "127.0.0.1"
    if parsed.port:
        port = int(parsed.port)
    elif parsed.scheme == "https":
        port = 443
    else:
        port = 80
    try:
        with socket.create_connection((host, port), timeout=5):
            return True
    except OSError:
        return False
