import hashlib
import os
import shutil
import socket
from pathlib import Path
from urllib.parse import urlparse


def env(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default) or "").strip()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _stage_ui_dataset_if_needed(dataset_path: str) -> str:
    raw = str(dataset_path or "").strip()
    if not raw:
        return raw
    source = Path(raw).expanduser().resolve()
    if not source.exists():
        return raw
    if env("UI_DATASET_STAGE", "1") == "0":
        return str(source)
    staging_root = Path(
        env("UI_DATASET_STAGING_ROOT", str((_repo_root() / "uploads" / "datasets" / "_ui_e2e").resolve()))
    ).expanduser().resolve()
    staging_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:8]
    staged = staging_root / f"{source.name}_{digest}"
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(source, staged)

    # Some lightweight fixtures only ship images + labelmap. Stage empty YOLO label files
    # so dataset-linked annotation flows can open them like a normal linked dataset.
    flat_images = staged / "images"
    flat_labels = staged / "labels"
    if flat_images.exists() and flat_images.is_dir() and not flat_labels.exists():
        flat_labels.mkdir(parents=True, exist_ok=True)
        for image_path in flat_images.iterdir():
            if not image_path.is_file():
                continue
            (flat_labels / f"{image_path.stem}.txt").write_text("", encoding="utf-8")
    for split in ("train", "val"):
        split_images = staged / split / "images"
        split_labels = staged / split / "labels"
        if split_images.exists() and split_images.is_dir() and not split_labels.exists():
            split_labels.mkdir(parents=True, exist_ok=True)
            for image_path in split_images.rglob("*"):
                if not image_path.is_file():
                    continue
                rel = image_path.relative_to(split_images)
                target = split_labels / rel.with_suffix(".txt")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("", encoding="utf-8")
    return str(staged)


def require_ui_env() -> tuple[str, str]:
    page_url = env("UI_PAGE_URL", f"{api_root()}/tator.html")
    dataset_path = env("UI_DATASET_PATH", str((_repo_root() / "tests" / "fixtures" / "fuzz_pack").resolve()))
    if not page_url:
        raise RuntimeError("Set UI_PAGE_URL to the hosted Tator UI URL.")
    if not dataset_path:
        raise RuntimeError("Set UI_DATASET_PATH to a valid server-local dataset path.")
    return page_url, _stage_ui_dataset_if_needed(dataset_path)


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
