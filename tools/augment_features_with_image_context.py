#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import localinferenceapi as api
from tools import build_ensemble_features as bef


def _load_npz(path: Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _feature_names(array: np.ndarray) -> List[str]:
    return [str(x) for x in array.tolist()]


def _extract_image_from_meta(meta_row: Any) -> str:
    try:
        obj = json.loads(str(meta_row))
    except Exception:
        return ""
    return str(obj.get("image") or "").strip()


def _find_img_embed_block(names: Sequence[str]) -> Tuple[int, int]:
    start = -1
    for idx, name in enumerate(names):
        if name.startswith("img_clf_emb_rp::"):
            start = idx
            break
    if start < 0:
        return -1, -1
    end = start
    while end < len(names) and names[end].startswith("img_clf_emb_rp::"):
        end += 1
    return start, end


def _find_insert_index(names: Sequence[str]) -> int:
    for idx, name in enumerate(names):
        if name.startswith("cand_label::"):
            return idx
    return len(names)


def _build_donor_embedding_map(
    donor_npz: Dict[str, Any],
    *,
    image_embed_dim: Optional[int],
) -> Tuple[Dict[str, np.ndarray], int]:
    names = _feature_names(donor_npz["feature_names"])
    start, end = _find_img_embed_block(names)
    if start < 0 or end <= start:
        raise SystemExit("donor_missing_img_clf_emb_block")
    dim = int(end - start)
    if image_embed_dim is not None and int(image_embed_dim) != dim:
        raise SystemExit(f"donor_img_embed_dim_mismatch: expected={image_embed_dim} got={dim}")
    X = np.asarray(donor_npz["X"], dtype=np.float32)
    meta = donor_npz["meta"]
    if X.shape[0] != len(meta):
        raise SystemExit("donor_shape_meta_mismatch")
    by_image: Dict[str, np.ndarray] = {}
    for row_idx, meta_row in enumerate(meta):
        image_name = _extract_image_from_meta(meta_row)
        if not image_name:
            continue
        vec = np.asarray(X[row_idx, start:end], dtype=np.float32).reshape(-1)
        prev = by_image.get(image_name)
        if prev is None:
            by_image[image_name] = vec
            continue
        if prev.shape != vec.shape:
            raise SystemExit(f"donor_inconsistent_embedding_shape:{image_name}")
        if not np.allclose(prev, vec, atol=1e-7):
            raise SystemExit(f"donor_inconsistent_embedding_values:{image_name}")
    return by_image, dim


def _resolve_image_path(dataset_id: str, image_name: str) -> Path:
    dataset_root = Path("uploads/qwen_runs/datasets") / dataset_id
    for split in ("val", "train"):
        candidate = dataset_root / split / image_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing_image:{image_name}")


def _activate_classifier(dataset_id: str, classifier_id: str) -> Dict[str, Any]:
    class _ClassifierResolveError(Exception):
        def __init__(self, status_code, detail):
            super().__init__(f"{status_code}:{detail}")
            self.status_code = status_code
            self.detail = detail

    classifier_path = api._resolve_agent_clip_classifier_path_impl(
        classifier_id,
        allowed_root=(api.UPLOAD_ROOT / "classifiers").resolve(),
        allowed_exts=api.CLASSIFIER_ALLOWED_EXTS,
        path_is_within_root_fn=api._path_is_within_root_impl,
        http_exception_cls=_ClassifierResolveError,
    )
    bef._activate_classifier_runtime(classifier_path, dataset_id)
    return api._load_clip_head_from_classifier(classifier_path)


def _encode_missing_image_embeddings(
    missing_images: Sequence[str],
    *,
    dataset_id: str,
    classifier_head: Dict[str, Any],
    image_embed_dim: int,
    image_embed_seed: int,
    embed_l2_normalize: bool,
    min_crop_size: int,
    device: Optional[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for image_name in missing_images:
        path = _resolve_image_path(dataset_id, image_name)
        with Image.open(path) as img:
            pil_img = img.convert("RGB")
        _probs, emb_rows = bef._encode_classifier_features(
            [pil_img],
            head=classifier_head,
            batch_size=1,
            device_override=device,
            min_crop_size=max(1, int(min_crop_size)),
            embed_proj_dim=int(image_embed_dim),
            embed_proj_seed=int(image_embed_seed),
            embed_l2_normalize=bool(embed_l2_normalize),
        )
        if len(emb_rows) != 1:
            raise SystemExit(f"missing_image_embed_failed:{image_name}")
        vec = np.asarray(emb_rows[0], dtype=np.float32).reshape(-1)
        if vec.shape[0] != int(image_embed_dim):
            raise SystemExit(f"missing_image_embed_dim_mismatch:{image_name}")
        out[image_name] = vec
    return out


def _insert_img_embed_block(
    *,
    base_X: np.ndarray,
    base_feature_names: List[str],
    image_by_row: Sequence[str],
    image_to_embed: Dict[str, np.ndarray],
    image_embed_dim: int,
) -> Tuple[np.ndarray, List[str]]:
    start, end = _find_img_embed_block(base_feature_names)
    X = base_X
    names = list(base_feature_names)
    if start >= 0 and end > start:
        X = np.concatenate([X[:, :start], X[:, end:]], axis=1)
        del names[start:end]
    insert_at = _find_insert_index(names)
    img_mat = np.zeros((X.shape[0], int(image_embed_dim)), dtype=np.float32)
    for row_idx, image_name in enumerate(image_by_row):
        vec = image_to_embed.get(image_name)
        if vec is None:
            raise SystemExit(f"missing_image_embedding_for_row:{image_name}")
        img_mat[row_idx, :] = vec
    X_out = np.concatenate([X[:, :insert_at], img_mat, X[:, insert_at:]], axis=1)
    embed_names = [f"img_clf_emb_rp::{idx:03d}" for idx in range(int(image_embed_dim))]
    names_out = names[:insert_at] + embed_names + names[insert_at:]
    return X_out, names_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject/refresh full-image embedding block in existing ensemble feature matrices."
    )
    parser.add_argument("--base-features", required=True, help="Existing feature .npz (candidate features).")
    parser.add_argument("--donor-features", required=True, help="Feature .npz containing img_clf_emb block to reuse.")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--dataset", required=True, help="Dataset id for missing-image fallback encoding.")
    parser.add_argument("--classifier-id", required=True, help="Classifier id/path for missing-image fallback.")
    parser.add_argument("--image-embed-dim", type=int, default=1024, help="Projected image embedding dim.")
    parser.add_argument("--image-embed-seed", type=int, default=4242, help="Image projection seed.")
    parser.add_argument("--device", default="cuda", help="Device for missing-image encoding.")
    parser.add_argument("--min-crop-size", type=int, default=4, help="Min image size for encoder entrypoint.")
    parser.add_argument(
        "--embed-no-l2-normalize",
        dest="embed_l2_normalize",
        action="store_false",
        help="Disable L2 normalization before projection.",
    )
    parser.set_defaults(embed_l2_normalize=True)
    parser.add_argument("--summary-json", default="", help="Optional summary json output path.")
    args = parser.parse_args()

    base_path = Path(args.base_features).resolve()
    donor_path = Path(args.donor_features).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_npz = _load_npz(base_path)
    donor_npz = _load_npz(donor_path)
    base_X = np.asarray(base_npz["X"], dtype=np.float32)
    base_names = _feature_names(base_npz["feature_names"])
    base_meta = base_npz["meta"]
    if base_X.shape[0] != len(base_meta):
        raise SystemExit("base_shape_meta_mismatch")

    image_to_embed, donor_dim = _build_donor_embedding_map(
        donor_npz, image_embed_dim=int(args.image_embed_dim)
    )
    if donor_dim != int(args.image_embed_dim):
        raise SystemExit(
            f"image_embed_dim_mismatch: requested={args.image_embed_dim} donor={donor_dim}"
        )

    image_by_row: List[str] = []
    unique_images: set[str] = set()
    for row in base_meta:
        image_name = _extract_image_from_meta(row)
        image_by_row.append(image_name)
        if image_name:
            unique_images.add(image_name)

    missing_images = sorted(
        image_name for image_name in unique_images if image_name and image_name not in image_to_embed
    )
    missing_encoded = 0
    if missing_images:
        classifier_head = _activate_classifier(args.dataset, args.classifier_id)
        encoded = _encode_missing_image_embeddings(
            missing_images,
            dataset_id=args.dataset,
            classifier_head=classifier_head,
            image_embed_dim=int(args.image_embed_dim),
            image_embed_seed=int(args.image_embed_seed),
            embed_l2_normalize=bool(args.embed_l2_normalize),
            min_crop_size=max(1, int(args.min_crop_size)),
            device=str(args.device or ""),
        )
        image_to_embed.update(encoded)
        missing_encoded = len(encoded)

    X_out, names_out = _insert_img_embed_block(
        base_X=base_X,
        base_feature_names=base_names,
        image_by_row=image_by_row,
        image_to_embed=image_to_embed,
        image_embed_dim=int(args.image_embed_dim),
    )

    base_npz["X"] = X_out
    base_npz["feature_names"] = np.asarray(names_out)
    base_npz["image_embed_proj_dim"] = np.asarray(int(args.image_embed_dim), dtype=np.int64)
    base_npz["image_embed_proj_seed"] = np.asarray(int(args.image_embed_seed), dtype=np.int64)
    np.savez_compressed(out_path, **base_npz)

    summary = {
        "status": "completed",
        "base_features": str(base_path),
        "donor_features": str(donor_path),
        "output": str(out_path),
        "rows": int(X_out.shape[0]),
        "cols": int(X_out.shape[1]),
        "image_embed_dim": int(args.image_embed_dim),
        "unique_images": int(len(unique_images)),
        "missing_images_before_fill": int(len(missing_images)),
        "missing_images_encoded": int(missing_encoded),
        "embed_l2_normalize": bool(args.embed_l2_normalize),
        "device": str(args.device or ""),
    }
    if args.summary_json:
        summary_path = Path(args.summary_json).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

