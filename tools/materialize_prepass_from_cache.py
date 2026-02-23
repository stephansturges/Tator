#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def _load_image_filter(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"image filter file not found: {p}")
    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        return set()
    try:
        payload = json.loads(raw)
        if isinstance(payload, list):
            return {str(v).strip() for v in payload if str(v).strip()}
    except json.JSONDecodeError:
        pass
    return {line.strip() for line in raw.splitlines() if line.strip()}


def _normalized_sources(raw: Any) -> List[str]:
    out: List[str] = []
    if isinstance(raw, (list, tuple)):
        for src in raw:
            name = str(src or "").strip().lower()
            if name and name not in out:
                out.append(name)
    return out


def _filter_detection(det: Dict[str, Any], dropped: Set[str]) -> Optional[Dict[str, Any]]:
    source = str(det.get("source") or det.get("score_source") or "unknown").strip().lower() or "unknown"
    source_list = _normalized_sources(det.get("source_list"))
    if source not in source_list:
        source_list.append(source)

    score_by_source: Dict[str, float] = {}
    raw_score_map = det.get("score_by_source")
    if isinstance(raw_score_map, dict):
        for raw_src, raw_score in raw_score_map.items():
            src_name = str(raw_src or "").strip().lower()
            if not src_name:
                continue
            try:
                score_val = float(raw_score)
            except (TypeError, ValueError):
                continue
            prev = score_by_source.get(src_name)
            if prev is None or score_val > prev:
                score_by_source[src_name] = score_val
    try:
        primary_score = float(det.get("score") or 0.0)
    except (TypeError, ValueError):
        primary_score = 0.0
    if source and (source not in score_by_source or primary_score > score_by_source[source]):
        score_by_source[source] = primary_score

    # Apply drop filter.
    source_list = [src for src in source_list if src not in dropped]
    score_by_source = {src: score for src, score in score_by_source.items() if src not in dropped}

    if not source_list and not score_by_source:
        return None
    if source in dropped or source not in source_list:
        # Reassign primary to strongest remaining source.
        if score_by_source:
            source = max(score_by_source.items(), key=lambda kv: kv[1])[0]
        elif source_list:
            source = source_list[0]
        else:
            return None

    out = dict(det)
    out["source"] = source
    out["score_source"] = source
    out["source_primary"] = source
    out["source_list"] = sorted(set(source_list + [source]))
    if score_by_source:
        out["score_by_source"] = score_by_source
        out["score"] = float(score_by_source.get(source, primary_score))
    else:
        out["score_by_source"] = {source: float(primary_score)}
        out["score"] = float(primary_score)
    return out


def _filter_provenance(provenance: Any, dropped: Set[str]) -> Any:
    if not isinstance(provenance, dict) or not dropped:
        return provenance
    out = dict(provenance)
    atoms = provenance.get("atoms")
    if isinstance(atoms, list):
        kept_atoms = []
        for atom in atoms:
            if not isinstance(atom, dict):
                continue
            source = str(
                atom.get("source_primary") or atom.get("source") or atom.get("score_source") or ""
            ).strip().lower()
            if source in dropped:
                continue
            kept_atoms.append(atom)
        out["atoms"] = kept_atoms
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize prepass JSONL from calibration cache records.")
    parser.add_argument("--cache-key", required=True, help="Prepass cache key under uploads/calibration_cache/prepass.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--images-file",
        default=None,
        help="Optional JSON list or newline list of images to include.",
    )
    parser.add_argument(
        "--drop-sources",
        default="",
        help="Comma-separated sources to drop (e.g. sam3_similarity).",
    )
    args = parser.parse_args()

    cache_dir = Path("uploads/calibration_cache/prepass") / str(args.cache_key).strip() / "images"
    if not cache_dir.exists():
        raise SystemExit(f"cache directory not found: {cache_dir}")

    image_filter = _load_image_filter(args.images_file)
    dropped = {src.strip().lower() for src in str(args.drop_sources).split(",") if src.strip()}
    paths = sorted(cache_dir.glob("*.json"))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for path in paths:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            image = str(record.get("image") or "").strip()
            if not image:
                continue
            if image_filter is not None and image not in image_filter:
                continue
            dets_out: List[Dict[str, Any]] = []
            for det in record.get("detections") or []:
                if not isinstance(det, dict):
                    continue
                filtered = _filter_detection(det, dropped)
                if filtered is not None:
                    dets_out.append(filtered)
            out_record = {
                "image": image,
                "dataset_id": record.get("dataset_id"),
                "detections": dets_out,
                "warnings": list(record.get("warnings") or []),
                "provenance": _filter_provenance(record.get("provenance"), dropped),
            }
            handle.write(json.dumps(out_record, ensure_ascii=True) + "\n")
            written += 1
    print(json.dumps({"output": str(out_path), "records": written, "dropped_sources": sorted(dropped)}, indent=2))


if __name__ == "__main__":
    main()
