#!/usr/bin/env python3
"""Summarize prepass stage/source attribution from cached prepass records."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _iter_records(images_dir: Path) -> Iterable[Dict[str, Any]]:
    for path in sorted(images_dir.glob("*.json")):
        try:
            yield json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue


def _load_config(cache_dir: Path) -> Dict[str, Any]:
    meta = cache_dir / "prepass.meta.json"
    if not meta.exists():
        return {}
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(data.get("prepass_config") or {})


def _count_stage_atoms(stage_atoms: Any) -> Tuple[Counter, Counter]:
    counts = Counter()
    images_with = Counter()
    if not isinstance(stage_atoms, dict):
        return counts, images_with
    for stage, runs in stage_atoms.items():
        if not isinstance(runs, dict):
            continue
        for run, atom_ids in runs.items():
            key = f"{stage}.{run}"
            n = len(atom_ids or [])
            counts[key] += n
            if n > 0:
                images_with[key] += 1
    return counts, images_with


def summarize(cache_dir: Path) -> Dict[str, Any]:
    images_dir = cache_dir / "images"
    records = list(_iter_records(images_dir))
    config = _load_config(cache_dir)

    stage_counts = Counter()
    stage_images_with = Counter()
    final_primary = Counter()
    warning_counts = Counter()
    total_dets = 0
    per_image_counts: List[int] = []
    provenance_missing = 0

    for record in records:
        detections = list(record.get("detections") or [])
        warnings = list(record.get("warnings") or [])
        prov = record.get("provenance")

        per_image_counts.append(len(detections))
        total_dets += len(detections)
        for det in detections:
            src = str(det.get("source") or det.get("score_source") or "unknown").strip().lower()
            final_primary[src] += 1
        for warning in warnings:
            key = str(warning).split(":", 1)[0]
            warning_counts[key] += 1

        if not isinstance(prov, dict):
            provenance_missing += 1
            continue
        counts, images_with = _count_stage_atoms(prov.get("stage_atoms"))
        stage_counts.update(counts)
        stage_images_with.update(images_with)

    images_total = len(records)
    avg_dets = (float(total_dets) / float(images_total)) if images_total else 0.0
    expected_window_similarity = bool(config.get("similarity_window_extension", False))
    expected_window_text = bool(config.get("sam3_text_window_extension", False))
    observed_similarity_windowed = int(stage_counts.get("sam3_similarity.windowed", 0))
    observed_text_windowed = int(stage_counts.get("sam3_text.windowed", 0))

    findings: List[str] = []
    if expected_window_similarity and observed_similarity_windowed == 0:
        findings.append(
            "similarity_window_extension=true but observed sam3_similarity.windowed atoms=0"
        )
    if expected_window_text and observed_text_windowed == 0:
        findings.append("sam3_text_window_extension=true but observed sam3_text.windowed atoms=0")
    if provenance_missing > 0:
        findings.append(f"provenance missing in {provenance_missing} image records")

    return {
        "cache_dir": str(cache_dir),
        "config": {
            "sam3_text_window_extension": config.get("sam3_text_window_extension"),
            "sam3_text_window_mode": config.get("sam3_text_window_mode"),
            "similarity_window_extension": config.get("similarity_window_extension"),
            "similarity_window_mode": config.get("similarity_window_mode"),
            "similarity_exemplar_strategy": config.get("similarity_exemplar_strategy"),
            "similarity_exemplar_count": config.get("similarity_exemplar_count"),
            "dedupe_iou": config.get("dedupe_iou"),
            "fusion_mode": config.get("fusion_mode"),
        },
        "images_total": images_total,
        "provenance_missing_images": provenance_missing,
        "final_detection_counts": {
            "total": total_dets,
            "avg_per_image": avg_dets,
            "primary_source": dict(final_primary.most_common()),
        },
        "stage_atom_counts": dict(stage_counts),
        "stage_images_with_atoms": dict(stage_images_with),
        "warnings": dict(warning_counts),
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepass-cache-dir",
        action="append",
        required=True,
        help="Path to uploads/calibration_cache/prepass/<key> (repeatable)",
    )
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    reports = []
    for raw_dir in args.prepass_cache_dir:
        cache_dir = Path(raw_dir).resolve()
        reports.append(summarize(cache_dir))

    out = {"reports": reports}
    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

