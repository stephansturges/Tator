"""Run a repeatable automatic-labeling benchmark matrix against a live backend."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_RELATIVE_THRESHOLDS = {
    "throughput_drop_pct": 0.15,
    "avg_latency_increase_pct": 0.20,
    "p95_latency_increase_pct": 0.20,
    "planner_avg_increase_pct": 0.20,
    "falcon_query_increase_pct": 0.10,
}


@dataclass(frozen=True)
class AutoLabelBenchmarkCase:
    name: str
    baseline_mode: str
    falcon_window_mode: str
    use_planner_caption: bool
    enable_yolo: bool
    enable_rfdetr: bool
    use_edr_package: bool


def build_benchmark_cases(profile: str) -> List[AutoLabelBenchmarkCase]:
    profile_norm = str(profile or "").strip().lower()
    if profile_norm == "weekly":
        return [
            AutoLabelBenchmarkCase("edr_full_image", "edr", "full_image", False, True, True, True),
            AutoLabelBenchmarkCase("edr_quadrants", "edr", "quadrants", False, True, True, True),
            AutoLabelBenchmarkCase("edr_planner_no_caption", "edr", "planner_auto", False, True, True, True),
            AutoLabelBenchmarkCase("edr_planner_with_caption", "edr", "planner_auto", True, True, True, True),
        ]
    if profile_norm != "nightly":
        raise ValueError(f"unsupported_profile:{profile}")
    return [
        AutoLabelBenchmarkCase("edr_full_image", "edr", "full_image", False, True, True, True),
        AutoLabelBenchmarkCase("edr_quadrants", "edr", "quadrants", False, True, True, True),
        AutoLabelBenchmarkCase("edr_planner_no_caption", "edr", "planner_auto", False, True, True, True),
        AutoLabelBenchmarkCase("edr_planner_with_caption", "edr", "planner_auto", True, True, True, True),
        AutoLabelBenchmarkCase("yolo_full_image", "yolo", "full_image", False, True, False, False),
        AutoLabelBenchmarkCase("yolo_quadrants", "yolo", "quadrants", False, True, False, False),
        AutoLabelBenchmarkCase("rfdetr_full_image", "rfdetr", "full_image", False, False, True, False),
        AutoLabelBenchmarkCase("rfdetr_quadrants", "rfdetr", "quadrants", False, False, True, False),
        AutoLabelBenchmarkCase("union_full_image", "union", "full_image", False, True, True, False),
        AutoLabelBenchmarkCase("union_quadrants", "union", "quadrants", False, True, True, False),
        AutoLabelBenchmarkCase("union_planner_no_caption", "union", "planner_auto", False, True, True, False),
        AutoLabelBenchmarkCase("union_planner_with_caption", "union", "planner_auto", True, True, True, False),
    ]


def load_sample_manifest(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        image_relpaths = [str(item).strip().replace("\\", "/") for item in payload if str(item).strip()]
        return {"image_relpaths": image_relpaths}
    if not isinstance(payload, dict):
        raise ValueError("sample_manifest_invalid")
    if isinstance(payload.get("image_relpaths"), list):
        image_relpaths = [
            str(item).strip().replace("\\", "/")
            for item in payload.get("image_relpaths") or []
            if str(item).strip()
        ]
    elif isinstance(payload.get("images"), list):
        image_relpaths = [
            str(row.get("image_relpath") or row.get("image_name") or "").strip().replace("\\", "/")
            for row in payload.get("images") or []
            if isinstance(row, dict)
            and str(row.get("image_relpath") or row.get("image_name") or "").strip()
        ]
    else:
        raise ValueError("sample_manifest_missing_images")
    return {
        "image_relpaths": image_relpaths,
        "dataset_id": str(payload.get("dataset_id") or "").strip() or None,
        "target_mode": str(payload.get("target_mode") or "").strip() or None,
        "split": str(payload.get("split") or "").strip() or None,
        "class_names": payload.get("class_names") if isinstance(payload.get("class_names"), list) else None,
    }


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    rank = max(0.0, min(1.0, float(q))) * float(len(values) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(sorted(values)[low])
    ordered = sorted(float(v) for v in values)
    weight = rank - float(low)
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def summarize_case_result(case: AutoLabelBenchmarkCase, job: Dict[str, Any], elapsed_sec: float) -> Dict[str, Any]:
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    images_processed = int(result.get("images_processed") or 0)
    image_times = [float(v) for v in (result.get("image_times_sec") or []) if isinstance(v, (int, float))]
    timings = result.get("timings_sec") if isinstance(result.get("timings_sec"), dict) else {}
    avg_latency = float(statistics.mean(image_times)) if image_times else 0.0
    throughput = float(images_processed) / float(elapsed_sec) if elapsed_sec > 0 and images_processed > 0 else 0.0
    planner_avg = (
        float(timings.get("planner") or 0.0) / float(images_processed)
        if images_processed > 0
        else 0.0
    )
    falcon_queries = int(result.get("falcon_query_count") or 0)
    return {
        "case": asdict(case),
        "dataset_id": result.get("dataset_id") or (job.get("request") or {}).get("dataset_id"),
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "elapsed_sec": float(elapsed_sec),
        "images_processed": images_processed,
        "images_total": int(result.get("images_total") or 0),
        "throughput_img_per_sec": throughput,
        "avg_latency_sec": avg_latency,
        "p50_latency_sec": percentile(image_times, 0.50),
        "p95_latency_sec": percentile(image_times, 0.95),
        "planner_avg_sec_per_image": planner_avg,
        "falcon_queries_per_image": (
            float(falcon_queries) / float(images_processed) if images_processed > 0 else 0.0
        ),
        "labels_added": int(result.get("labels_added") or 0),
        "duplicates_dropped": int(result.get("duplicates_dropped") or 0),
        "zero_write_images": int(result.get("zero_write_images") or 0),
        "writes_applied": int(result.get("writes_applied") or 0),
        "timings_sec": timings,
    }


def load_optional_json(path: Optional[str]) -> Dict[str, Any]:
    raw = str(path or "").strip()
    if not raw:
        return {}
    payload = json.loads(Path(raw).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_object_required:{raw}")
    return payload


def compare_to_baseline(
    summary: Dict[str, Any],
    baseline: Dict[str, Any],
    *,
    thresholds: Optional[Dict[str, float]] = None,
) -> List[str]:
    rules = dict(DEFAULT_RELATIVE_THRESHOLDS)
    if isinstance(thresholds, dict):
        rules.update({k: float(v) for k, v in thresholds.items() if isinstance(v, (int, float))})
    failures: List[str] = []

    def _ratio(current_key: str, baseline_key: str) -> Optional[float]:
        current = float(summary.get(current_key) or 0.0)
        previous = float(baseline.get(baseline_key) or 0.0)
        if previous <= 0.0:
            return None
        return (current - previous) / previous

    throughput_delta = _ratio("throughput_img_per_sec", "throughput_img_per_sec")
    if throughput_delta is not None and throughput_delta < -float(rules["throughput_drop_pct"]):
        failures.append(f"throughput_regressed:{throughput_delta:.4f}")

    avg_delta = _ratio("avg_latency_sec", "avg_latency_sec")
    if avg_delta is not None and avg_delta > float(rules["avg_latency_increase_pct"]):
        failures.append(f"avg_latency_regressed:{avg_delta:.4f}")

    p95_delta = _ratio("p95_latency_sec", "p95_latency_sec")
    if p95_delta is not None and p95_delta > float(rules["p95_latency_increase_pct"]):
        failures.append(f"p95_latency_regressed:{p95_delta:.4f}")

    planner_delta = _ratio("planner_avg_sec_per_image", "planner_avg_sec_per_image")
    if planner_delta is not None and planner_delta > float(rules["planner_avg_increase_pct"]):
        failures.append(f"planner_latency_regressed:{planner_delta:.4f}")

    falcon_delta = _ratio("falcon_queries_per_image", "falcon_queries_per_image")
    if falcon_delta is not None and falcon_delta > float(rules["falcon_query_increase_pct"]):
        failures.append(f"falcon_query_regressed:{falcon_delta:.4f}")

    if str(summary.get("status") or "").strip().lower() != "completed":
        failures.append(f"job_not_completed:{summary.get('status')}")
    return failures


def _http_json(url: str, *, method: str = "GET", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_case_payload(
    *,
    dataset_id: str,
    image_relpaths: Sequence[str],
    case: AutoLabelBenchmarkCase,
    edr_package_id: Optional[str],
    target_mode: str,
    split: str,
    class_names: Optional[Sequence[str]],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "dataset_id": dataset_id,
        "image_relpaths": list(image_relpaths),
        "max_images": len(list(image_relpaths)),
        "split": split or "all",
        "unlabeled_only": False,
        "target_mode": target_mode or "auto",
        "falcon_window_mode": case.falcon_window_mode,
        "use_planner_caption": bool(case.use_planner_caption),
        "class_names": list(class_names) if class_names else None,
        "edr_package_id": edr_package_id if case.use_edr_package else None,
        "enable_yolo": bool(case.enable_yolo),
        "enable_rfdetr": bool(case.enable_rfdetr),
    }
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            payload[key] = value
    return payload


def wait_for_job(api_root: str, job_id: str, *, poll_secs: float, timeout_secs: float) -> Dict[str, Any]:
    deadline = time.time() + float(timeout_secs)
    while True:
        job = _http_json(f"{api_root.rstrip('/')}/auto_label/jobs/{job_id}")
        if str(job.get("status") or "").strip().lower() in {"completed", "failed", "cancelled"}:
            return job
        if time.time() >= deadline:
            raise TimeoutError(f"auto_label_benchmark_timeout:{job_id}")
        time.sleep(float(poll_secs))


def run_case(
    *,
    api_root: str,
    payload: Dict[str, Any],
    case: AutoLabelBenchmarkCase,
    poll_secs: float,
    timeout_secs: float,
) -> Dict[str, Any]:
    started = time.perf_counter()
    job = _http_json(f"{api_root.rstrip('/')}/auto_label/jobs", method="POST", payload=payload)
    finished = wait_for_job(api_root, str(job.get("job_id") or "").strip(), poll_secs=poll_secs, timeout_secs=timeout_secs)
    elapsed = time.perf_counter() - started
    return summarize_case_result(case, finished, elapsed)


def write_benchmark_output(
    output_path: Path,
    *,
    profile: str,
    dataset_id: str,
    image_count: int,
    results: Sequence[Dict[str, Any]],
    baseline_path: Optional[str],
    failures: Dict[str, List[str]],
) -> None:
    payload = {
        "profile": profile,
        "dataset_id": dataset_id,
        "image_count": image_count,
        "results": list(results),
        "baseline_path": baseline_path,
        "failures": failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def acquire_benchmark_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": os.getpid(), "created_at": time.time()}
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        try:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        pid = int(existing.get("pid") or 0)
        if pid > 0:
            try:
                os.kill(pid, 0)
            except OSError:
                lock_path.unlink(missing_ok=True)
                return acquire_benchmark_lock(lock_path)
        raise RuntimeError(f"benchmark_lock_active:{lock_path}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        lock_path.unlink(missing_ok=True)
        raise


def release_benchmark_lock(lock_path: Path) -> None:
    lock_path.unlink(missing_ok=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the automatic-label benchmark matrix.")
    parser.add_argument("--api-root", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--sample-manifest", required=True)
    parser.add_argument("--profile", choices=["nightly", "weekly"], default="nightly")
    parser.add_argument("--edr-package-id", default=None)
    parser.add_argument("--baseline-json", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dataset-id-map-json", default=None)
    parser.add_argument("--payload-overrides-json", default=None)
    parser.add_argument("--payload-overrides-by-case-json", default=None)
    parser.add_argument("--poll-secs", type=float, default=3.0)
    parser.add_argument("--timeout-secs", type=float, default=7200.0)
    args = parser.parse_args(argv)

    manifest = load_sample_manifest(Path(args.sample_manifest))
    dataset_id = str(manifest.get("dataset_id") or args.dataset_id).strip()
    image_relpaths = list(manifest.get("image_relpaths") or [])
    if not dataset_id or not image_relpaths:
        raise SystemExit("auto_label_benchmark_manifest_invalid")

    dataset_id_map_raw = load_optional_json(args.dataset_id_map_json)
    dataset_id_map = {str(key): str(value) for key, value in dataset_id_map_raw.items() if str(value or "").strip()}
    payload_overrides = load_optional_json(args.payload_overrides_json)
    payload_overrides_by_case_raw = load_optional_json(args.payload_overrides_by_case_json)
    payload_overrides_by_case = {
        str(key): value
        for key, value in payload_overrides_by_case_raw.items()
        if isinstance(value, dict)
    }

    cases = build_benchmark_cases(args.profile)
    results = []
    out_path = Path(args.output_json)
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    acquire_benchmark_lock(lock_path)
    try:
        for case in cases:
            case_dataset_id = dataset_id_map.get(case.name, dataset_id)
            case_overrides = dict(payload_overrides)
            case_overrides.update(payload_overrides_by_case.get(case.name) or {})
            payload = build_case_payload(
                dataset_id=case_dataset_id,
                image_relpaths=image_relpaths,
                case=case,
                edr_package_id=args.edr_package_id,
                target_mode=str(manifest.get("target_mode") or "auto"),
                split=str(manifest.get("split") or "all"),
                class_names=manifest.get("class_names"),
                overrides=case_overrides,
            )
            print(
                f"[auto-label-benchmark] starting case={case.name} dataset_id={case_dataset_id} images={len(image_relpaths)}",
                file=sys.stdout,
                flush=True,
            )
            summary = run_case(
                api_root=args.api_root,
                payload=payload,
                case=case,
                poll_secs=float(args.poll_secs),
                timeout_secs=float(args.timeout_secs),
            )
            results.append(summary)
            write_benchmark_output(
                out_path,
                profile=args.profile,
                dataset_id=dataset_id,
                image_count=len(image_relpaths),
                results=results,
                baseline_path=args.baseline_json,
                failures={},
            )
            print(
                "[auto-label-benchmark] finished "
                f"case={case.name} status={summary.get('status')} "
                f"images_processed={summary.get('images_processed')} "
                f"labels_added={summary.get('labels_added')} "
                f"throughput={float(summary.get('throughput_img_per_sec') or 0.0):.4f}",
                file=sys.stdout,
                flush=True,
            )

        baseline_payload = {}
        failures: Dict[str, List[str]] = {}
        if args.baseline_json:
            baseline_payload = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
            baseline_by_name = {
                str(item.get("case", {}).get("name") or ""): item
                for item in baseline_payload.get("results") or []
                if isinstance(item, dict)
            }
            for result in results:
                case_name = str(result.get("case", {}).get("name") or "")
                baseline = baseline_by_name.get(case_name)
                if baseline:
                    failures[case_name] = compare_to_baseline(result, baseline)

        write_benchmark_output(
            out_path,
            profile=args.profile,
            dataset_id=dataset_id,
            image_count=len(image_relpaths),
            results=results,
            baseline_path=args.baseline_json,
            failures=failures,
        )
        if any(failures.values()):
            return 2
        return 0
    finally:
        release_benchmark_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
