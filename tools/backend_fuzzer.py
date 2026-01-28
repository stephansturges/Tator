#!/usr/bin/env python3
"""
Lightweight backend fuzzer that can spin up uvicorn and hit a handful of API endpoints
with synthetic payloads. Designed to mirror the in-UI fuzzer for quick smoke tests.
"""
import argparse
import json
import base64
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, List, Tuple, Optional

from PIL import Image
import io
import random
import itertools


def _gen_random_image_base64(size: int = 96) -> str:
    """Create a small RGB PNG with random colored rectangles and return its base64 (no header)."""
    size = max(8, int(size))
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw_count = 20
    pixels = img.load()
    for _ in range(draw_count):
        color = tuple(int(c) for c in Image.new("RGB", (1, 1), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))).getpixel((0, 0)))
        x0 = random.randint(0, size - 8)
        y0 = random.randint(0, size - 8)
        w = random.randint(4, size // 3)
        h = random.randint(4, size // 3)
        for y in range(y0, min(size, y0 + h)):
            for x in range(x0, min(size, x0 + w)):
                pixels[x, y] = color
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _http_get(url: str, timeout: float = 5.0) -> Tuple[int | None, bytes | None, str | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read(), None
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), str(exc)
    except Exception as exc:  # noqa: BLE001
        return None, None, str(exc)


def _http_post_json(url: str, payload: dict, timeout: float = 8.0) -> Tuple[int | None, bytes | None, str | None]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), None
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), str(exc)
    except Exception as exc:  # noqa: BLE001
        return None, None, str(exc)


def _poll_job(url: str, timeout: float = 30.0, interval: float = 1.0) -> Tuple[bool, dict | None, str]:
    deadline = time.time() + timeout
    last_err = "timeout"
    while time.time() < deadline:
        status, body, err = _http_get(url, timeout=interval + 1)
        if status is None:
            last_err = err or f"HTTP {status}"
            time.sleep(interval)
            continue
        try:
            data = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            data = {}
        state = data.get("status")
        if state in {"completed", "failed", "cancelled"}:
            return state == "completed", data, str(state)
        last_err = f"state={state}"
        time.sleep(interval)
    return False, None, last_err


def wait_for_server(base_url: str, deadline: float = 30.0, proc: subprocess.Popen | None = None) -> Tuple[bool, str]:
    start = time.time()
    last_error = "no response yet"
    while time.time() - start < deadline:
        status, _, err = _http_get(f"{base_url}/sam_slots", timeout=3.0)
        if status is not None:
            return True, "ok"
        if err:
            last_error = err
        if proc is not None and proc.poll() is not None:
            return False, "server process exited"
        time.sleep(0.5)
    return False, last_error


def run_tests(
    base_url: str,
    include_qwen: bool,
    include_sam3: bool,
    include_auto: bool,
    include_clip: bool,
    include_agent_mining: bool,
    sam3_dataset_id: Optional[str],
    request_timeout: float = 8.0,
    max_sam3_tests: int = 20,
    max_qwen_tests: int = 8,
) -> List[Tuple[str, bool, str]]:
    results: List[Tuple[str, bool, str]] = []
    img_b64 = _gen_random_image_base64(96)
    image_name = "fuzz_image.png"
    def _name_with_suffix(base: str, suffix: str) -> str:
        return f"{base} [{suffix}]" if suffix else base

    tests: List[Tuple[str, Callable[[], Tuple[int | None, bytes | None, str | None]]]] = [
        ("sam_slots", lambda: _http_get(f"{base_url}/sam_slots", timeout=request_timeout)),
        (
            "sam_point (sam1)",
            lambda: _http_post_json(
                f"{base_url}/sam_point",
                {"point_x": 16, "point_y": 16, "image_base64": img_b64, "sam_variant": "sam1", "image_name": image_name},
                timeout=request_timeout,
            ),
        ),
        (
            "sam_point_multi (sam1)",
            lambda: _http_post_json(
                f"{base_url}/sam_point_multi",
                {"positive_points": [[8, 8], [24, 24]], "negative_points": [], "image_base64": img_b64, "sam_variant": "sam1", "image_name": image_name},
                timeout=request_timeout,
            ),
        ),
        (
            "sam_bbox (sam1)",
            lambda: _http_post_json(
                f"{base_url}/sam_bbox",
                {"bbox_left": 8, "bbox_top": 8, "bbox_width": 32, "bbox_height": 32, "image_base64": img_b64, "sam_variant": "sam1", "image_name": image_name},
                timeout=request_timeout,
            ),
        ),
    ]
    if include_clip:
        tests.append(("clip/backbones", lambda: _http_get(f"{base_url}/clip/backbones", timeout=request_timeout)))
        tests.append(("clip/active_model (GET)", lambda: _http_get(f"{base_url}/clip/active_model", timeout=request_timeout)))
        tests.append(
            (
                "clip/active_model (POST)",
                lambda: _http_post_json(
                    f"{base_url}/clip/active_model",
                    {},
                    timeout=request_timeout,
                ),
            )
        )
        tests.append(
            (
                "predict_base64",
                lambda: _http_post_json(
                    f"{base_url}/predict_base64",
                    {"image_base64": img_b64, "uuid": "fuzz"},
                    timeout=request_timeout,
                ),
            )
        )
    if include_qwen:
        qwen_items = ["car, person", "tree, road", "dog, cat"]
        prompt_types = ["bbox", "point", "bbox_sam"]
        max_results_opts = [1, 2, 5]
        qwen_tests = []
        for items in qwen_items:
            for ptype in prompt_types:
                for maxr in max_results_opts:
                    label = f"{ptype} items={items} max={maxr}"
                    payload = {"item_list": items, "prompt_type": ptype, "max_results": maxr}
                    qwen_tests.append((_name_with_suffix("qwen/infer", label), payload))
        qwen_tests.append((_name_with_suffix("qwen/infer", "manual prompt"), {"prompt": "detect object", "prompt_type": "bbox", "max_results": 3}))
        random.shuffle(qwen_tests)
        for label, extra in qwen_tests[:max_qwen_tests]:
            tests.append(
                (
                    label,
                    lambda extra=extra: _http_post_json(
                        f"{base_url}/qwen/infer",
                        {
                            **extra,
                            "image_base64": img_b64,
                            "image_name": image_name,
                        },
                        timeout=request_timeout,
                    ),
                )
        )
    if include_sam3:
        thresholds = [0.0, 0.2, 0.5, 0.8, 1.0]
        mask_thresholds = [0.0, 0.3, 0.5, 0.9]
        min_sizes = [0, 1, 10, 100]
        epsilons = [0.0, 0.1, 0.5, 2.0]
        max_results_opts = [1, 5, 10]
        combos = list(itertools.product(thresholds, mask_thresholds, min_sizes, epsilons, max_results_opts))
        random.shuffle(combos)
        for thr, mthr, msz, eps, maxr in combos[:max_sam3_tests]:
            label = f"thr={thr},mask_thr={mthr},min={msz},eps={eps},max={maxr}"
            tests.append(
                (
                    _name_with_suffix("sam3/text_prompt", label),
                    lambda thr=thr, mthr=mthr, msz=msz, eps=eps, maxr=maxr: _http_post_json(
                        f"{base_url}/sam3/text_prompt",
                        {
                            "text_prompt": "object",
                            "threshold": thr,
                            "mask_threshold": mthr,
                            "max_results": maxr,
                            "min_size": msz,
                            "simplify_epsilon": eps,
                            "image_base64": img_b64,
                            "sam_variant": "sam3",
                            "image_name": image_name,
                        },
                        timeout=request_timeout,
                    ),
                )
            )
        tests.append(
            (
                "sam3/text_prompt_auto",
                lambda: _http_post_json(
                    f"{base_url}/sam3/text_prompt_auto",
                    {
                        "text_prompt": "object",
                        "threshold": 0.4,
                        "mask_threshold": 0.3,
                        "max_results": 5,
                        "min_size": 0,
                        "simplify_epsilon": 0.5,
                        "image_base64": img_b64,
                        "sam_variant": "sam3",
                        "image_name": image_name,
                    },
                    timeout=request_timeout,
                ),
            )
        )
        # Visual similarity prompt using a random bbox
        tests.append(
            (
                "sam3/visual_prompt",
                lambda: _http_post_json(
                    f"{base_url}/sam3/visual_prompt",
                    {
                        "bbox_left": 8,
                        "bbox_top": 8,
                        "bbox_width": 32,
                        "bbox_height": 32,
                        "threshold": 0.4,
                        "mask_threshold": 0.5,
                        "max_results": 5,
                        "min_size": 0,
                        "simplify_epsilon": 0.5,
                        "image_base64": img_b64,
                        "sam_variant": "sam3",
                        "image_name": image_name,
                    },
                    timeout=request_timeout,
                ),
            )
        )
        if sam3_dataset_id:
            tests.append(
                (
                    "sam3/prompt_helper/jobs",
                    lambda: _http_post_json(
                        f"{base_url}/sam3/prompt_helper/jobs",
                        {
                            "dataset_id": sam3_dataset_id,
                            "target_class": "fuzz_cat",
                            "prompts": ["fuzz", "object"],
                            "max_images": 2,
                        },
                        timeout=request_timeout,
                    ),
                )
            )
            tests.append(
                (
                    "sam3/prompt_helper/search",
                    lambda: _http_post_json(
                        f"{base_url}/sam3/prompt_helper/search",
                        {
                            "dataset_id": sam3_dataset_id,
                            "target_class": "fuzz_cat",
                            "prompts_by_class": {1: ["fuzz", "object"]},
                            "max_prompts": 3,
                            "max_images": 3,
                            "score_threshold": 0.2,
                            "max_dets": 50,
                            "iou_threshold": 0.5,
                        },
                        timeout=request_timeout,
                    ),
                )
            )
            tests.append(
            (
                "sam3/prompt_helper/recipe",
                lambda: _http_post_json(
                    f"{base_url}/sam3/prompt_helper/recipe",
                    {
                        "dataset_id": sam3_dataset_id,
                        "target_class": "fuzz_cat",
                        "class_id": 1,
                        "prompts": [
                            {"prompt": "fuzz", "threshold": 0.2},
                            {"prompt": "object", "threshold": 0.2},
                        ],
                        "max_prompts": 3,
                        "max_images": 3,
                        "score_threshold": 0.2,
                        "max_dets": 50,
                        "iou_threshold": 0.5,
                        },
                        timeout=request_timeout,
                    ),
                )
            )
        if include_agent_mining and sam3_dataset_id:
            clip_classifier = None
            try:
                st_clf, body_clf, _ = _http_get(f"{base_url}/clip/classifiers", timeout=request_timeout)
                if st_clf and 200 <= st_clf < 300 and body_clf:
                    parsed_clf = json.loads(body_clf.decode("utf-8"))
                    if isinstance(parsed_clf, list) and parsed_clf:
                        first_entry = parsed_clf[0] or {}
                        clip_classifier = first_entry.get("rel_path") or first_entry.get("path")
            except Exception:
                clip_classifier = None
            if clip_classifier:
                def _start_agent_job():
                    # Try to restrict mining to a single class to keep test runs fast.
                    class_ids = None
                    try:
                        st_cls, body_cls, _ = _http_get(
                            f"{base_url}/sam3/datasets/{urllib.parse.quote(sam3_dataset_id)}/classes",
                            timeout=request_timeout,
                        )
                        if st_cls and 200 <= st_cls < 300 and body_cls:
                            parsed_cls = json.loads(body_cls.decode("utf-8"))
                            class_ids = parsed_cls.get("class_ids") or []
                    except Exception:
                        class_ids = None
                    first_class = None
                    if isinstance(class_ids, list) and class_ids:
                        try:
                            first_class = int(class_ids[0])
                        except Exception:
                            first_class = None
                    return _http_post_json(
                        f"{base_url}/agent_mining/jobs",
                        {
                            "dataset_id": sam3_dataset_id,
                            "eval_image_count": 5,
                            "split_seed": 42,
                            "reuse_cache": True,
                            "classes": [first_class] if first_class is not None else None,
                            # Keep prompt expansion off for smoke tests unless explicitly desired.
                            "prompt_llm_max_prompts": 0,
                            "steps_max_steps_per_recipe": 2,
                            "steps_max_visual_seeds_per_step": 2,
                            "mask_threshold": 0.5,
                            "max_results": 5,
                            "min_size": 0,
                            "simplify_epsilon": 0.5,
                            "clip_head_classifier_path": clip_classifier,
                        },
                        timeout=request_timeout,
                    )

                tests.append(
                    (
                        "agent_mining/jobs (sample)",
                        _start_agent_job,
                    )
                )
                tests.append(
                    (
                        "agent_mining/jobs (list)",
                        lambda: _http_get(f"{base_url}/agent_mining/jobs", timeout=request_timeout),
                    )
                )
            else:
                results.append(("agent_mining/jobs (skip)", True, "No CLIP classifier available"))
    if include_auto:
        tests.append(
            (
                "sam_point_auto (sam1)",
                lambda: _http_post_json(
                    f"{base_url}/sam_point_auto",
                    {"point_x": 16, "point_y": 16, "image_base64": img_b64, "sam_variant": "sam1", "image_name": image_name},
                    timeout=request_timeout,
                ),
            )
        )
        tests.append(
            (
                "sam_point_multi_auto (sam1)",
                lambda: _http_post_json(
                    f"{base_url}/sam_point_multi_auto",
                    {"positive_points": [[8, 8], [24, 24]], "negative_points": [], "image_base64": img_b64, "sam_variant": "sam1", "image_name": image_name},
                    timeout=request_timeout,
                ),
            )
        )
        tests.append(
            (
                "sam_bbox_auto (sam1)",
                lambda: _http_post_json(
                    f"{base_url}/sam_bbox_auto",
                    {"bbox_left": 8, "bbox_top": 8, "bbox_width": 32, "bbox_height": 32, "image_base64": img_b64, "sam_variant": "sam1", "image_name": image_name},
                    timeout=request_timeout,
                ),
            )
        )
        tests.append(
            (
                "qwen/prepass (minimal)",
                lambda: _http_post_json(
                    f"{base_url}/qwen/prepass",
                    {
                        "image_base64": img_b64,
                        "image_name": image_name,
                        "dataset_id": sam3_dataset_id,
                        "prepass_only": True,
                        "prepass_caption": False,
                        "enable_yolo": False,
                        "enable_rfdetr": False,
                        "enable_sam3_text": False,
                        "enable_sam3_similarity": False,
                        "prepass_keep_all": True,
                    },
                    timeout=request_timeout,
                ),
            )
        )
    for name, fn in tests:
        status, body, err = fn()
        ok = status is not None and 200 <= status < 300
        detail = f"HTTP {status}" if status is not None else (err or "no response")
        if body:
            try:
                parsed = json.loads(body.decode("utf-8"))
                detail = f"{detail} ({parsed})"
            except Exception:
                pass
        results.append((name, ok, detail))
        # If we just started an agent_mining job, poll it briefly so downstream endpoints can be hit.
        if include_agent_mining and name.startswith("agent_mining/jobs (sample)") and status is not None and status // 100 == 2:
            try:
                data = json.loads(body.decode("utf-8")) if body else {}
            except Exception:
                data = {}
            job_id = data.get("job_id")
            if job_id:
                ok_poll, job_data, poll_err = _poll_job(f"{base_url}/agent_mining/jobs/{job_id}", timeout=120.0, interval=1.0)
                results.append((f"agent_mining/jobs/{job_id} poll", ok_poll, poll_err or "unknown"))
                if ok_poll:
                    st, body_latest, err_latest = _http_get(f"{base_url}/agent_mining/results/latest", timeout=request_timeout)
                    ok_latest = st is not None and 200 <= st < 300
                    detail_latest = f"HTTP {st}" if st is not None else (err_latest or "no response")
                    latest_payload = None
                    if body_latest:
                        try:
                            parsed_latest = json.loads(body_latest.decode("utf-8"))
                            latest_payload = parsed_latest
                            detail_latest = f"{detail_latest} ({parsed_latest.get('status','')})"
                        except Exception:
                            pass
                    results.append(("agent_mining/results/latest", ok_latest, detail_latest))
                    if ok_latest and latest_payload:
                        try:
                            # Save a recipe for the first mined class, then apply it to a synthetic image payload.
                            job_res = latest_payload.get("result") or {}
                            classes = job_res.get("classes") or []
                            first_class = classes[0] if classes else None
                            recipe_body = (first_class or {}).get("recipe") or {}
                            class_id = (first_class or {}).get("id")
                            class_name = (first_class or {}).get("name")
                            dataset_id = job_res.get("dataset_id") or sam3_dataset_id

                            if recipe_body and class_name:
                                label = f"fuzz_{class_name}_{int(time.time())}"
                                st_save, body_save, err_save = _http_post_json(
                                    f"{base_url}/agent_mining/recipes",
                                    {
                                        "dataset_id": dataset_id,
                                        "class_id": class_id,
                                        "class_name": class_name,
                                        "label": label,
                                        "recipe": recipe_body,
                                    },
                                    timeout=request_timeout,
                                )
                                ok_save = st_save is not None and 200 <= st_save < 300
                                detail_save = f"HTTP {st_save}" if st_save is not None else (err_save or "no response")
                                saved = None
                                if body_save:
                                    try:
                                        saved = json.loads(body_save.decode("utf-8"))
                                    except Exception:
                                        saved = None
                                results.append(("agent_mining/recipes (save)", ok_save, detail_save))

                                if ok_save and isinstance(saved, dict) and saved.get("id"):
                                    rid = saved.get("id")
                                    st_zip, _, err_zip = _http_get(
                                        f"{base_url}/agent_mining/recipes/{urllib.parse.quote(str(rid))}/export",
                                        timeout=request_timeout,
                                    )
                                    ok_zip = st_zip is not None and 200 <= st_zip < 300
                                    detail_zip = f"HTTP {st_zip}" if st_zip is not None else (err_zip or "no response")
                                    results.append(("agent_mining/recipes export", ok_zip, detail_zip))

                                    # Apply to an in-memory image payload to validate the portable recipe path.
                                    st_app, _, err_app = _http_post_json(
                                        f"{base_url}/agent_mining/apply_image",
                                        {
                                            "image_base64": img_b64,
                                            "image_name": image_name,
                                            "sam_variant": "sam3",
                                            "recipe": saved,
                                            "mask_threshold": 0.5,
                                            "min_size": 0,
                                            "simplify_epsilon": 0.5,
                                            "max_results": 50,
                                        },
                                        timeout=request_timeout,
                                    )
                                    ok_app = st_app is not None and 200 <= st_app < 300
                                    detail_app = f"HTTP {st_app}" if st_app is not None else (err_app or "no response")
                                    results.append(("agent_mining/apply_image", ok_app, detail_app))

                                    # Cleanup saved recipe so fuzzing doesn't spam the recipe list.
                                    try:
                                        del_req = urllib.request.Request(
                                            f"{base_url}/agent_mining/recipes/{urllib.parse.quote(str(rid))}",
                                            method="DELETE",
                                        )
                                        with urllib.request.urlopen(del_req, timeout=request_timeout) as resp_del:
                                            ok_del = 200 <= resp_del.status < 300
                                        results.append(("agent_mining/recipes delete", ok_del, ""))
                                    except Exception as exc:
                                        results.append(("agent_mining/recipes delete", False, f"exception:{exc}"))
                        except Exception as exc:
                            results.append(("agent_mining recipe save/apply", False, f"exception:{exc}"))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Spin up uvicorn and fuzz backend endpoints.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for uvicorn / connect host for reuse.")
    parser.add_argument("--port", default=8000, type=int, help="Bind port for uvicorn / connect port for reuse.")
    parser.add_argument("--base-url", default=None, help="Override base URL (e.g., http://localhost:8000).")
    parser.add_argument("--reuse-server", action="store_true", help="Do not launch uvicorn; assume it is already running.")
    parser.add_argument("--include-qwen", action="store_true", help="Exercise Qwen endpoints (requires model).")
    parser.add_argument("--include-sam3", action="store_true", help="Exercise SAM3 text endpoints (requires sam3 weights).")
    parser.add_argument("--include-auto", action="store_true", help="Exercise auto-class endpoints (sam_point_auto, sam_bbox_auto).")
    parser.add_argument("--include-clip", action="store_true", help="Exercise CLIP inference/activation endpoints.")
    parser.add_argument("--include-agent-mining", action="store_true", help="Exercise agent_mining endpoints (requires dataset).")
    parser.add_argument("--wait-seconds", type=float, default=45.0, help="How long to wait for the server to become responsive.")
    parser.add_argument("--request-timeout", type=float, default=30.0, help="Timeout (seconds) for each individual request.")
    parser.add_argument("--max-sam3-tests", type=int, default=30, help="Cap on SAM3 text prompt combinations to try.")
    parser.add_argument("--max-qwen-tests", type=int, default=8, help="Cap on Qwen prompt combinations to try.")
    parser.add_argument("--sam3-dataset-id", default=None, help="Dataset ID to use for prompt_helper tests.")
    args = parser.parse_args()

    base_url = args.base_url or f"http://{args.host}:{args.port}"
    proc = None
    try:
        if not args.reuse_server:
            env = os.environ.copy()
            cmd = [sys.executable, "-m", "uvicorn", "app:app", "--host", args.host, "--port", str(args.port), "--lifespan", "off"]
            proc = subprocess.Popen(cmd)
            print(f"[fuzzer] Launched uvicorn ({' '.join(cmd)})")
            ready, reason = wait_for_server(base_url, proc=proc, deadline=args.wait_seconds)
            if not ready:
                print(f"[fuzzer] Server did not become ready in time ({reason}).")
                return 1
        else:
            print(f"[fuzzer] Reusing server at {base_url}")
            ready, reason = wait_for_server(base_url, proc=None, deadline=args.wait_seconds)
            if not ready:
                print(f"[fuzzer] Server not reachable: {reason}")
                return 1

        results = run_tests(
            base_url,
            include_qwen=args.include_qwen,
            include_sam3=args.include_sam3,
            include_auto=args.include_auto,
            include_clip=args.include_clip,
            include_agent_mining=args.include_agent_mining,
            sam3_dataset_id=args.sam3_dataset_id,
            request_timeout=args.request_timeout,
            max_sam3_tests=max(1, args.max_sam3_tests),
            max_qwen_tests=max(1, args.max_qwen_tests),
        )
        failures = 0
        for name, ok, detail in results:
            prefix = "PASS" if ok else "FAIL"
            print(f"[{prefix}] {name}: {detail}")
            if not ok:
                failures += 1
        if failures:
            print(f"[fuzzer] Completed with {failures} failing test(s).")
            return 2
        print("[fuzzer] All tests passed.")
        return 0
    finally:
        if proc:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
