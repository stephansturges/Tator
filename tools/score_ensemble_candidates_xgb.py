#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools.policy_runtime import (
    apply_hand_policy,
    apply_selected_policy,
    load_selected_policy,
    predict_base_probabilities,
    resolve_thresholds,
    transform_base_features,
    _should_apply_source_bias as _runtime_should_apply_source_bias,
)


def _parse_meta_rows(meta_raw) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in meta_raw:
        if isinstance(row, dict):
            rows.append(dict(row))
            continue
        try:
            rows.append(json.loads(str(row)))
        except Exception:
            rows.append({})
    return rows


def _should_apply_source_bias(policy, *, primary_source, has_detector_support):
    return _runtime_should_apply_source_bias(
        policy,
        primary_source=primary_source,
        has_detector_support=has_detector_support,
    )


def _load_policy(path_or_json) -> Dict[str, Any]:
    if not path_or_json:
        return {}
    raw = str(path_or_json).strip()
    if not raw:
        return {}
    payload = raw
    if raw[:1] not in "{[":
        maybe_path = Path(raw)
        try:
            if maybe_path.exists() and maybe_path.is_file():
                payload = maybe_path.read_text(encoding="utf-8")
        except OSError:
            payload = raw
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --policy-json payload: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("Invalid --policy-json payload: expected JSON object.")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidates with ensemble XGBoost.")
    parser.add_argument("--model", required=True, help="Model .json path.")
    parser.add_argument("--meta", required=True, help="Model meta json.")
    parser.add_argument("--data", required=True, help="Input .npz data.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold.")
    parser.add_argument(
        "--policy-json",
        type=str,
        default=None,
        help="Optional legacy policy JSON file/string override when forcing hand policy.",
    )
    parser.add_argument(
        "--force-hand-policy",
        action="store_true",
        help="Ignore any selected learned policy artifact and apply the hand policy path.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    meta_path = Path(args.meta)
    meta = json.loads(meta_path.read_text())
    data = np.load(args.data, allow_pickle=True)
    X_raw = data["X"].astype(np.float32)
    feature_names = [str(name) for name in data.get("feature_names", [])]
    parsed_meta = _parse_meta_rows(data["meta"])

    X_base = transform_base_features(X_raw, feature_names, meta)
    base_probs = predict_base_probabilities(X_base, meta_rows=parsed_meta, model_path=model_path, meta=meta)

    selected_policy = None if args.force_hand_policy else load_selected_policy(
        meta,
        base_dir=model_path.parent,
        meta_dir=meta_path.parent,
    )
    out_path = Path(args.output)
    default_threshold, thresholds_by_label = resolve_thresholds(meta)

    if selected_policy is not None:
        final_probs, policy_info = apply_selected_policy(
            X_full=X_raw,
            feature_names_full=feature_names,
            meta_rows=parsed_meta,
            base_probs=base_probs,
            selected_policy=selected_policy,
        )
        selected_meta = selected_policy["meta"]
        default_threshold = float(selected_meta.get("calibrated_threshold") or default_threshold)
        thresholds_by_label = {
            str(k): float(v)
            for k, v in (selected_meta.get("calibrated_thresholds") or {}).items()
        } or thresholds_by_label
        with out_path.open("w", encoding="utf-8") as f:
            for idx, entry in enumerate(parsed_meta):
                label = str(entry.get("label") or "").strip().lower()
                thr = float(args.threshold) if args.threshold is not None else float(thresholds_by_label.get(label, default_threshold))
                final_prob = float(final_probs[idx])
                scored = dict(entry)
                scored["ensemble_prob_raw"] = float(base_probs[idx])
                scored["ensemble_prob"] = final_prob
                scored["ensemble_accept"] = bool(final_prob >= thr)
                scored["ensemble_threshold"] = float(thr)
                scored["ensemble_policy_variant"] = str(selected_policy.get("variant") or "")
                scored["ensemble_policy_schema_hash"] = str(policy_info.get("feature_schema_hash") or "")
                scored["ensemble_policy_blocked"] = False
                scored["ensemble_policy_block_reason"] = None if final_prob >= thr else "threshold"
                f.write(json.dumps(scored, ensure_ascii=True) + "\n")
        return

    policy = {}
    if args.policy_json:
        policy = _load_policy(args.policy_json)
    elif isinstance(meta.get("ensemble_policy"), dict):
        policy = dict(meta.get("ensemble_policy"))

    legacy_rows = apply_hand_policy(
        probs=base_probs,
        meta_rows=parsed_meta,
        policy=policy,
        default_threshold=default_threshold,
        thresholds_by_label=thresholds_by_label,
        threshold_override=float(args.threshold) if args.threshold is not None else None,
    )
    with out_path.open("w", encoding="utf-8") as f:
        for idx, entry in enumerate(parsed_meta):
            row = legacy_rows[idx]
            scored = dict(entry)
            scored["ensemble_prob_raw"] = float(row["prob_raw"])
            scored["ensemble_prob"] = float(row["prob"])
            scored["ensemble_accept"] = bool(row["accept"])
            scored["ensemble_threshold"] = float(row["threshold"])
            scored["ensemble_policy_variant"] = "legacy_hand_policy"
            scored["ensemble_policy_blocked"] = bool(row["blocked_reason"] and row["blocked_reason"] != "threshold")
            scored["ensemble_policy_block_reason"] = str(row["blocked_reason"]) if row["blocked_reason"] else None
            f.write(json.dumps(scored, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
