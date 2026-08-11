import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _single_thread_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        'OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'NUMEXPR_NUM_THREADS',
    ):
        env[key] = '1'
    return env


def _feature_names():
    return [
        'cand_has_yolo',
        'cand_has_rfdetr',
        'cand_has_sam3_text',
        'cand_has_sam3_similarity',
        'cand_score_yolo',
        'cand_score_rfdetr',
        'cand_raw_score_yolo',
        'cand_raw_score_rfdetr',
        'support_count_total',
        'support_atom_count',
        'support_atom_same_label_count',
        'support_run_count',
        'support_source_count',
        'support_score_mean',
        'support_score_max',
        'support_score_std',
        'support_iou_mean',
        'support_iou_max',
        'support_source_entropy',
        'support_source_entropy_norm',
        'support_source_max_share',
        'support_detector_share',
        'support_sam_share',
        'support_detector_count',
        'support_sam_count',
        'det_iou_max_yolo_same_label',
        'det_iou_max_rfdetr_same_label',
        'det_iou_max_detector_same_label',
        'det_iou_max_detector_any_label',
        'geom_center_x',
        'geom_center_y',
        'geom_width',
        'geom_height',
        'geom_area',
        'geom_aspect_ratio',
        'geom_prior_area_z',
        'geom_prior_aspect_z',
        'geom_prior_area_tail',
        'geom_prior_aspect_tail',
        'ctx_neighbor_count_all',
        'ctx_neighbor_count_same',
        'ctx_neighbor_ratio_same',
        'ctx_neighbor_score_mean_same',
        'ctx_total_area',
        'ctx_avg_area',
        'ctx_avg_aspect_ratio',
        'cand_run_max::yolo_full',
        'cand_run_count::yolo_full',
        'cand_run_max::sam3_similarity_windowed',
        'cand_run_count::sam3_similarity_windowed',
        'clf_prob::person',
        'clf_prob::truck',
        'sam3_text_max::person',
        'sam3_text_count::person',
        'sam3_sim_max::person',
        'sam3_sim_count::person',
        'ctx_source_count::person::yolo',
        'ctx_source_mean::person::yolo',
        'ctx_source_count::person::rfdetr',
        'ctx_source_mean::person::rfdetr',
        'ctx_source_count::person::sam3_text',
        'ctx_source_mean::person::sam3_text',
        'ctx_source_count::person::sam3_similarity',
        'ctx_source_mean::person::sam3_similarity',
        'sam3_text_max::truck',
        'sam3_text_count::truck',
        'sam3_sim_max::truck',
        'sam3_sim_count::truck',
        'ctx_source_count::truck::yolo',
        'ctx_source_mean::truck::yolo',
        'ctx_source_count::truck::rfdetr',
        'ctx_source_mean::truck::rfdetr',
        'ctx_source_count::truck::sam3_text',
        'ctx_source_mean::truck::sam3_text',
        'ctx_source_count::truck::sam3_similarity',
        'ctx_source_mean::truck::sam3_similarity',
    ]


def _row_values(label: str, positive: bool, source: str):
    is_person = label == 'person'
    label_prob = 0.86 if positive else 0.28
    other_prob = 1.0 - label_prob
    has_detector = 0.0 if source.startswith('sam3') else 1.0
    has_yolo = 1.0 if source == 'yolo' else 0.0
    has_rfdetr = 1.0 if source == 'rfdetr' else 0.0
    has_sam_text = 1.0 if source == 'sam3_text' else 0.0
    has_sam_sim = 1.0 if source == 'sam3_similarity' else 0.0
    support_detector_share = 0.75 if positive and has_detector else (0.15 if not has_detector else 0.35)
    support_sam_share = 1.0 - support_detector_share
    base = {
        'cand_has_yolo': has_yolo,
        'cand_has_rfdetr': has_rfdetr,
        'cand_has_sam3_text': has_sam_text,
        'cand_has_sam3_similarity': has_sam_sim,
        'cand_score_yolo': 0.8 if has_yolo else 0.0,
        'cand_score_rfdetr': 0.78 if has_rfdetr else 0.0,
        'cand_raw_score_yolo': 0.8 if has_yolo else 0.0,
        'cand_raw_score_rfdetr': 0.78 if has_rfdetr else 0.0,
        'support_count_total': 6 if positive else 2,
        'support_atom_count': 5 if positive else 1,
        'support_atom_same_label_count': 4 if positive else 1,
        'support_run_count': 3 if positive else 1,
        'support_source_count': 2 if positive else 1,
        'support_score_mean': 0.72 if positive else 0.22,
        'support_score_max': 0.9 if positive else 0.4,
        'support_score_std': 0.1 if positive else 0.2,
        'support_iou_mean': 0.65 if positive else 0.18,
        'support_iou_max': 0.92 if positive else 0.25,
        'support_source_entropy': 0.5 if positive else 0.9,
        'support_source_entropy_norm': 0.3 if positive else 0.8,
        'support_source_max_share': 0.7 if positive else 1.0,
        'support_detector_share': support_detector_share,
        'support_sam_share': support_sam_share,
        'support_detector_count': 2 if has_detector else 0,
        'support_sam_count': 1 if positive else 2,
        'det_iou_max_yolo_same_label': 0.82 if positive and has_yolo else 0.0,
        'det_iou_max_rfdetr_same_label': 0.81 if positive and has_rfdetr else 0.0,
        'det_iou_max_detector_same_label': 0.83 if positive and has_detector else 0.1,
        'det_iou_max_detector_any_label': 0.83 if positive and has_detector else 0.1,
        'geom_center_x': 0.4,
        'geom_center_y': 0.5,
        'geom_width': 0.2 if positive else 0.1,
        'geom_height': 0.3 if positive else 0.1,
        'geom_area': 0.06 if positive else 0.01,
        'geom_aspect_ratio': 0.8,
        'geom_prior_area_z': 0.3 if positive else 1.4,
        'geom_prior_aspect_z': 0.2 if positive else 1.1,
        'geom_prior_area_tail': 0.0 if positive else 0.5,
        'geom_prior_aspect_tail': 0.0 if positive else 0.4,
        'ctx_neighbor_count_all': 5 if positive else 2,
        'ctx_neighbor_count_same': 3 if positive else 1,
        'ctx_neighbor_ratio_same': 0.6 if positive else 0.3,
        'ctx_neighbor_score_mean_same': 0.75 if positive else 0.3,
        'ctx_total_area': 0.4,
        'ctx_avg_area': 0.08,
        'ctx_avg_aspect_ratio': 0.9,
        'cand_run_max::yolo_full': 0.82 if has_yolo else 0.0,
        'cand_run_count::yolo_full': 1 if has_yolo else 0,
        'cand_run_max::sam3_similarity_windowed': 0.76 if has_sam_sim else 0.0,
        'cand_run_count::sam3_similarity_windowed': 1 if has_sam_sim else 0,
        'clf_prob::person': label_prob if is_person else other_prob,
        'clf_prob::truck': other_prob if is_person else label_prob,
        'sam3_text_max::person': 0.6 if is_person else 0.0,
        'sam3_text_count::person': 2 if is_person else 0,
        'sam3_sim_max::person': 0.65 if is_person else 0.0,
        'sam3_sim_count::person': 4 if is_person else 0,
        'ctx_source_count::person::yolo': 1 if is_person and has_yolo else 0,
        'ctx_source_mean::person::yolo': 0.8 if is_person and has_yolo else 0.0,
        'ctx_source_count::person::rfdetr': 1 if is_person and has_rfdetr else 0,
        'ctx_source_mean::person::rfdetr': 0.78 if is_person and has_rfdetr else 0.0,
        'ctx_source_count::person::sam3_text': 1 if is_person and has_sam_text else 0,
        'ctx_source_mean::person::sam3_text': 0.6 if is_person and has_sam_text else 0.0,
        'ctx_source_count::person::sam3_similarity': 1 if is_person and has_sam_sim else 0,
        'ctx_source_mean::person::sam3_similarity': 0.76 if is_person and has_sam_sim else 0.0,
        'sam3_text_max::truck': 0.6 if not is_person else 0.0,
        'sam3_text_count::truck': 2 if not is_person else 0,
        'sam3_sim_max::truck': 0.65 if not is_person else 0.0,
        'sam3_sim_count::truck': 4 if not is_person else 0,
        'ctx_source_count::truck::yolo': 1 if (not is_person) and has_yolo else 0,
        'ctx_source_mean::truck::yolo': 0.8 if (not is_person) and has_yolo else 0.0,
        'ctx_source_count::truck::rfdetr': 1 if (not is_person) and has_rfdetr else 0,
        'ctx_source_mean::truck::rfdetr': 0.78 if (not is_person) and has_rfdetr else 0.0,
        'ctx_source_count::truck::sam3_text': 1 if (not is_person) and has_sam_text else 0,
        'ctx_source_mean::truck::sam3_text': 0.6 if (not is_person) and has_sam_text else 0.0,
        'ctx_source_count::truck::sam3_similarity': 1 if (not is_person) and has_sam_sim else 0,
        'ctx_source_mean::truck::sam3_similarity': 0.76 if (not is_person) and has_sam_sim else 0.0,
    }
    return base


def _build_fixture_npz(path: Path):
    names = _feature_names()
    rows = []
    y = []
    meta = []
    for image_idx in range(12):
        label = 'person' if image_idx % 2 == 0 else 'truck'
        pos_source = 'yolo' if image_idx % 3 == 0 else 'rfdetr'
        neg_source = 'sam3_similarity'
        for positive, source in [(True, pos_source), (False, neg_source)]:
            values = _row_values(label, positive, source)
            rows.append([values[name] for name in names])
            y.append(1 if positive else 0)
            source_list = [source] if source.startswith('sam3') else [source, 'sam3_similarity']
            if positive and source in {'yolo', 'rfdetr'}:
                score_by_source = {source: 0.82, 'sam3_similarity': 0.55}
                primary_source = 'sam3_similarity'
                source_list = [source, 'sam3_similarity']
            else:
                score_by_source = {source: 0.55 if positive else 0.42}
                primary_source = source
            meta.append(json.dumps({
                'image': f'img_{image_idx}.jpg',
                'label': label,
                'bbox_xyxy_px': [10.0, 10.0, 40.0, 40.0],
                'score_source': primary_source,
                'source_list': source_list,
                'score_by_source': score_by_source,
            }))
    X = np.asarray(rows, dtype=np.float32)
    np.savez(path, X=X, y=np.asarray(y, dtype=np.int64), meta=np.asarray(meta, dtype=object), feature_names=np.asarray(names, dtype=object))


def _train_base_model(npz_path: Path, out_prefix: Path):
    model_path = out_prefix.with_suffix('.json')
    meta_path = out_prefix.with_suffix('.meta.json')
    code = """
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

npz_path = Path(sys.argv[1])
model_path = Path(sys.argv[2])
meta_path = Path(sys.argv[3])
data = np.load(npz_path, allow_pickle=True)
X = data['X'].astype(np.float32)
y = data['y'].astype(np.int64)
meta_rows = [json.loads(str(row)) for row in data['meta']]
train_images = {f'img_{idx}.jpg' for idx in range(8)}
val_images = {f'img_{idx}.jpg' for idx in range(8, 12)}
train_idx = [idx for idx, row in enumerate(meta_rows) if row['image'] in train_images]
val_idx = [idx for idx, row in enumerate(meta_rows) if row['image'] in val_images]
dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
dval = xgb.DMatrix(X[val_idx], label=y[val_idx])
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 3,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1.0,
    'gamma': 0.0,
    'lambda': 1.0,
    'alpha': 0.0,
    'tree_method': 'hist',
    'seed': 7,
    'nthread': 1,
}
booster = xgb.train(params, dtrain, num_boost_round=40, evals=[(dtrain, 'train'), (dval, 'val')], verbose_eval=False)
booster.save_model(str(model_path))
meta_payload = {
    'model_path': str(model_path),
    'calibrated_threshold': 0.5,
    'calibrated_thresholds': {'person': 0.5, 'truck': 0.5},
    'feature_mean': None,
    'feature_std': None,
    'log1p_counts': False,
    'standardize': False,
    'split_seed': 7,
    'val_ratio': 0.2,
    'split_val_images': sorted(val_images),
    'split_train_images': sorted(train_images),
    'split_type': 'fixed',
    'xgb_params': params,
    'calibration_optimize': 'f1',
    'n_estimators': 40,
    'best_iteration': int(getattr(booster, 'best_iteration', 0) or 0),
    'ensemble_policy': {
        'sam_bias_scope': 'sam_only',
        'logit_bias_by_source_class': {
            'sam3_text': {'__default__': -1.4},
            'sam3_similarity': {'__default__': -1.2},
        },
        'sam_only_min_prob_default': 0.15,
        'consensus_iou_default': 0.7,
        'consensus_iou_by_source_class': {
            'sam3_text': {'__default__': 0.7},
            'sam3_similarity': {'__default__': 0.7},
        },
        'consensus_class_aware': True,
    },
    'split_head': {'enabled': False, 'route': 'detector_support', 'models': {}},
    'sam3_text_quality': {'enabled': False},
}
meta_path.write_text(json.dumps(meta_payload), encoding='utf-8')
"""
    result = subprocess.run(
        [sys.executable, '-c', code, str(npz_path), str(model_path), str(meta_path)],
        env=_single_thread_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        if "No module named 'xgboost'" in result.stderr:
            pytest.skip("xgboost is not installed")
        raise AssertionError(f"base XGBoost training failed\n{result.stdout}\n{result.stderr}".strip())
    return model_path, meta_path


@pytest.mark.parametrize(
    ("variant", "expected_variants"),
    [
        ("bakeoff", {"lreg", "xgb"}),
        ("xgb", {"xgb"}),
        ("lreg", {"lreg"}),
    ],
)
def test_train_policy_layer_and_runtime_scoring(tmp_path: Path, variant: str, expected_variants: set[str]):
    npz_path = tmp_path / 'labeled.npz'
    _build_fixture_npz(npz_path)
    model_path, meta_path = _train_base_model(npz_path, tmp_path / 'ensemble_xgb')
    policy_dir = tmp_path / 'policy_layer'
    subprocess.run(
        [
            sys.executable,
            'tools/train_policy_layer.py',
            '--input', str(npz_path),
            '--base-model', str(model_path),
            '--base-meta', str(meta_path),
            '--output-dir', str(policy_dir),
            '--variant', variant,
            '--seed', '7',
            '--nested-folds', '4',
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=_single_thread_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    selection = json.loads((policy_dir / 'policy_layer_selection.json').read_text())
    updated_meta = json.loads(meta_path.read_text())
    assert set(selection['trained_variants']) == expected_variants
    assert selection['selected_variant'] in expected_variants
    assert updated_meta['selected_policy_layer']['variant'] == selection['selected_variant']
    assert (policy_dir / 'policy_layer_report.md').exists()
    if variant == 'xgb':
        assert (policy_dir / 'ensemble_policy_xgb.json').exists()
        assert not (policy_dir / 'ensemble_policy_lr.joblib').exists()
    elif variant == 'lreg':
        assert (policy_dir / 'ensemble_policy_lr.joblib').exists()
        assert not (policy_dir / 'ensemble_policy_xgb.json').exists()
    else:
        assert (policy_dir / 'ensemble_policy_xgb.json').exists()
        assert (policy_dir / 'ensemble_policy_lr.joblib').exists()

    scored_path = tmp_path / 'scored.jsonl'
    subprocess.run(
        [
            sys.executable,
            'tools/score_ensemble_candidates_xgb.py',
            '--model', str(model_path),
            '--meta', str(meta_path),
            '--data', str(npz_path),
            '--output', str(scored_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=_single_thread_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [json.loads(line) for line in scored_path.read_text().splitlines() if line.strip()]
    assert lines
    assert all('ensemble_policy_variant' in line for line in lines)
    assert {line['ensemble_policy_variant'] for line in lines} == {selection['selected_variant']}


def test_selected_policy_resolves_from_copied_meta(tmp_path: Path):
    npz_path = tmp_path / 'labeled.npz'
    _build_fixture_npz(npz_path)
    model_path, meta_path = _train_base_model(npz_path, tmp_path / 'ensemble_xgb')
    policy_dir = tmp_path / 'policy_layer'
    subprocess.run(
        [
            sys.executable,
            'tools/train_policy_layer.py',
            '--input', str(npz_path),
            '--base-model', str(model_path),
            '--base-meta', str(meta_path),
            '--output-dir', str(policy_dir),
            '--variant', 'xgb',
            '--seed', '7',
            '--nested-folds', '4',
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=_single_thread_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    copied_dir = tmp_path / 'copied_meta'
    copied_dir.mkdir()
    copied_meta = copied_dir / meta_path.name
    copied_meta.write_text(meta_path.read_text(), encoding='utf-8')

    scored_path = tmp_path / 'scored_from_copied_meta.jsonl'
    subprocess.run(
        [
            sys.executable,
            'tools/score_ensemble_candidates_xgb.py',
            '--model', str(model_path),
            '--meta', str(copied_meta),
            '--data', str(npz_path),
            '--output', str(scored_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=_single_thread_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [json.loads(line) for line in scored_path.read_text().splitlines() if line.strip()]
    assert lines
    assert {line['ensemble_policy_variant'] for line in lines} == {'xgb'}
