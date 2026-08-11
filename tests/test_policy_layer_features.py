import numpy as np

from tools.policy_layer_features import build_policy_feature_matrix


def test_build_policy_feature_matrix_derives_expected_flags_and_classifier_summaries():
    feature_names = [
        'cand_has_yolo',
        'cand_has_rfdetr',
        'cand_has_sam3_text',
        'cand_has_sam3_similarity',
        'support_detector_share',
        'support_sam_share',
        'support_iou_max',
        'det_iou_max_detector_same_label',
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
    ]
    X = np.array([
        [1, 0, 1, 0, 0.7, 0.3, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.12, 1.2, 0.1, 0.2, 0.0, 0.0, 5, 2, 0.4, 0.8, 0.2, 0.1, 1.1, 0.9, 2, 0.8, 0.2, 0.6, 3, 0.4, 2, 1, 0.9, 0, 0.0, 1, 0.5, 1, 0.3],
        [0, 1, 0, 1, 0.2, 0.8, 0.3, 0.4, 0.5, 0.6, 0.2, 0.3, 0.06, 0.8, 0.4, 0.5, 0.1, 0.2, 2, 1, 0.5, 0.4, 0.1, 0.05, 0.9, 0.0, 0, 0.3, 0.7, 0.1, 1, 0.8, 6, 0, 0.0, 1, 0.7, 0, 0.0, 3, 0.8],
    ], dtype=np.float32)
    meta_rows = [
        {'image': 'img1.jpg', 'label': 'person', 'score_source': 'sam3_text', 'source_list': ['yolo', 'sam3_text']},
        {'image': 'img2.jpg', 'label': 'person', 'score_source': 'sam3_similarity', 'source_list': ['sam3_similarity']},
    ]
    base_probs = np.array([0.8, 0.35], dtype=np.float32)

    bundle = build_policy_feature_matrix(X, feature_names, meta_rows, base_probs)
    names = bundle['feature_names']
    matrix = bundle['X']

    idx_margin = names.index('clf_prob_margin')
    idx_top1 = names.index('clf_label_is_top1')
    idx_sam_only = names.index('is_sam_only')
    idx_detector_support = names.index('has_detector_support')
    idx_primary_text = names.index('is_sam3_text_primary')
    idx_primary_sim = names.index('is_sam3_similarity_primary')

    assert matrix.shape[0] == 2
    assert np.isclose(matrix[0, idx_margin], np.float32(0.6))
    assert np.isclose(matrix[0, idx_top1], np.float32(1.0))
    assert np.isclose(matrix[0, idx_detector_support], np.float32(1.0))
    assert np.isclose(matrix[0, idx_sam_only], np.float32(0.0))
    assert np.isclose(matrix[0, idx_primary_text], np.float32(1.0))

    assert np.isclose(matrix[1, idx_margin], np.float32(0.4))
    assert np.isclose(matrix[1, idx_detector_support], np.float32(0.0))
    assert np.isclose(matrix[1, idx_sam_only], np.float32(1.0))
    assert np.isclose(matrix[1, idx_primary_sim], np.float32(1.0))
    assert bundle['feature_schema_hash']


def test_build_policy_feature_matrix_adds_anchor_similarity_features():
    feature_names = [
        'cand_has_yolo',
        'cand_has_rfdetr',
        'cand_has_sam3_text',
        'cand_has_sam3_similarity',
        'support_detector_share',
        'support_sam_share',
        'support_iou_max',
        'det_iou_max_detector_same_label',
        'clf_emb_rp::000',
        'clf_emb_rp::001',
        'clf_prob::person',
        'clf_prob::truck',
    ]
    X = np.array([
        [1, 0, 0, 0, 1.0, 0.0, 0.8, 0.9, 1.0, 0.0, 0.9, 0.1],
        [1, 0, 0, 0, 1.0, 0.0, 0.8, 0.9, 0.95, 0.05, 0.85, 0.15],
        [0, 0, 0, 1, 0.0, 1.0, 0.2, 0.1, 0.98, 0.02, 0.75, 0.25],
    ], dtype=np.float32)
    meta_rows = [
        {'image': 'img1.jpg', 'label': 'person', 'score_source': 'yolo', 'source_list': ['yolo']},
        {'image': 'img1.jpg', 'label': 'person', 'score_source': 'rfdetr', 'source_list': ['rfdetr']},
        {'image': 'img1.jpg', 'label': 'person', 'score_source': 'sam3_similarity', 'source_list': ['sam3_similarity']},
    ]
    base_probs = np.array([0.99, 0.96, 0.35], dtype=np.float32)

    bundle = build_policy_feature_matrix(
        X,
        feature_names,
        meta_rows,
        base_probs,
        anchor_similarity={
            'enabled': True,
            'min_base_prob': 0.95,
            'topk_same_label': 2,
            'topk_any': 4,
            'require_detector_support': True,
        },
    )
    names = bundle['feature_names']
    matrix = bundle['X']

    idx_same_count = names.index('anchor_same_label_count')
    idx_same_max = names.index('anchor_same_label_cos_max')
    idx_margin = names.index('anchor_margin_same_vs_other')

    assert matrix.shape[1] > len(feature_names)
    assert matrix[2, idx_same_count] >= 1.0
    assert matrix[2, idx_same_max] > 0.9
    assert matrix[2, idx_margin] > 0.0
    assert bundle['anchor_similarity']['enabled'] is True
