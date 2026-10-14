"""Cache-only replay of selected object rows in the saved feature space."""
import copy
import numpy as np
from fastapi import HTTPException

from services.dataset_pruning_components import object_records


class SelectedObjectFeatures:
    def __init__(self, rows, width):
        self.rows = rows
        self.shape = (max(rows, default=-1)+1, width)

    def __getitem__(self, indices):
        if np.isscalar(indices): return self.rows[int(indices)]
        return np.asarray([self.rows[int(index)] for index in indices], np.float32)


def reconstruct_cached_object_rows(api, cache, images, recipe, manifest):
    """Reconstruct only these images, with original global feature indices.

    Missing optional caches or legacy transforms remain unmeasured. This
    function cannot encode images, train heads, refit a transform or publish.
    """
    original = {(image['image_id'], obj['label_row']): obj['feature_index']
                for image in images for obj in image['objects']}
    if not original or recipe.get('evidence_mode') == 'scene_only':
        return SelectedObjectFeatures({}, 0)
    records = object_records(api, copy.deepcopy(images), recipe['bbox']['preprocessing'])
    matrices, weights = [], []
    for component in recipe['bbox']['components']:
        entry = manifest.get('components', {}).get('bbox_'+component['id'])
        if not entry or 'adjustment' not in entry or not entry.get('raw_recipe'):
            raise HTTPException(409, 'Direct comparison is unmeasured: its saved component/transform is unavailable. Recompute the selected pruning analysis.')
        raw = []
        keys = [cache.key(record['_image_sha256'], entry['raw_recipe'], record['geometry_key']) for record in records]
        for vector in cache.get_many(keys):
            if vector is None:
                raise HTTPException(409, 'Direct comparison is unmeasured: an optional raw descriptor was cleared or changed. Recompute missing features with the selected recipe.')
            if len(vector) != entry['dimension']:
                raise HTTPException(409, 'Direct comparison is unmeasured: a cached component dimension changed. Recompute the selected pruning analysis.')
            raw.append(vector)
        try:
            adjusted = api._class_analysis_replay_fitted_embedding_adjustment(np.asarray(raw, np.float32), records, entry['adjustment'])
        except ValueError as error:
            raise HTTPException(409, 'Direct comparison is unmeasured: '+str(error)) from error
        matrices.append(adjusted); weights.append(entry['weight'])
    if not matrices:
        raise HTTPException(409, 'Direct object comparison is unmeasured: no saved object representation is available.')
    fused = api.fuse_class_analysis_feature_branches(matrices, weights)
    return SelectedObjectFeatures({original[(record['image_id'], record['label_line_index'])]: fused[index]
                                   for index, record in enumerate(records)}, fused.shape[1])
