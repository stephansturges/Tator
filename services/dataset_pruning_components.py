"""Reuse the DQE crop, SAM, SALAD, C-RADIO and independent adjustment producers.

Raw component identity excludes fusion weights and fitted population transforms.
Every branch is normalized independently; final weights use the established
sqrt-weight concatenation. Whole-image C-RADIO batches contain equal dimensions,
so its preserve-input policy cannot depend on unrelated batch companions.
"""
from collections import defaultdict
import copy
import json
from pathlib import Path
import hashlib
import inspect
import marshal

import numpy as np
from fastapi import HTTPException

from services.dataset_pruning import digest
from services.dataset_pruning_cache import file_state
from services.data_ingestion import normalize_rows
from services.dataset_pruning_correction import adjust_component_to_memmap, validate_strength
from utils.class_analysis_quality import resolve_quality_recipe

PREPROCESS_FIELDS = ('crop_mode', 'padding_ratio', 'preprocess_mode', 'canonical_size', 'background_mode', 'embedding_view_mode')


def cradio_component_dimension(model, pooling, view_mode):
    """Use the shared encoder's native summary/spatial spec, then concatenate views."""
    from utils.cradio_embedding import _import_cradio_mlx, _cradio_mlx_variant
    try:
        _import_cradio_mlx()
        from cradio_mlx.mlx_so400m import VARIANT_SPECS
        spec = VARIANT_SPECS[_cradio_mlx_variant(model)]
        summary, spatial = spec.embed_dim * len(spec.summary_idxs), spec.embed_dim
    except (ImportError, RuntimeError):
        # The same two supported checkpoint topologies are used by Torch when
        # the optional MLX package is absent. Actual encoded width is checked
        # again before assembly; unsupported models are not guessed.
        spatial = {'nvidia/C-RADIOv4-SO400M': 1152, 'nvidia/C-RADIOv4-H': 1280}.get(model)
        if spatial is None: raise HTTPException(400, 'Unknown C-RADIO descriptor topology')
        summary = 2 * spatial
    width = summary if pooling == 'summary' else spatial if pooling == 'spatial_mean' else summary + spatial
    return width * (2 if view_mode in {'tight_standard', 'standard_context', 'tight_context'} else 1)


def inherited_components(api, cfg):
    resolved = api._class_analysis_resolved_recipe(cfg)
    components = []
    for branch in resolved['projection_features']['branches']:
        nested = branch.get('components') or [branch]
        for item in nested:
            identifier = item['id']
            component = {'id': identifier, 'encoder': 'dinov3' if identifier == 'salad' else identifier,
                         'model': cfg.get('quality_cradio_model') or api.CRADIO_DEFAULT_MODEL if identifier == 'cradio' else cfg.get('encoder_model') or api.CLASS_ANALYSIS_DEFAULT_DINOV3_MODEL,
                         'weight': float(branch.get('weight', 1)) * float(item.get('weight', 1)) if branch.get('components') else float(item.get('weight', 1)),
                         'dimension': 768 if identifier == 'sam3' else int(item.get('dimensions', 0)),
                         'pooling': 'summary' if identifier == 'cradio' else cfg.get('dinov3_pooling', 'pooler'),
                         'adjustment': 'none' if identifier == 'cradio' else cfg.get('embedding_adjustment', 'remove_size_bias'),
                         'status': 'existing DQE producer; cached components reused where content and producer are attested'}
            components.append(component)
            if identifier == 'sam3':
                from services.mlx_sam3 import mlx_sam3_status
                component['model'] = mlx_sam3_status('mlx-bf16')['model_id']
                component['pooling'] = 'mask_multiscale'
            if identifier == 'salad':
                component['status'] = 'Historical DQE crop head was not persisted: resolve an identical persisted version or train one new consistent head and encode every current crop'
                component['requires_training_if_no_compatible_head'] = True
    return resolved, components


def object_records(api, images, preprocessing):
    records = []
    primary, padding, _ = api._embedding_view_specs(crop_mode=preprocessing['crop_mode'], padding_ratio=preprocessing['padding_ratio'], view_mode=preprocessing['embedding_view_mode'])[0]
    for image in images:
        image_records = []
        for obj in image['objects']:
            obj['feature_index'] = len(records) + len(image_records)
            width, height = image['width'], image['height']
            # Use the original DQE parser's pixel arithmetic and clipping.
            # Rearranging normalized arithmetic changes float hex/cache keys
            # and can move a padded/context boundary by one source pixel.
            geometry = api._class_analysis_parse_yolo_geometry(image['label_lines'][obj['label_row']],
                         image_width=width, image_height=height)
            if not geometry or geometry['kind'] != 'bbox':
                raise HTTPException(409, 'Object geometry is not supported by the shared bbox producer')
            bbox = geometry['bbox_xyxy']
            bounds = api._class_analysis_crop_bounds(bbox, image_width=width, image_height=height, crop_mode=primary, padding_ratio=padding)
            crop_key = api._class_analysis_cache_digest({'version': api.CLASS_ANALYSIS_CACHE_VERSION, 'image_sha256': image['image_sha256'],
                'bbox_float64_hex': [float(v).hex() for v in bbox], 'crop_xyxy': [int(v) for v in bounds],
                **{key: preprocessing[key] for key in PREPROCESS_FIELDS},
                'padding_ratio': round(float(preprocessing['padding_ratio']), 6)})
            image_records.append({'_image_path': image['image_path'], '_image_sha256': image['image_sha256'],
                '_image_width': width, '_image_height': height, 'bbox_xyxy': bbox, 'crop_xyxy': bounds,
                'width': int(round(bbox[2]-bbox[0])), 'height': int(round(bbox[3]-bbox[1])),
                'crop_cache_key': crop_key, 'geometry_key': obj['geometry_key'], 'image_id': image['image_id'],
                'image_name': image['image_relpath'], 'image_relpath': image['image_relpath'], 'split': image['split'],
                'class_id': obj['class_id'], 'label_line_index': obj['label_row'], 'label_line': image['label_lines'][obj['label_row']],
                'embedding_views': [], 'embedding_view_mode': preprocessing['embedding_view_mode']})
        pack = api._class_analysis_image_pack_group_key(image['image_sha256'], image_records)
        for record in image_records: record['image_pack_key'] = pack
        records.extend(image_records)
    return records


def raw_component_recipe(api, component, preprocessing, cfg):
    base = component['encoder']
    if base == 'sam3':
        import importlib.metadata
        import dataclasses
        import platform
        import sys
        import services.mlx_sam3 as sam_runtime
        from services.mlx_sam3 import mlx_sam3_status
        runtime = {'backend': 'mlx-bf16', 'descriptor_scales': 3, 'dimension': 768,
                   'policy': 'full_source_image_mask_pool_or_bbox_fallback', 'status': mlx_sam3_status('mlx-bf16')}
        model_path = Path(runtime['status'].get('model_path') or '')
        files = [model_path] if model_path.is_file() else sorted(model_path.glob('*.safetensors')) + sorted(model_path.glob('config.json')) if model_path.is_dir() else []
        runtime['checkpoint_artifacts'] = [{'name': path.name, 'sha256': api._class_analysis_file_sha256_cached(path), 'bytes': path.stat().st_size} for path in files]
        runtime['library_versions'] = {}
        runtime['library_versions']['python'] = platform.python_version()
        for package in ['mlx', 'mlx-vlm', 'Pillow', 'numpy']:
            try: runtime['library_versions'][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: runtime['library_versions'][package] = 'unavailable'
        dependencies = [Path(sam_runtime.__file__)]
        try:
            package_root = Path(importlib.metadata.distribution('mlx-vlm').locate_file('mlx_vlm/models/sam3'))
            dependencies.extend(sorted(package_root.rglob('*.py')))
        except importlib.metadata.PackageNotFoundError: pass
        runtime['implementation_files'] = [{'path': str(path), 'sha256': api._class_analysis_file_sha256_cached(path)} for path in dependencies]
        loaded = {}
        for module_name, module in list(sys.modules.items()):
            if module_name != 'services.mlx_sam3' and not module_name.startswith('mlx_vlm.models.sam3'): continue
            module_path = Path(getattr(module, '__file__', '') or '')
            expected = {}
            if module_path.suffix == '.py' and module_path.is_file():
                def collect_code(code):
                    expected[code.co_qualname] = code
                    for child in code.co_consts:
                        if inspect.iscode(child): collect_code(child)
                collect_code(compile(module_path.read_bytes(), str(module_path), 'exec', dont_inherit=True))
            for name, value in vars(module).items():
                functions = [(name, value)] if inspect.isfunction(value) else [(name+'.'+key, member) for key, member in vars(value).items() if inspect.isfunction(member)] if inspect.isclass(value) and getattr(value, '__module__', None) == module_name else []
                for label, function in functions:
                    if getattr(function, '__module__', None) == module_name:
                        code = inspect.unwrap(function).__code__
                        generated_dataclass = (inspect.isclass(value) and dataclasses.is_dataclass(value)
                            and code.co_filename == '<string>' and label.rsplit('.', 1)[-1] in
                            {'__init__', '__repr__', '__eq__', '__hash__', '__setattr__', '__delattr__'})
                        if not generated_dataclass and expected.get(code.co_qualname) != code:
                            raise HTTPException(409, 'SAM3 implementation changed after loading ('+module_name+'.'+label+'). Restart this backend before producing reusable descriptors.')
                        if module_name == 'services.mlx_sam3':
                            # marshal's string-reference/intern flags vary
                            # across otherwise code-equivalent processes. The
                            # verified source and qualified function name are
                            # stable; loaded-code equality above still rejects
                            # a serving process whose implementation drifted.
                            loaded[module_name+'.'+label] = digest({'source_sha256': api._class_analysis_file_sha256_cached(module_path),
                                                                  'qualname': code.co_qualname,
                                                                  'generated_dataclass': generated_dataclass,
                                                                  'generated_code': hashlib.sha256(marshal.dumps(code, 0)).hexdigest() if generated_dataclass else None})
        runtime['loaded_implementation'] = loaded
        runtime['dependency_policy'] = 'Pinned package sources and versions; already-loaded package code checked against those sources. Service loaded-code identity is stable before and after model loading.'
        runtime['persistent_cache_safe'] = bool(files)
    elif base == 'cradio':
        runtime = api.cradio_runtime_identity(component['model'], requested_backend=cfg.get('cradio_backend', 'auto'),
                  pooling=component['pooling'], output_mode='summary_only' if component['pooling'] == 'summary' else 'summary_and_spatial')
        from services.dataset_pruning_adapter import cradio_preprocessing_identity
        preprocessing_identity = cradio_preprocessing_identity(api, runtime)
        if preprocessing_identity is not None: runtime['preprocessing'] = preprocessing_identity
        if not runtime.get('persistent_cache_safe'): runtime['process_epoch'] = api.CLASS_ANALYSIS_UNRESOLVED_RUNTIME_EPOCH
    else:
        runtime = api._local_salad_encoder_identity(base, component['model'], backend=cfg.get('cradio_backend', 'auto') if base == 'cradio' else '', pooling=component['pooling'])
    functions = [api._class_analysis_stream_encode_records_locked, api._encode_embedding_items_for_head, api._encode_pil_batch_for_head, api._embedding_make_crop_views]
    if base == 'sam3': functions.extend([api._class_analysis_extract_sam3_mask_embeddings, api._class_analysis_sam3_mask_feature_vector])
    implementation = {fn.__name__: hashlib.sha256(marshal.dumps(inspect.unwrap(fn).__code__)).hexdigest() for fn in functions}
    from services.dataset_pruning_adapter import salad_inference_identity
    head_producer = salad_inference_identity(api, scope='bbox_crop', encoder_backend=runtime.get('resolved_backend')) if component.get('head_id') else None
    return {'schema': 'pruning-raw-component-v1', 'scope': 'bbox_crop', 'id': component['id'], 'encoder': base,
            'model': runtime['status']['model_id'] if base == 'sam3' else component['model'],
            'pooling': 'mask_multiscale' if base == 'sam3' else component['pooling'], 'head_id': component.get('head_id', ''),
            'head_sha256': component.get('head_sha256', ''), 'preprocessing': preprocessing if base != 'sam3' else {'policy': 'full_image_bbox_prompt'},
            'producer': runtime, 'head_producer': head_producer, 'implementation': implementation, 'cache_version': api.CLASS_ANALYSIS_CACHE_VERSION}


def build_recipes(adapter, request, source):
    api = adapter.api
    cfg = copy.deepcopy(adapter.analysis_config(request.get('analysis_job_id', '')))
    config = request.get('config') or {}
    mode = config.get('evidence_mode', 'compound')
    if mode not in {'compound', 'objects_only', 'scene_only'}:
        raise HTTPException(400, 'Unknown evidence mode')
    resolved, inherited = inherited_components(api, cfg)
    variant = config.get('object_recipe', 'inherit_dqe')
    if variant == 'demo_raw_v1': variant = 'demo_raw_dino_sam'
    if variant not in {'inherit_dqe', 'demo_raw_dino_sam', 'custom', 'fast_map_v1', 'precise_compact_v1', 'thorough_quality_v1', 'pruning_thorough_v2'}:
        raise HTTPException(400, 'Unknown object feature recipe')
    if variant not in {'inherit_dqe', 'demo_raw_dino_sam', 'custom'}:
        cfg = api._normalize_class_analysis_request({**cfg, 'quality_recipe': 'thorough_quality_v1' if variant == 'pruning_thorough_v2' else variant})
        resolved, inherited = inherited_components(api, cfg)
    components = [] if mode == 'scene_only' else copy.deepcopy(config.get('object_components') or inherited)
    if variant == 'pruning_thorough_v2':
        # Reuse the shared Thorough producers and weights. Its historical
        # raw-C quality recipe remains unchanged; this is a pruning candidate.
        for component in components:
            component['adjustment'] = 'remove_size_bias'
    if variant == 'demo_raw_dino_sam' and mode != 'scene_only':
        components = [item for item in inherited if item['id'] in {'dinov3', 'sam3'}]
        for item in components: item['weight'] = .75 if item['id'] == 'dinov3' else .25; item['adjustment'] = 'none'
    preprocessing = {'crop_mode': cfg.get('crop_mode', 'padded_square'), 'padding_ratio': float(cfg.get('padding_ratio', .08)),
                     'preprocess_mode': cfg.get('preprocess_mode', 'canonical'), 'canonical_size': int(cfg.get('canonical_size', 336)),
                     'background_mode': cfg.get('background_mode', 'full_crop'), 'embedding_view_mode': cfg.get('embedding_view_mode', 'single')}
    if mode != 'scene_only':
        preprocessing.update({key: value for key, value in (config.get('object_preprocessing') or {}).items() if key in PREPROCESS_FIELDS})
    preprocessing['padding_ratio'] = float(preprocessing['padding_ratio'])
    preprocessing['canonical_size'] = int(preprocessing['canonical_size'])
    if not 0 <= preprocessing['padding_ratio'] <= 1 or not 64 <= preprocessing['canonical_size'] <= 1024:
        raise HTTPException(400, 'Invalid object preprocessing dimensions')
    if preprocessing['crop_mode'] not in {'tight', 'padded', 'padded_square'} or preprocessing['preprocess_mode'] not in {'native', 'canonical'} or preprocessing['embedding_view_mode'] not in {'single', 'tight_standard', 'standard_context', 'tight_context'} or preprocessing['background_mode'] not in {'full_crop', 'mean_fill_outside_box', 'blur_outside_box', 'darken_outside_box'}:
        raise HTTPException(400, 'Unknown object view/preparation policy')
    if config.get('object_adjustment', 'inherit') not in {'inherit', 'raw', 'remove', 'remove_all'}:
        raise HTTPException(400, 'Unknown object size adjustment policy')
    total = 0
    enabled = []
    heads = {head['id']: head for head in api._list_local_salad_heads() if head.get('status') == 'available'}
    for component in components:
        if component.get('enabled', True) is False: continue
        identifier = component.get('id')
        if identifier not in {'dinov3', 'sam3', 'salad', 'cradio'}: raise HTTPException(400, 'Unsupported object component')
        if any(item['id'] == identifier for item in enabled): raise HTTPException(400, 'An object component cannot be repeated')
        original = next((item for item in inherited if item['id'] == identifier), {})
        component = {**original, **component}
        component['encoder'] = 'dinov3' if identifier == 'salad' else identifier
        component['model'] = component.get('model') or (api.CRADIO_DEFAULT_MODEL if identifier == 'cradio' else cfg.get('encoder_model') or api.CLASS_ANALYSIS_DEFAULT_DINOV3_MODEL)
        component['pooling'] = component.get('pooling') or ('summary' if identifier == 'cradio' else 'pooler')
        if identifier == 'sam3':
            from services.mlx_sam3 import mlx_sam3_status
            component['model'] = mlx_sam3_status('mlx-bf16')['model_id']
            component['pooling'] = 'mask_multiscale'
        if identifier in {'dinov3', 'cradio'}:
            if component['pooling'] not in ({'summary', 'spatial_mean', 'summary_spatial_concat'} if identifier == 'cradio' else {'pooler', 'cls', 'patch_mean', 'cls_patch_concat'}):
                raise HTTPException(400, 'Unknown component pooling policy')
            dimensions = api._class_analysis_resolved_recipe({'quality_recipe': 'custom', 'encoder_type': identifier, 'encoder_model': component['model'],
                          'embedding_aggregation': 'pooled', 'dinov3_pooling': component['pooling'], 'cradio_pooling': component['pooling'],
                          'embedding_view_mode': preprocessing['embedding_view_mode']})
            component['dimension'] = cradio_component_dimension(component['model'], component['pooling'], preprocessing['embedding_view_mode']) if identifier == 'cradio' else dimensions['projection_features']['branches'][0]['dimensions']
        weight = component.get('weight', 1)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not np.isfinite(weight) or weight <= 0:
            raise HTTPException(400, 'Enabled component weights must be finite and positive')
        total += weight; component['weight'] = float(weight)
        adjustment = config.get('object_adjustment', 'inherit')
        if adjustment != 'inherit':
            component['adjustment'] = 'none' if adjustment == 'raw' else 'remove_size_bias' if adjustment == 'remove_all' or identifier != 'cradio' else 'none'
        component['adjustment'] = api._class_analysis_normalize_embedding_adjustment(component.get('adjustment', cfg.get('embedding_adjustment')))
        try:
            declared_strength = validate_strength(component.get('adjustment_strength'),
                default=0 if component['adjustment'] == 'none' else 1)
            component['adjustment_strength'] = 0 if component['adjustment'] == 'none' and adjustment != 'inherit' else declared_strength
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        if component['adjustment'] == 'none' and component['adjustment_strength'] != 0:
            raise HTTPException(400, 'Raw object components must use correction strength 0')
        if identifier == 'salad' and component.get('head_id'):
            meta = heads.get(component['head_id'])
            if not meta or meta.get('representation_scope') != 'bbox_crop' or meta.get('encoder_type', 'dinov3') != 'dinov3' or meta.get('encoder_model') != component['model'] or meta.get('archived'):
                raise HTTPException(400, 'Select an available bbox-crop SALAD head matching this backbone/model')
            component['head_sha256'] = meta['head_sha256']
            component['dimension'] = meta['descriptor_dim']
            component['training_fingerprint'] = (meta.get('training_signature') or {}).get('training_fingerprint', '')
            component['training_settings'] = (meta.get('training_signature') or {}).get('settings') or {}
            stored_view = (meta.get('training_signature') or {}).get('view_recipe') or {}
            if stored_view and any(stored_view.get(key) != preprocessing.get(key) for key in PREPROCESS_FIELDS):
                raise HTTPException(400, 'This crop head was trained with another view/preparation recipe; select a compatible version or train a new one')
        enabled.append(component)
    if mode != 'scene_only' and (not enabled or total <= 0): raise HTTPException(400, 'Enable at least one object component')
    for component in enabled: component['weight'] /= total
    scene_components = [] if mode == 'objects_only' else config.get('scene_components')
    if scene_components is None:
        legacy = adapter.legacy_recipes(request, source)['scene']
        scene_components = [{'encoder': legacy['encoder'], 'model': legacy['model'], 'head_id': legacy['head_id'], 'pooling': legacy['pooling'], 'weight': 1, 'adjustment': 'none'}]
    scenes = []
    total_scene_weight = 0
    for component in scene_components:
        if component.get('enabled', True) is False: continue
        component = copy.deepcopy(component)
        encoder = component.get('encoder', 'dinov3_pooled')
        if encoder not in {'dinov3_pooled', 'cradio_pooled', 'local_salad'}: raise HTTPException(400, 'SAM3 whole-image descriptors are not supported')
        if encoder == 'local_salad':
            meta = heads.get(component.get('head_id'))
            if not meta or meta.get('representation_scope', 'whole_image') != 'whole_image' or meta.get('archived'): raise HTTPException(400, 'Select an available whole-image SALAD profile')
            component['model'] = meta['encoder_model']
            if meta.get('encoder_type') == 'cradio':
                component['pooling'] = meta.get('cradio_pooling') or 'summary'
        weight = component.get('weight', 1)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not np.isfinite(weight) or weight <= 0: raise HTTPException(400, 'Scene component weights must be finite and positive')
        component['weight'] = float(weight); total_scene_weight += weight
        component['model'] = component.get('model') or (api.CRADIO_DEFAULT_MODEL if encoder == 'cradio_pooled' else api.CLASS_ANALYSIS_DEFAULT_DINOV3_MODEL)
        component['pooling'] = component.get('pooling', 'summary'); component['head_id'] = component.get('head_id', '')
        from services.dataset_pruning_adapter import scene_recipe
        component['raw_recipe'] = scene_recipe(api, encoder, component['model'], component['head_id'], component['pooling'], input_size=component.get('input_size'))
        component['pooling'] = component['raw_recipe']['pooling']
        component['adjustment'] = api._class_analysis_normalize_embedding_adjustment(component.get('adjustment', 'none'))
        try:
            component['adjustment_strength'] = validate_strength(component.get('adjustment_strength'),
                default=0 if component['adjustment'] == 'none' else 1)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        if component['adjustment'] == 'none' and component['adjustment_strength'] != 0:
            raise HTTPException(400, 'Raw whole-image components must use correction strength 0')
        scenes.append(component)
    if (mode != 'objects_only' and not scenes) or len(scenes) > 12: raise HTTPException(400, 'Select 1–12 whole-image components')
    for component in scenes: component['weight'] /= total_scene_weight
    demo_bound = (mode == 'compound' and config.get('scene_requirement', 'required') == 'required' and
                  variant == 'demo_raw_dino_sam' and len(scenes) == 1 and scenes[0]['encoder'] == 'local_salad' and
                  scenes[0].get('adjustment', 'none') in {'raw', 'none'} and
                  scenes[0]['raw_recipe']['head_sha256'] == 'be870ed5e5704fd00594721cb17700d240e1eb212507788a1ce6f94611819ac5' and
                  [item['id'] for item in enabled] == ['dinov3', 'sam3'] and all(item['adjustment'] == 'none' for item in enabled) and
                  preprocessing == {'crop_mode': 'padded_square', 'padding_ratio': 0.08, 'preprocess_mode': 'canonical', 'canonical_size': 336,
                                    'background_mode': 'full_crop', 'embedding_view_mode': 'tight_context'} and
                  enabled[0]['model'] == 'facebook/dinov3-vitb16-pretrain-lvd1689m' and
                  enabled[0]['weight'] == .75 and enabled[1]['weight'] == .25 and
                  config.get('scale_protection', 'native_pixels_and_occupancy') == 'occupancy_legacy' and
                  all(config.get(key) == value for key, value in {'scene_threshold': .92, 'negative_scene_threshold': .94, 'object_threshold': .80, 'index_threshold': .87,
                                                               'object_weight': .90, 'rarity_scale_threshold': .60, 'neighbor_k': 32}.items()))
    return {'schema': 'pruning-component-recipe-v1', 'source_identity': source['identity'], 'evidence_mode': mode,
            'scene': {'components': scenes, 'enabled': mode != 'objects_only'},
            'bbox': {'components': enabled, 'enabled': mode != 'scene_only', 'preprocessing': preprocessing, 'training_payload': cfg, 'variant': variant},
            'inherited_dqe_recipe': resolved, 'qualification': 'exact WALDO40 raw demo feature binding; gates still depend on the selected numeric recipe' if demo_bound else 'draft: this feature space requires pruning calibration; DQE recipe is inherited, gates are not validated',
            'size_note': 'Resizing does not remove learned size confounding. Each enabled component subtracts its declared share of the complete fitted size-and-shape prediction, then normalizes. Compatible saved fits are reused only for the same raw producer and accepted geometry population; otherwise the shared fitter records a new fit. Native bbox pixels, image dimensions and normalized occupancy remain separate coverage protection.',
            'baseline_policy': 'Historical DQE Thorough keeps C-RADIO raw; pruning_thorough_v2 corrects every component as a candidate requiring its own calibration.'}


def resolve_heads(adapter, images, recipes, directory, event, stats):
    api = adapter.api; bbox = recipes['bbox']
    if not bbox['components']:
        ordinal = 0
        for image in images:
            for obj in image['objects']:
                obj['feature_index'] = ordinal
                ordinal += 1
        return []
    records = object_records(api, images, bbox['preprocessing'])
    if not records:
        for component in bbox['components']:
            component['raw_recipe'] = raw_component_recipe(api, component, bbox['preprocessing'], bbox['training_payload'])
            component['status'] = 'No accepted labelled objects; bbox descriptor and crop-head training are unmeasured and unnecessary'
        return records
    for component in bbox['components']:
        if component['id'] != 'salad' or component.get('head_id'): continue
        cfg = {**bbox['training_payload'], **bbox['preprocessing'], 'encoder_model': component['model']}
        settings, train_indices, fingerprint = api._class_analysis_salad_training_plan(records, cfg)
        proxy = api.ClassAnalysisJob(job_id=directory.name, cancel_event=event)
        _, runtime = api._class_analysis_train_salad_head(records, cfg, settings=settings, train_indices=train_indices,
                      training_fingerprint=fingerprint, job=proxy, out_dir=directory / 'features' / 'salad-training')
        component['head_id'] = runtime['persistent_head_id']; component['head_sha256'] = runtime['persistent_head_sha256']
        component['dimension'] = settings.descriptor_dim
        component['training_fingerprint'] = fingerprint
        component['training_settings'] = {key: value for key, value in settings.to_dict().items() if key not in {'weight', 'dino_weight', 'sam_weight', 'salad_weight', 'descriptor_dim'}}
        stats['heads_trained'] += int(not runtime.get('reused_existing_head'))
        management = api._get_dataset_salad_profiles()
        for attempt in range(3):
            snapshot = management.snapshot(recipes['source_identity'])
            try:
                management.mutate({'request_id': digest({'source': recipes['source_identity'], 'head': component['head_sha256'], 'action': 'attach'})[:32],
                                   'source_identity': recipes['source_identity'], 'expected_revision': snapshot['revision'],
                                   'action': 'attach', 'head_id': component['head_id']}, api._list_local_salad_heads())
                break
            except HTTPException as error:
                if error.status_code != 409 or attempt == 2: raise
    for component in bbox['components']:
        component['raw_recipe'] = raw_component_recipe(api, component, bbox['preprocessing'], bbox['training_payload'])
    return records


def check_inputs(images):
    if any(file_state(Path(image['image_path'])) != image['image_state'] for image in images):
        raise HTTPException(409, 'Source image bytes changed during feature production')


def component_size_diagnostics(api, raw, adjusted, records, name, transform, *, max_rows=512):
    """Paired deterministic bounded PCA/covariate audit, separate from the fit."""
    result = {'component': name, 'policy': transform.get('mode'), 'applied': transform.get('applied', False),
              'correction_strength': transform.get('correction_strength', 1 if transform.get('applied') else 0),
              'base_fit_reused': transform.get('base_fit_reused', False),
              'reason': transform.get('skipped_reason'), 'fit_count': transform.get('point_count', 0),
              'fit_population': 'saved compatible geometry fit' if transform.get('base_fit_reused') else 'all current accepted records in this representation',
              'fit_population_count': (transform.get('fitted_transform') or {}).get('fit_population_count', transform.get('point_count', 0)),
              'diagnostic_policy': 'paired deterministic sample; top3 principal components versus original size covariates; no causal or accuracy claim'}
    if len(records) < 6:
        return {**result, 'diagnostic_status': 'unmeasured_not_enough_records'}
    indices = np.unique(np.linspace(0, len(records)-1, min(max_rows, len(records)), dtype=np.int64))
    covariates, names = api._class_analysis_record_covariates([records[int(index)] for index in indices])
    result.update(sample_count=len(indices), sample_row_indices=indices.tolist(), covariates=names)
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits
    for label, matrix in [('raw', raw), ('adjusted', adjusted)]:
        sample = normalize_rows(np.asarray(matrix[indices], np.float32))
        rank = min(3, len(sample)-1, sample.shape[1])
        if rank <= 0 or not np.any(sample.std(axis=0) > 1e-8):
            result[label + '_size_association'] = None
            continue
        with threadpool_limits(limits=2):
            projection = PCA(n_components=rank, svd_solver='randomized', random_state=42, iterated_power=2).fit_transform(sample)
        associations = [{'pc': pc+1, 'covariate': covariate, 'absolute_pearson': abs(api._class_analysis_pearson(projection[:, pc], covariates[:, column]))}
                        for pc in range(rank) for column, covariate in enumerate(names)]
        result[label + '_size_association'] = max((item['absolute_pearson'] for item in associations), default=None)
        result[label + '_associations'] = associations
    return {**result, 'diagnostic_status': 'measured_paired_sample'}


def fuse_to_memmap(api, matrices, components, path):
    if not matrices: return np.empty((0, 0), np.float32)
    rows = len(matrices[0]); dimensions = sum(matrix.shape[1] for matrix in matrices)
    output = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(rows, dimensions))
    for start in range(0, rows, 512):
        output[start:start + 512] = api.fuse_class_analysis_feature_branches([matrix[start:start + 512] for matrix in matrices], [component['weight'] for component in components])
    output.flush()
    return output


def component_features(adapter, images, recipes, cache, directory, event, stats, progress, records):
    api = adapter.api; root = directory / 'features'; root.mkdir(parents=True, exist_ok=True)
    stats.setdefault('components', {})
    manifest = {'schema': 'pruning-component-production-v1', 'source_images': len(images), 'object_rows': len(records), 'components': {},
                'evidence_mode': recipes.get('evidence_mode', 'compound'),
                'measured_branches': {'scene': bool(recipes['scene']['components']), 'bbox': bool(recipes['bbox']['components'])},
                'row_digest': digest([(record['_image_sha256'], record['geometry_key']) for record in records]),
                'input_policy': recipes['bbox']['preprocessing'], 'normalization': 'independent_unit_rows_sqrt_weight_concat_unit_rows'}
    dependencies = {}
    def remember_dependencies(component, keys):
        for reference in cache.reference_dependencies(keys): dependencies[reference['path']] = reference
        if component.get('head_id'):
            head_path = api._local_salad_head_path(component['head_id'])
            expected = component.get('head_sha256') or component['raw_recipe'].get('head_sha256')
            if not expected or api._class_analysis_file_sha256_cached(head_path) != expected:
                raise HTTPException(409, 'A selected SALAD head changed before component publication')
            dependencies[str(head_path.resolve())] = {'path': str(head_path.resolve()), 'state': file_state(head_path),
                                                     'sha256': expected, 'role': 'salad_head', 'required': True}
    scenes = []
    for ordinal, component in enumerate(recipes['scene']['components']):
        recipe = component['raw_recipe']
        from services.dataset_pruning_adapter import whole_source_recipe
        keys = [cache.key(image['image_sha256'], whole_source_recipe(recipe, image['width'], image['height'])) for image in images]
        unique_missing = list({key: i for i, key in enumerate(keys) if cache.get(key) is None}.values())
        stats.setdefault('components', {})['scene_' + str(ordinal)] = {'reused': len(images)-len(unique_missing), 'encoded': 0}
        batches = defaultdict(list)
        for index in unique_missing:
            # The existing C-RADIO producer preserves same-size inputs, but
            # otherwise resizes to512. Equal-source-size grouping fixes that
            # batch-dependent ambiguity without altering product preprocessing.
            group = (images[index]['width'], images[index]['height']) if recipe['base_encoder'] == 'cradio' else (0, 0)
            batches[group].append(index)
        proxy = api.DataIngestionJob(job_id=directory.name, cancel_event=event)
        for ids in batches.values():
            for start in range(0, len(ids), 16):
                batch = ids[start:start + 16]
                check_inputs([images[index] for index in batch])
                prepared = [{**images[index]['media_row'], 'image_path': images[index]['image_path'], 'source_type': 'image'} for index in batch]
                values = api._data_ingestion_encode_prepared_images(prepared, job=proxy, encoder=component['encoder'], model_name=component['model'],
                         salad_head_id=component['head_id'], cradio_pooling=component['pooling'], batch_size=16,
                         cradio_input_size=component.get('input_size'), dinov3_pooling=component['pooling'] if recipe['base_encoder'] == 'dinov3' else 'pooler')
                check_inputs([images[index] for index in batch])
                cache.put_many([(keys[index], vector) for index, vector in zip(batch, values)])
                stats['components']['scene_' + str(ordinal)]['encoded'] += len(batch)
                progress(.15 + .2*(ordinal+1)/len(recipes['scene']['components']), 'Whole-image component: cached unchanged images; encoding missing descriptors')
        first = cache.get(keys[0])
        matrix = np.lib.format.open_memmap(root / f'scene-{ordinal}-raw.npy', mode='w+', dtype=np.float32, shape=(len(keys), len(first)))
        for index, key in enumerate(keys): matrix[index] = cache.get(key)
        matrix.flush()
        component['dimension'] = matrix.shape[1]
        remember_dependencies(component, keys)
        scene_records = [{'_image_width': image['width'], '_image_height': image['height'], 'width': image['width'], 'height': image['height'],
                          'bbox_xyxy': [0, 0, image['width'], image['height']], '_image_path': image['image_path']} for image in images]
        mode = component.get('adjustment', 'none')
        adjusted, transform = adjust_component_to_memmap(api, matrix, scene_records, mode=mode,
            strength=component.get('adjustment_strength'), output_path=root / f'scene-{ordinal}-adjusted.npy',
            fit_cache_root=directory.parent / 'cache' / 'corrections', raw_recipe=recipe,
            base_transform=recipes.get('saved_component_fits', {}).get('scene_' + str(ordinal)))
        manifest.setdefault('size_diagnostics', []).append(component_size_diagnostics(api, matrix, adjusted, scene_records, 'whole_image_' + str(ordinal), transform))
        scenes.append(adjusted)
        manifest['components']['scene_' + str(ordinal)] = {'raw_recipe': recipe, 'dimension': matrix.shape[1], 'weight': component['weight'],
                                                        'adjustment': transform, 'input_dimensions': 'per-image source dimensions bound in raw cache keys',
                                                        'encoder_input': recipe['producer'].get('input_policy') or recipe['producer'].get('input_size')}
    stats['scene_reused'] = min((value['reused'] for key, value in stats['components'].items() if key.startswith('scene_')), default=0)
    stats['scene_encoded'] = sum(value['encoded'] for key, value in stats['components'].items() if key.startswith('scene_'))
    scene_matrix = fuse_to_memmap(api, scenes, recipes['scene']['components'], root / 'scenes.npy') if scenes else np.empty((len(images), 0), np.float32)
    object_output = None
    object_offset = 0
    completed_components = 0
    expected_dimensions = sum(component.get('dimension', 0) for component in recipes['bbox']['components'])
    for component in recipes['bbox']['components']:
        identifier = component['id']; recipe = component['raw_recipe']
        keys = [cache.key(record['_image_sha256'], recipe, record['geometry_key']) for record in records]
        if not keys: continue
        # Import only a independently addressable, producer-bound slice. The
        # historical unpersisted crop SALAD slice is intentionally excluded.
        if identifier in {'dinov3', 'sam3'}:
            adapter.import_component_dqe(images, component, recipes['bbox']['preprocessing'], cache, read_request(directory)['analysis_job_id'])
        missing = list({key: index for index, key in enumerate(keys) if cache.get(key) is None}.values())
        info = {'reused': len(keys)-len(missing), 'encoded': 0, 'producer_cache_hits': 0}
        stats['components']['bbox_' + identifier] = info
        branch_root = root / identifier; branch_root.mkdir(exist_ok=True)
        proxy = api.ClassAnalysisJob(job_id=directory.name, cancel_event=event)
        if missing:
            selected = [records[index] for index in missing]
            check_inputs(images)
            if identifier == 'sam3':
                encoded, runtime = api._class_analysis_extract_sam3_mask_embeddings(selected, job=proxy, output_path=branch_root / 'raw.npy')
                info['mask_evidence'] = runtime
            else:
                head = {'encoder_type': component['encoder'], 'encoder_model': component['model'], 'normalize_embeddings': True,
                        'embedding_aggregation': 'pooled', 'dinov3_pooling': component['pooling'], 'cradio_pooling': component['pooling'],
                        'cradio_output_mode': 'summary_only' if component['pooling'] == 'summary' else 'summary_and_spatial'}
                head['qualified_producer_identity'] = recipe['producer']
                if identifier == 'salad':
                    loaded, meta = api._load_local_salad_head(component['head_id'], device_name='cpu')
                    if meta.get('representation_scope') != 'bbox_crop': raise HTTPException(400, 'Crop component requires a bbox-trained head')
                    head.update(embedding_aggregation=api.CLASS_ANALYSIS_SAM3_SALAD_FUSION_SCHEMA,
                                embedding_salad_training_fingerprint=component.get('training_fingerprint') or digest(meta.get('training_signature', {})),
                                _class_analysis_salad_head=loaded)
                    saved_settings = component.get('training_settings') or {}
                    head.update(api.class_analysis_salad_settings({**recipes['bbox']['training_payload'], **{'salad_' + key: value for key, value in saved_settings.items()}}).to_request_fields())
                details = {}
                encoded = api._class_analysis_stream_encode_records(selected, {**recipes['bbox']['training_payload'], **recipes['bbox']['preprocessing']},
                          job=proxy, head=head, batch_size=int(api._class_analysis_feature_batch_plan(recipes['bbox']['training_payload'], head)['batch_size']),
                          out_dir=branch_root, cache_stats=details)
                info['producer_cache_hits'] = details.get('hits', 0)
                if identifier == 'salad': encoded = encoded[:, -component['dimension']:]
            check_inputs(images)
            for start in range(0, len(missing), 128):
                cache.put_many([(keys[index], vector) for index, vector in zip(missing[start:start+128], encoded[start:start+128])])
            info['encoded'] = len(missing)-info['producer_cache_hits']
        first = cache.get(keys[0])
        matrix = np.lib.format.open_memmap(branch_root / 'assembled-raw.npy', mode='w+', dtype=np.float32, shape=(len(keys), len(first)))
        for index, key in enumerate(keys):
            vector = cache.get(key)
            if vector is None or len(vector) != matrix.shape[1]: raise ValueError('A raw component is missing or incompatible')
            matrix[index] = vector
        matrix.flush()
        component['dimension'] = matrix.shape[1]
        remember_dependencies(component, keys)
        if object_output is None:
            if expected_dimensions <= 0: raise ValueError('Resolve component dimensions before assembly')
            object_output = np.lib.format.open_memmap(root / 'objects.npy', mode='w+', dtype=np.float32, shape=(len(records), expected_dimensions))
        if object_offset + matrix.shape[1] > object_output.shape[1]: raise ValueError('Object component width changed; resolve the recipe again')
        target_slice = object_output[:, object_offset:object_offset + matrix.shape[1]]
        adjusted, transform = adjust_component_to_memmap(api, matrix, records, mode=component['adjustment'],
            strength=component.get('adjustment_strength'), output_path=branch_root / 'adjusted.npy', output_array=target_slice,
            fit_cache_root=directory.parent / 'cache' / 'corrections', raw_recipe=recipe,
            base_transform=recipes.get('saved_component_fits', {}).get('bbox_' + identifier))
        manifest.setdefault('size_diagnostics', []).append(component_size_diagnostics(api, matrix, adjusted, records, 'bbox_' + identifier, transform))
        for start in range(0, len(records), 512): target_slice[start:start+512] *= np.sqrt(component['weight'])
        object_offset += matrix.shape[1]
        completed_components += 1
        manifest['components']['bbox_' + identifier] = {'raw_recipe': recipe, 'dimension': matrix.shape[1], 'weight': component['weight'], 'adjustment': transform,
                                                     'preparation': recipes['bbox']['preprocessing'], 'encoder_input': recipe['producer'].get('input_policy') or recipe['producer'].get('input_size'),
                                                     'head_id': component.get('head_id', ''), 'head_sha256': component.get('head_sha256', ''), 'cache': info}
        matrix_path = Path(matrix.filename)
        del adjusted, target_slice
        if getattr(matrix, '_mmap', None) is not None: matrix._mmap.close()
        matrix_path.unlink(missing_ok=True)
        # The producer work matrix is transient; raw cache rows survive.
        if missing and isinstance(encoded, np.memmap):
            producer_path = Path(encoded.filename)
            encoded._mmap.close()
            producer_path.unlink(missing_ok=True)
        progress(.4 + .3*completed_components/len(recipes['bbox']['components']), f'Object component {identifier}: {info["reused"]} reused; {info["encoded"]} encoded')
    if object_output is not None:
        if object_offset != object_output.shape[1]: raise ValueError('Object feature component dimensions do not match their declared recipe')
        for start in range(0, len(records), 512):
            chunk = object_output[start:start+512]
            chunk /= np.maximum(np.linalg.norm(chunk, axis=1, keepdims=True), 1e-12)
        object_output.flush()
    object_matrix = object_output if object_output is not None else np.empty((0, 0), np.float32)
    stats['bbox_reused'] = min((value['reused'] + value['producer_cache_hits'] for key, value in stats['components'].items() if key.startswith('bbox_')), default=0)
    stats['bbox_encoded'] = sum(value['encoded'] for key, value in stats['components'].items() if key.startswith('bbox_'))
    recipes['component_manifest'] = manifest
    recipes['size_diagnostics'] = manifest.get('size_diagnostics', [])
    (directory / 'components.json').write_text(json.dumps(manifest, sort_keys=True, allow_nan=False))
    dependency_manifest = {'schema': 'pruning-artifact-dependencies-v1', 'dependencies': list(dependencies.values()),
                           'optional_caches': [{'path': str(cache.path.parent.resolve()), 'role': 'pruning_raw_and_derived_cache'},
                                               {'path': str(api.CLASS_ANALYSIS_CACHE_ROOT.resolve()), 'role': 'shared_DQE_producer_cache'}],
                           'review_materialization': ['scene.npy', 'evidence.sqlite3', 'images.json', 'result.json', 'review.sqlite3'],
                           'optional_cache_note': 'Saved review uses durable scene and pair evidence; optional caches accelerate future feature production.'}
    (directory / 'dependencies.json').write_text(json.dumps(dependency_manifest, sort_keys=True, allow_nan=False))
    return scene_matrix, object_matrix


def read_request(directory):
    return json.loads((directory / 'request.json').read_text())


def produce_consistent_crop_salad(api, records, payload, out_dir, event, *, train_indices=None):
    """Controlled report/production entrypoint using the qualified shared trainer.

    The old anonymous crop head cannot encode changed boxes. This creates or
    resolves one persisted compatible version, then describes every supplied
    current record in that single640D (or selected preset) space. It performs no
    source annotation writes and never imports the old anonymous SALAD slice.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(payload)
    settings, selected, fingerprint = api._class_analysis_salad_training_plan(records, cfg)
    if train_indices is not None:
        selected = list(train_indices)
        if len(set(selected)) != len(selected) or any(type(index) is not int or not 0 <= index < len(records) for index in selected):
            raise ValueError('Invalid explicit training record indices')
        fingerprint = api.class_analysis_salad_training_fingerprint(records, selected, settings, seed=int(cfg.get('seed', 42)))
    proxy = api.ClassAnalysisJob(job_id='pruning-controlled-crop', cancel_event=event)
    head, runtime = api._class_analysis_train_salad_head(records, cfg, settings=settings, train_indices=selected,
                     training_fingerprint=fingerprint, job=proxy, out_dir=out_dir)
    head_spec = {'encoder_type': 'dinov3', 'encoder_model': cfg.get('encoder_model') or api.CLASS_ANALYSIS_DEFAULT_DINOV3_MODEL,
                 'normalize_embeddings': True, 'dinov3_pooling': cfg.get('dinov3_pooling', 'pooler'),
                 'cradio_pooling': api.normalize_cradio_pooling(cfg.get('cradio_pooling')),
                 'embedding_aggregation': api.CLASS_ANALYSIS_SAM3_SALAD_FUSION_SCHEMA,
                 'embedding_salad_head_id': str(cfg.get('embedding_salad_head_id') or ''),
                 'embedding_salad_training_fingerprint': fingerprint, '_class_analysis_salad_head': head,
                 **settings.to_request_fields()}
    stats = {}
    combined = api._class_analysis_stream_encode_records(records, cfg, job=proxy, head=head_spec,
               batch_size=int(api._class_analysis_feature_batch_plan(cfg, head_spec)['batch_size']), out_dir=out_dir, cache_stats=stats)
    raw_path = out_dir / 'crop-salad-raw.npy'
    raw = np.lib.format.open_memmap(raw_path, mode='w+', dtype=np.float32, shape=(len(records), settings.descriptor_dim))
    for start in range(0, len(records), 512):
        raw[start:start+512] = normalize_rows(combined[start:start+512, -settings.descriptor_dim:])
    raw.flush()
    receipt = {'schema': 'consistent-current-crop-salad-production-v1', 'representation_scope': 'bbox_crop',
               'head_id': runtime['persistent_head_id'], 'head_sha256': runtime['persistent_head_sha256'],
               'head_runtime': runtime, 'training_fingerprint': fingerprint, 'training_record_indices': selected,
               'training_settings': settings.to_dict(), 'raw_shape': list(raw.shape), 'raw_state': file_state(raw_path),
               'record_digest': api._class_analysis_embedding_record_digest(records),
               'row_content_digest': digest([(record.get('_image_sha256'), record.get('bbox_xyxy')) for record in records]),
               'policy': 'one persisted head for every current row; anonymous historical640D excluded', 'producer_cache': stats,
               'preparation': {key: cfg.get(key) for key in PREPROCESS_FIELDS}}
    (out_dir / 'crop-salad-production.json').write_text(json.dumps(receipt, sort_keys=True, allow_nan=False))
    # Release the redundant combined DINO+SALAD assembly after durable compact
    # crop output publication. Existing per-image producer caches stay reusable.
    if isinstance(combined, np.memmap):
        combined_path = Path(combined.filename)
        combined._mmap.close(); combined_path.unlink(missing_ok=True)
    return raw, receipt
