"""Thin adapters to Tator's existing source, encoder, neighbor and export code."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import threading
import inspect
import marshal
import time

import numpy as np
from fastapi import HTTPException

from services.dataset_pruning import digest, geometry_key
from services.dataset_pruning_cache import DescriptorCache, file_state
from services.data_ingestion import normalize_rows
from services.dataset_pruning_producer import read_summary


def content_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): value.update(chunk)
    return value.hexdigest()


def cradio_preprocessing_identity(api, runtime):
    """Pin the actual native producer, including checkpoint-owned conditioning.

    The MLX encoder consumes RGB pixels rescaled to [0,1]. It applies the
    checkpoint's input_conditioner inside forward; the optional CLIP utility
    in the same package is not this encoder's preprocessing path.
    """
    if runtime.get('resolved_backend') != 'mlx' or not runtime.get('available'):
        return None
    import importlib.metadata
    import sys
    from utils.cradio_embedding import _import_cradio_mlx
    encoders = _import_cradio_mlx()
    package_root = Path(inspect.getfile(encoders[1])).parent
    dependencies = sorted(package_root.rglob('*.py'))
    dependencies.append(Path(__file__).parents[1] / 'utils' / 'cradio_embedding.py')
    versions = {}
    for package in ['mlx', 'Pillow', 'numpy']:
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = 'unavailable'
    # Check already-loaded native code without making cold/hot imports change
    # the identity. Package source pins are present in either case.
    for module_name, module in list(sys.modules.items()):
        if module_name != 'cradio_mlx' and not module_name.startswith('cradio_mlx.'): continue
        module_path = Path(getattr(module, '__file__', '') or '')
        if module_path.suffix != '.py' or not module_path.is_file(): continue
        expected = {}
        def collect_code(code):
            expected[code.co_qualname] = code
            for child in code.co_consts:
                if inspect.iscode(child): collect_code(child)
        collect_code(compile(module_path.read_bytes(), str(module_path), 'exec', dont_inherit=True))
        for name, value in vars(module).items():
            functions = [(name, value)] if inspect.isfunction(value) else [(name+'.'+key, member) for key, member in vars(value).items() if inspect.isfunction(member)] if inspect.isclass(value) and getattr(value, '__module__', None) == module_name else []
            for label, function in functions:
                if getattr(function, '__module__', None) != module_name: continue
                code = inspect.unwrap(function).__code__
                if code.co_filename == str(module_path) and expected.get(code.co_qualname) != code:
                    raise HTTPException(409, 'C-RADIO implementation changed after loading ('+module_name+'.'+label+'). Restart this backend before producing reusable descriptors.')
    return {'schema': 'cradio-native-mlx-preprocessing-v1',
            'resize': 'BICUBIC', 'geometry': 'direct_stretch', 'rgb': 'uint8_RGB',
            'rescale': 'float32_RGB_divide_255',
            'normalization': {'policy': 'checkpoint_input_conditioner',
                              'mean_key': 'radio_model.input_conditioner.norm_mean',
                              'std_key': 'radio_model.input_conditioner.norm_std',
                              'formula': '(pixels - norm_mean) / norm_std',
                              'dtype': runtime.get('dtype'),
                              'checkpoint_sha256': runtime.get('checkpoint_sha256')},
            'library_versions': versions,
            'implementation_files': [{'path': str(path), 'sha256': api._class_analysis_file_sha256_cached(path)} for path in dependencies],
            'dependency_policy': 'Pinned native package and wrapper sources; already-loaded native code checked against those sources. Identity is independent of module import order.'}


def salad_inference_identity(api, *, scope, encoder_backend):
    """Reuse shared private-head identity; bind the actual inference consumer."""
    backend = api.resolve_local_salad_backend()
    implementation = api._local_salad_producer_implementation_identity()
    device = 'mlx' if backend == 'mlx' else 'cpu' if scope == 'bbox_crop' else api.resolve_cradio_aux_torch_device() if encoder_backend == 'mlx' else encoder_backend
    return {'schema': 'pruning-salad-inference-v1', 'backend': backend, 'device': device,
            'numerical_implementation': {key: implementation[key] for key in ['schema', 'files', 'loaded_code'] if key in implementation},
            'consumer_code': hashlib.sha256(marshal.dumps(inspect.unwrap(api._encode_local_salad_head_np).__code__)).hexdigest()}


def scene_recipe(api, encoder, model, head_id="", pooling="summary", *, input_size=None):
    base = "cradio" if encoder=="cradio_pooled" else "dinov3"
    if head_id:
        _,meta = api._load_local_salad_head(head_id,device_name="cpu")
        if meta.get("representation_scope", "whole_image") != "whole_image":
            raise HTTPException(400, "A bbox-trained SALAD head cannot describe whole images")
        base = meta.get("encoder_type") or "dinov3"
    if base == 'dinov3' and pooling == 'summary': pooling = 'pooler'
    if base == 'dinov3' and input_size is not None:
        raise HTTPException(400, 'An explicit whole-image grid is supported only for C-RADIO')
    if input_size is not None and (type(input_size) is not int or not 64 <= input_size <= 2048 or input_size % 16):
        raise HTTPException(400, 'C-RADIO whole-image grid must be a multiple of16 from64 to2048')
    runtime = api._class_analysis_embedding_head_cache_identity({"encoder_type":base,"encoder_model":model,"cradio_pooling":pooling})
    functions = [api._data_ingestion_encode_prepared_images, api._encode_pil_batch_for_head]
    producer = api._local_salad_encoder_identity(base, model, backend='auto' if base == 'cradio' else '', pooling=pooling)
    if base == 'cradio':
        from utils.cradio_embedding import _resolve_cradio_mlx_image_size, _coerce_mlx_size_dimension, _encode_cradio_images_mlx
        functions.extend([_resolve_cradio_mlx_image_size, _coerce_mlx_size_dimension, _encode_cradio_images_mlx, api.encode_cradio_images])
        if input_size is not None: producer = {**producer, 'input_size': input_size, 'input_policy_override': 'explicit_per_call_square_grid'}
        preprocessing = cradio_preprocessing_identity(api, producer)
        if preprocessing is not None: producer = {**producer, 'preprocessing': preprocessing}
    head_producer = salad_inference_identity(api, scope='whole_image', encoder_backend=producer.get('resolved_backend')) if head_id else None
    return {"schema":"pruning-whole-component-v1", "scope": "whole_image", "encoder":encoder,"model":model,"pooling":pooling,
            "head_id":head_id,"head_sha256":content_hash(api._local_salad_head_path(head_id)) if head_id else "",
            "preprocess_version":api.LOCAL_SALAD_CACHE_VERSION,
            "runtime":runtime, 'base_encoder': base,
            'producer': producer, 'head_producer': head_producer, 'input_size': input_size,
            'input_batch_policy': 'equal_source_dimensions_preserve_grid' if base == 'cradio' else 'existing_ingestion_processor',
            'implementation': {fn.__name__: hashlib.sha256(marshal.dumps(inspect.unwrap(fn).__code__)).hexdigest() for fn in functions}}


def whole_source_recipe(recipe, width, height):
    """Bind final grid from the declared per-image producer policy, not batch peers."""
    producer = recipe.get('producer') or {}
    grid = (producer.get('input_policy') or {}).get('size')
    if recipe.get('base_encoder') == 'cradio' and (producer.get('resolved_backend') == 'mlx' or recipe.get('input_size') is not None):
        from utils.cradio_embedding import _coerce_mlx_size_dimension
        size = producer.get('input_size')
        grid = [_coerce_mlx_size_dimension(width), _coerce_mlx_size_dimension(height)] if size == 'preserve-input' else [int(size), int(size)] if type(size) is int else None
    return {**recipe, 'source_dimensions': [int(width), int(height)], 'resolved_encoder_grid': grid}


def publish_ingestion_descriptors(api, prepared, embeddings, encoder, model, head_id="", pooling="summary", *, expected_recipe=None):
    """Publish existing ingestion work, including candidates, for future reuse."""
    if embeddings is None or len(prepared)!=len(embeddings): return
    cache = DescriptorCache(api.CLASS_ANALYSIS_ROOT/"pruning"/"cache")
    recipe = scene_recipe(api,encoder,model,head_id,pooling)
    if expected_recipe is None or recipe != expected_recipe: return
    for row,feature in zip(prepared,embeddings):
        path = Path(row.get("image_path") or row.get("path"))
        # Encoding rows retain file size/mtime from before inference.
        st = path.stat()
        if st.st_size != int(row.get("size",0)) or st.st_mtime_ns != int(row.get("mtime_ns",0)):
            continue
        before = file_state(path)
        if row.get("pruning_file_state") != before: continue
        sha = content_hash(path)
        if file_state(path) != before: continue
        from PIL import Image
        with Image.open(path) as image: width, height = image.size
        row_recipe = whole_source_recipe(recipe, width, height)
        if recipe['base_encoder'] == 'cradio' and recipe['producer'].get('resolved_backend') == 'mlx':
            # Ingestion preserves its existing mixed-batch512 behavior. Such
            # vectors cannot populate pruning's independent native-grid key.
            if row.get('pruning_encoded_grid') != row_recipe['resolved_encoder_grid']: continue
        if file_state(path) != before: continue
        row["image_sha256"] = sha
        cache.put(cache.key(sha,row_recipe),feature)
    return recipe


class TatorPruningAdapter:
    def __init__(self, api): self.api = api

    @contextmanager
    def guard(self):
        library = self.api._get_storage_library_service().library
        with library.protect(owner="dataset_pruning",purpose="worker"):
            try: yield
            finally: library.invalidate()

    def capabilities(self, analysis_job_id=""):
        heads = self.api._list_local_salad_heads()
        from services.dataset_pruning_components import inherited_components
        cfg = self.analysis_config(analysis_job_id)
        resolved, components = inherited_components(self.api, cfg)
        inherited = {"origin": "active_dqe" if analysis_job_id else "shared_dqe_default_preset", "resolved": resolved,
                     "recipe_config": cfg,
                     "bbox": {"components": components, "preprocessing": {key: cfg.get(key) for key in
                        ["crop_mode", "padding_ratio", "preprocess_mode", "canonical_size", "background_mode", "embedding_view_mode"]}}}
        from services.dataset_pruning_components import build_recipes
        presets = []
        for identifier, label in [('inherit_dqe', 'Use current DQE recipe' if analysis_job_id else 'Use shared DQE default recipe'),
                                  ('fast_map_v1', 'Existing DQE Fast'), ('precise_compact_v1', 'Existing DQE Precise compact'),
                                  ('thorough_quality_v1', 'Historical DQE Thorough · raw C-RADIO'),
                                  ('pruning_thorough_v2', 'Pruning comparison · correct C-RADIO too'), ('demo_raw_dino_sam', 'Historical raw WALDO40 demo')]:
            recipe = build_recipes(self, {'analysis_job_id': analysis_job_id, 'config': {'object_recipe': identifier, 'evidence_mode': 'objects_only'}}, {'identity': 'capability-snapshot'})
            presets.append({'id': identifier, 'label': label, 'bbox': {key: recipe['bbox'][key] for key in ['components', 'preprocessing']},
                            'recipe_config': recipe['bbox']['training_payload'], 'resolved': recipe['inherited_dqe_recipe']})
        return {"version":3,"recipe_lab": {"correction_strength": [0, 1], "historical_object_recipe": "thorough_quality_v1", "candidate_object_recipe": "pruning_thorough_v2", "qualification": "Candidate for pruning comparison; DQE ranking default is unchanged. Calibrate each representation separately."},"heads":[h for h in heads if h.get("status")=="available"], 'object_recipes': presets,
                "default_config": {"scene_threshold": .92, "negative_scene_threshold": .94, "object_threshold": .80,
                                   "index_threshold": .87, "object_weight": .90, "rarity_scale_threshold": .60,
                                   "neighbor_k": 32, "scene_requirement": "required", "evidence_mode": "compound",
                                   "object_recipe": "inherit_dqe", "object_adjustment": "inherit", "scale_protection": "native_pixels_and_occupancy"},
                "presets": [{"id": "demo_raw_dino_sam", "label": "Historical raw WALDO40 demo", "qualification": "Measured only for its exact frozen raw feature recipe"},
                            {"id": "inherit_dqe", "label": "Complete active DQE recipe", "qualification": "Inherited features; pruning gates are an unqualified draft"}],
                "qualification": "Thresholds need calibration when descriptors, adjustment or weights change",
                "inherited_recipe": inherited,
                "scene_adjustments": ["raw", "remove"],
                "default_cradio_model": self.api.CRADIO_DEFAULT_MODEL,
                "default_dinov3_model": self.api.CLASS_ANALYSIS_DEFAULT_DINOV3_MODEL,
                "scene_models": {"dinov3": [self.api.CLASS_ANALYSIS_DEFAULT_DINOV3_MODEL], "cradio": list(self.api.CRADIO_SUPPORTED_MODELS)},
                "scene_default":"matching_local_salad_or_existing_dinov3",
                "coverage":"Estimated scene and class-balanced bbox feature coverage","source":"complete_dataset"}

    def validate_request(self,request):
        analysis_job_id = str(request.get('analysis_job_id') or '')
        if analysis_job_id:
            job = self.api._get_class_analysis_job(analysis_job_id)
            if job.status != "completed": raise HTTPException(409,"Finish or restore the selected DQE analysis first")
        source = request.get("annotation_source") or {}
        if source.get("kind") not in {"dataset","transient"} or not str(source.get("id") or ""):
            raise HTTPException(400,"Attach the dataset annotation workspace before pruning")
        if not isinstance(request.get("expected_revision"),int) or request["expected_revision"]<0:
            raise HTTPException(400,"Save the source annotation revision before pruning")
        if analysis_job_id: self.check_checkpoint(source,analysis_job_id)
        state = self.api._pruning_annotation_source_state(source["kind"],source["id"])
        if state["session_revision"] != request["expected_revision"]:
            raise HTTPException(409,"Annotations changed. Save and refresh before pruning")

    def analysis_config(self, analysis_job_id=""):
        if analysis_job_id: return dict(self.api._get_class_analysis_job(analysis_job_id).request)
        return self.api._normalize_class_analysis_request({'quality_recipe': 'thorough_quality_v1', 'analysis_scope': 'all_classes'})

    def check_checkpoint(self,source,analysis_job_id):
        self.api._annotation_export_require_class_analysis_checkpoint(
            {"class_analysis_job_id":analysis_job_id},
            source_mode="linked" if source["kind"]=="dataset" else "transient",
            source_id=source["id"])

    def annotation_states(self,kind,identifier,media,root,entry=None,session=None):
        """Bind both effective files and absent overrides before reading labels.

        Revision checks cover controlled mutations; these stat bindings cover
        external edits and captions that do not increment annotation revision.
        """
        api=self.api
        paths=set()
        if entry is not None:
            overlay=api._dataset_overlay_root_from_entry(entry,ensure=False)
            paths.add(overlay/"captions"/"captions.jsonl")
        else:
            root=api._transient_session_annotation_source_root(session,root)
            overlay=None
        layout=str((entry or session).get("yolo_layout") or "flat")
        paths.add(root/"labelmap.txt")
        for row in media:
            split,rel=row["split"],Path(row["image_relpath"])
            paths.add(api._annotation_source_label_path(root,layout,split,rel))
            if entry is not None:
                paths.add(api._annotation_overlay_label_path(entry,split,rel,ensure=False))
            else:
                token=hashlib.sha256(f"{split}/{rel.as_posix()}".encode()).hexdigest()
                paths.add(root/".workspace_captions"/(token+".json"))
            for source in [root]+([overlay] if overlay else []):
                for prefix in [source/"text_labels",source/split/"text_labels"] if source==root else [source/"text_labels",source/"text_labels"/split]:
                    paths.add(prefix/api._annotation_text_relpath(rel))
                    paths.add(prefix/api._annotation_legacy_text_name(rel))
        return {str(p):file_state(p) if p.is_file() else None for p in sorted(paths)}

    def source(self,request):
        api = self.api
        self.validate_request(request)
        handle = request["annotation_source"]
        kind,identifier = handle["kind"],handle["id"]
        entry=session=None
        if kind == "dataset":
            entry = api._resolve_dataset_entry(identifier)
            root = api._dataset_effective_root_from_entry(entry)
            media = api._data_ingestion_dataset_media_rows(identifier,field_name="reference_files")
        else:
            session = api._pruning_transient_session_snapshot(identifier)
            root = api._transient_session_dataset_root(session)
            media = api._data_ingestion_transient_session_media_rows(identifier,field_name="reference_files",_session_snapshot=session)
        states=self.annotation_states(kind,identifier,media,root,entry,session)
        manifest = api._transient_annotation_manifest_from_session(identifier,session,publish_progress=False) if session is not None else api._class_analysis_annotation_transaction_manifest(kind,identifier)
        records=api._export_stored_caption_records_only(identifier,entry)["captions"] if entry is not None else []
        captions={}
        for record in records: captions.setdefault(record["image_key"],[]).append(record)
        by_key = {(r["split"],r["image_relpath"]):r for r in media}
        rows = []
        for raw in manifest.get("images",[]):
            key = (raw["split"],raw["image_relpath"])
            row = by_key.get(key)
            if row is None: raise HTTPException(409,"Some source images are unavailable. Restore the complete source before pruning")
            image_key=f"{key[0]}/{key[1]}" if (entry or session).get("yolo_layout")=="split" else key[1]
            current = {**raw,"image_path":row["path"],"media_row":row,"text_label":str(raw.get("text_label") or ""),
                       "caption_records":captions.get(image_key,captions.get(f"{key[0]}/{key[1]}",[]))}
            if session is not None:
                overlay = session.get("overlay_labels") or {}
                source_root=api._transient_session_annotation_source_root(session,root)
                path = api._annotation_source_label_path(source_root,str(session.get("yolo_layout") or "flat"),raw["split"],Path(raw["image_relpath"]))
                current["label_source_present"] = f"{key[0]}:{key[1]}" in overlay or path.is_file()
                cap=source_root/".workspace_captions"/(hashlib.sha256(f"{key[0]}/{key[1]}".encode()).hexdigest()+".json")
                if cap.is_file(): current["caption_records"]=json.loads(cap.read_text())
            rows.append(current)
        if len(rows)!=len(media): raise HTTPException(409,"Source inventory and annotations changed during scanning")
        if states!=self.annotation_states(kind,identifier,media,root,entry,session):
            raise HTTPException(409,"Source annotations changed during scanning. Analyze again.")
        revision = int(manifest.get("session_revision",0))
        if revision != request["expected_revision"]: raise HTTPException(409,"Source annotation revision changed during scanning")
        return {"kind":kind,"id":identifier,"identity":f"{kind}:{identifier}","revision":revision,
                "dataset_root":str(root.resolve()),"registry_path":str(api._dataset_meta_storage_root_from_entry(entry)) if entry else "",
                "storage_mode":manifest.get("storage_mode","managed"),"annotation_states":states,
                "labelmap":list(manifest.get("labelmap") or []),"image_count":len(rows),"images":rows,
                "dataset_label":manifest.get("dataset_label",identifier),"analysis_job_id":request["analysis_job_id"]}

    def check_revision(self,source):
        if source.get('analysis_job_id'): self.check_checkpoint(source,source['analysis_job_id'])
        state = self.api._pruning_annotation_source_state(source["kind"],source["id"])
        if int(state["session_revision"])!=source["revision"] or list(state["labelmap"])!=source["labelmap"]:
            raise HTTPException(409,"Source annotations changed. Analyze again; cached image features will be reused")
        states=source.get("annotation_states")
        if states is None:
            path=source.get("annotation_states_path")
            if not path or not Path(path).is_file(): raise HTTPException(409,"This review needs a fresh analysis; cached descriptors will be reused")
            states=json.loads(Path(path).read_text())
        for raw,expected in states.items():
            path=Path(raw)
            actual=file_state(path) if path.is_file() else None
            if actual!=expected: raise HTTPException(409,"Source annotation or caption files changed. Analyze again; cached features will be reused")

    def check_inventory(self,source,images):
        if source["kind"]=="dataset":
            rows=self.api._data_ingestion_dataset_media_rows(source["id"],field_name="reference_files")
        else:
            snapshot = self.api._pruning_transient_session_snapshot(source['id'])
            rows=self.api._data_ingestion_transient_session_media_rows(source["id"],field_name="reference_files",_session_snapshot=snapshot)
        if {(r["split"],r["image_relpath"],str(Path(r["path"]).resolve())) for r in rows} != {(im["split"],im["image_relpath"],str(Path(im["image_path"]).resolve())) for im in images}:
            raise HTTPException(409,"Source image inventory changed. Analyze the complete dataset again; cached features will be reused")
        if any(not Path(im["image_path"]).is_file() or file_state(Path(im["image_path"]))!=im["image_state"] for im in images):
            raise HTTPException(409,"Source image bytes changed. Analyze again; unchanged features will be reused")

    def recipes(self, request, source):
        from services.dataset_pruning_components import build_recipes
        return build_recipes(self, request, source)

    def resolve_features(self, images, recipes, directory, event, stats):
        from services.dataset_pruning_components import resolve_heads
        return resolve_heads(self, images, recipes, directory, event, stats)

    def legacy_recipes(self,request,source):
        api = self.api
        cfg = self.analysis_config(request.get('analysis_job_id', ''))
        head_id = str(request.get("salad_head_id") or "").strip()
        heads = self.capabilities()["heads"]
        if not head_id or head_id=="auto":
            matches = [h for h in heads if h.get("representation_scope", "whole_image") == "whole_image" and not h.get("archived") and ((source["kind"]=="dataset" and h.get("reference_dataset_id")==source["id"])
                       or (source["kind"]=="transient" and h.get("reference_session_id")==source["id"]))]
            head_id = max(matches,key=lambda h:h.get("updated_at",0))["id"] if matches else ""
        if head_id == "dinov3": head_id=""
        head_meta = next((h for h in heads if h["id"]==head_id),None) if head_id else None
        if head_id and head_meta is None: raise HTTPException(400,"The selected SALAD head is unavailable")
        if head_meta and head_meta.get("representation_scope", "whole_image") != "whole_image":
            raise HTTPException(400, "Select a whole-image SALAD profile for the scene recipe")
        encoder = "local_salad" if head_id else "dinov3_pooled"
        model = head_meta["encoder_model"] if head_meta else api.CLASS_ANALYSIS_DEFAULT_DINOV3_MODEL
        pooling = head_meta.get("cradio_pooling") or "summary" if head_meta else "summary"
        scene = scene_recipe(api,encoder,model,head_id,pooling)
        # Remove the locally trained crop-SALAD branch from raw DQE fusion.
        # Its head is not persisted by older DQE jobs. DINO plus SAM remains
        # reproducible for changed crops and avoids retraining a crop head.
        aggregation = str(cfg.get("embedding_aggregation") or "")
        bbox = {"schema":"dqe-pruning-raw-bbox-v2","encoder_type":cfg.get("encoder_type","dinov3"),
                "encoder_model":cfg.get("encoder_model") or api.CLASS_ANALYSIS_DEFAULT_DINOV3_MODEL,
                "aggregation":"sam3_mask_fusion_v1" if aggregation in {"sam3_mask_fusion_v1","sam3_mask_salad_fusion_v1"} else "mean",
                **{k:cfg.get(k) for k in ["crop_mode","padding_ratio","preprocess_mode","canonical_size","background_mode","embedding_view_mode","dinov3_pooling","cradio_pooling"]}}
        bbox["producer_cache_version"]=api.CLASS_ANALYSIS_CACHE_VERSION
        bbox["cradio_output_mode"]="summary_only" if bbox["encoder_type"]=="cradio" and bbox.get("cradio_pooling")=="summary" else "summary_and_spatial"
        bbox["runtime"] = api._class_analysis_embedding_head_cache_identity({**bbox,"normalize_embeddings":True,"embedding_aggregation":"mean"})
        return {"scene":scene,"bbox":bbox,"scene_note":"Locally trained SALAD" if head_id else "Existing DINOv3 scene encoder; no matching locally trained SALAD head was found"}

    def producer_matches(self,job,directory):
        """Validate the persisted DQE recipe against the current producer.

        Reconstruct the existing head, including its historical crop SALAD
        training fingerprint, without retraining or reading every crop.
        """
        api=self.api;cfg=job.request
        try:
            manifest=json.loads((directory/"embeddings_manifest.json").read_text())
            summary=read_summary(directory/"result.json")
            head={"encoder_type":cfg.get("encoder_type","dinov3"),"encoder_model":cfg.get("encoder_model") or api.CLASS_ANALYSIS_DEFAULT_DINOV3_MODEL,
                  "normalize_embeddings":True,"dinov3_pooling":api._embedding_normalize_dinov3_pooling(cfg.get("dinov3_pooling")),
                  "cradio_pooling":api.normalize_cradio_pooling(cfg.get("cradio_pooling")),
                  "embedding_aggregation":api._embedding_normalize_aggregation(cfg.get("embedding_aggregation")),
                  "embedding_salad_head_id":str(cfg.get("embedding_salad_head_id") or "")}
            head["cradio_output_mode"]="summary_only" if head["encoder_type"]=="cradio" and head["cradio_pooling"]=="summary" else "summary_and_spatial"
            if head["embedding_aggregation"]=="sam3_mask_salad_fusion_v1":
                head.update(api.class_analysis_salad_settings(cfg).to_request_fields())
                head["embedding_salad_training_fingerprint"]=summary["embedding_adjustment_info"]["fusion"]["salad_training_fingerprint"]
            return manifest["recipe_digest"]==api._class_analysis_embedding_recipe_digest(cfg,head)
        except (OSError,KeyError,ValueError,TypeError): return False

    def import_dqe(self,images,recipe,cache,job_id, *, start=0, end=None):
        api = self.api
        job = api._get_class_analysis_job(job_id)
        directory = api._class_analysis_job_dir(job_id,create=False)
        if not self.producer_matches(job,directory): return
        raw_path = api._safe_existing_regular_file_within_root_impl(directory/"raw_embeddings.npy",directory)
        if raw_path is None: return
        raw_state=file_state(raw_path)
        manifest_path=directory/"embeddings_manifest.json"
        manifest=json.loads(manifest_path.read_text())
        frozen=manifest.get("raw_artifact_state")
        if frozen is not None:
            if raw_state!=frozen: return
        elif max(raw_state[3:])>manifest_path.stat().st_mtime_ns:
            # Older producers did not save raw-state attestation. Ordinary
            # post-publication edits (including preserved mtime) change ctime.
            # Copied legacy artifacts need fresh encoding rather than retagging.
            return
        matrix = np.load(raw_path,mmap_mode="r",allow_pickle=False)
        if matrix.ndim!=2 or matrix.dtype!=np.float32: return
        cfg = job.request
        if end is None:
            end = matrix.shape[1]
            if cfg.get("embedding_aggregation")=="sam3_mask_salad_fusion_v1":
                end -= api.class_analysis_salad_settings(cfg).descriptor_dim
        if not 0 <= start < end <= matrix.shape[1]: return
        # Verify source pixels against the analysis's frozen manifest, never
        # associate rows merely because their filenames happen to match.
        old_sha = {}
        if cfg.get("source_mode") in {"active_workspace","workspace"}:
            old_source = api._class_analysis_source(dict(cfg))
            old_sha = {(r["split"],r["image_relpath"]):r.get("image_sha256") for r in old_source["manifest"].get("images",[])}
        current_sha = {(im["split"],im["image_relpath"]):im["image_sha256"] for im in images}
        wanted = {(im["image_sha256"],o["geometry_key"]):cache.key(im["image_sha256"],recipe,o["geometry_key"]) for im in images for o in im["objects"]}
        entries, group_rows = [], []
        group_key = None
        def consume_group():
            if not group_rows: return
            sha = old_sha.get(group_key)
            if not sha:
                possible = current_sha.get(group_key)
                if possible and api._class_analysis_image_pack_group_key(possible,[r for _,r in group_rows]) == group_rows[0][1].get("image_pack_key"):
                    sha = possible
            if not sha: return
            for ordinal,row in group_rows:
                line = str(row.get("label_line") or "").split()
                if len(line)!=5: continue
                key = wanted.get((sha,geometry_key([float(v) for v in line[1:]])))
                if key is not None: entries.append((key,ordinal))
        for ordinal,row in api._class_analysis_iter_bound_metadata_rows(job):
            if ordinal >= len(matrix): raise ValueError("DQE feature/metadata row count mismatch")
            identity = (row.get("split"),row.get("image_relpath"))
            if group_key != identity:
                consume_group(); group_rows=[]; group_key=identity
            group_rows.append((ordinal,row))
            if len(entries)>=1024:
                cache.put_references(entries,raw_path,end,expected_state=raw_state,start=start); entries=[]
        consume_group()
        if entries: cache.put_references(entries,raw_path,end,expected_state=raw_state,start=start)
        if file_state(raw_path)!=raw_state: raise ValueError("DQE raw features changed during import")

    def import_reference(self,images,recipe,cache):
        api = self.api
        if recipe["encoder"] != "local_salad": return
        _,meta = api._load_local_salad_head(recipe["head_id"],device_name="cpu")
        rows = [{**im["media_row"],"image_path":im["image_path"]} for im in images]
        entry = api._data_ingestion_reference_embedding_cache_entry({"salad_head_id":recipe["head_id"]},rows,meta,
                   encoder_model=recipe["model"],cradio_pooling=recipe["pooling"])
        if not entry or entry["spec"].get("version")!=2 or entry["spec"].get("pruning_scene_recipe")!=recipe: return
        # Weak path/size/mtime-only legacy entries must never be retagged with
        # today's pixel SHA. V2 binds full file state and the producer recipe.
        features = api._data_ingestion_load_reference_embedding_cache(entry)
        if features is not None:
            for image,feature in zip(images,features): cache.put(cache.key(image["image_sha256"],recipe),feature)

    def import_candidates(self,images,recipe,cache):
        """Import saved ingestion rows only when pixel and recipe binding exists.

        Older candidate artifacts without these fingerprints require a first
        encoding. Filenames or similar-looking pixels never establish reuse.
        """
        missing = {im["image_sha256"] for im in images if cache.get(cache.key(im["image_sha256"],recipe)) is None}
        if not missing: return
        root = self.api._data_ingestion_storage_root(create=False)
        for directory in root.glob("di_*"):
            if not missing: break
            try:
                path = self.api._safe_existing_regular_file_within_root_impl(directory/"result.json",root)
                if path is None: continue
                result = json.loads(path.read_text())
                if result.get("summary",{}).get("pruning_scene_recipe") != recipe: continue
                items = [i for i in result.get("items",[]) if i.get("image_sha256") in missing]
                if not items: continue
                artifact = self.api._safe_existing_regular_file_within_root_impl(directory/"embeddings.npz",root)
                if artifact is None: continue
                with np.load(artifact,allow_pickle=False) as arrays:
                    features=arrays["candidate_embeddings"]
                    for item in items:
                        row = item["index"]
                        if not isinstance(row,int) or not 0<=row<len(features): continue
                        sha = item["image_sha256"]
                        cache.put(cache.key(sha,recipe),features[row]); missing.discard(sha)
            except (OSError,ValueError,KeyError,TypeError):
                continue

    def import_component_dqe(self, images, component, preprocessing, cache, job_id):
        if not job_id: return
        api = self.api
        cfg = api._get_class_analysis_job(job_id).request
        expected = {"crop_mode": cfg.get("crop_mode", "padded_square"), "padding_ratio": float(cfg.get("padding_ratio", .08)),
                    "preprocess_mode": cfg.get("preprocess_mode", "canonical"), "canonical_size": int(cfg.get("canonical_size", 336)),
                    "background_mode": cfg.get("background_mode", "full_crop"), "embedding_view_mode": cfg.get("embedding_view_mode", "single")}
        if expected != preprocessing: return
        if component['id'] != 'sam3' and component['model'] != (cfg.get('encoder_model') or api.CLASS_ANALYSIS_DEFAULT_DINOV3_MODEL): return
        directory = api._class_analysis_job_dir(job_id, create=False)
        path = directory / 'raw_embeddings.npy'
        if not path.is_file(): return
        matrix = np.load(path, mmap_mode='r', allow_pickle=False)
        salad = api.class_analysis_salad_settings(cfg).descriptor_dim if cfg.get('embedding_aggregation') == api.CLASS_ANALYSIS_SAM3_SALAD_FUSION_SCHEMA else 0
        sam = 768 if cfg.get('embedding_aggregation') in {api.CLASS_ANALYSIS_SAM3_SALAD_FUSION_SCHEMA, api.CLASS_ANALYSIS_SAM3_MASK_FUSION_SCHEMA} else 0
        dino = matrix.shape[1]-salad-sam
        if component['id'] == 'sam3':
            if not sam: return
            start, end = dino, dino+sam
        else:
            if component['pooling'] != cfg.get('dinov3_pooling', 'pooler'): return
            start, end = 0, dino
        self.import_dqe(images, component['raw_recipe'], cache, job_id, start=start, end=end)

    def features(self,images,recipes,cache,directory,event,stats,progress, records=None):
        from services.dataset_pruning_components import component_features, object_records
        self.cancel_event = event
        records = records if records is not None else object_records(self.api, images, recipes['bbox']['preprocessing'])
        return component_features(self, images, recipes, cache, directory, event, stats, progress, records)

    def legacy_features(self,images,recipes,cache,directory,event,stats,progress):
        api = self.api
        self.cancel_event = event
        scene_recipe_,bbox_recipe = recipes["scene"],{**recipes["bbox"], 'geometry_preparation': 'original_dqe_pixel_first_clipped_v1'}
        root = directory/"features"
        root.mkdir(exist_ok=True)
        scene_keys = [cache.key(im["image_sha256"],scene_recipe_) for im in images]
        if any(cache.get(key) is None for key in scene_keys): self.import_reference(images,scene_recipe_,cache)
        self.import_candidates(images,scene_recipe_,cache)
        self.import_dqe(images,bbox_recipe,cache,read_json_job(directory)["analysis_job_id"])
        missing = list({key:i for i,key in enumerate(scene_keys) if cache.get(key) is None}.values())
        stats["scene_reused"] = len(images)-len(missing)
        fake_job = api.DataIngestionJob(job_id=directory.name,cancel_event=event)
        for start in range(0,len(missing),16):
            ids = missing[start:start+16]
            prepared = [{**images[i]["media_row"],"image_path":images[i]["image_path"],"source_type":"image"} for i in ids]
            progress(.12+.3*(start+len(ids))/max(1,len(missing)),f"Encoding {start+len(ids)}/{len(missing)} missing scene descriptors")
            features = api._data_ingestion_encode_prepared_images(prepared,job=fake_job,encoder=scene_recipe_["encoder"],
                model_name=scene_recipe_["model"],salad_head_id=scene_recipe_["head_id"],cradio_pooling=scene_recipe_["pooling"],batch_size=16)
            if any(file_state(Path(images[i]["image_path"])) != images[i]["image_state"] for i in ids):
                raise ValueError("Source image changed while encoding scene descriptors")
            if scene_recipe(api,scene_recipe_["encoder"],scene_recipe_["model"],scene_recipe_["head_id"],scene_recipe_["pooling"]) != scene_recipe_:
                raise ValueError("Scene encoder or SALAD head changed during analysis")
            for i,feature in zip(ids,features): cache.put(scene_keys[i],feature)
            stats["scene_encoded"] += len(ids)
        scenes = normalize_rows(np.stack([cache.get(key) for key in scene_keys]))
        object_keys,records = [],[]
        for image in images:
            for obj in image["objects"]:
                obj["feature_index"] = len(records)
                w,h = image["width"],image["height"]
                geometry = api._class_analysis_parse_yolo_geometry(image['label_lines'][obj['label_row']], image_width=w, image_height=h)
                if not geometry or geometry['kind'] != 'bbox': raise HTTPException(409, 'Object geometry is not supported by the shared bbox producer')
                object_keys.append(cache.key(image["image_sha256"],bbox_recipe,obj["geometry_key"]))
                records.append({"_image_path":image["image_path"],"bbox_xyxy":geometry['bbox_xyxy'],
                                "image_name":image["image_relpath"],"class_id":obj["class_id"]})
        missing_objects = list({key:i for i,key in enumerate(object_keys) if cache.get(key) is None}.values())
        stats["bbox_reused"] = len(records)-len(missing_objects)
        head = {"encoder_type":bbox_recipe["encoder_type"],"encoder_model":bbox_recipe["encoder_model"],"normalize_embeddings":True,
                "embedding_aggregation":"mean", "cradio_output_mode":bbox_recipe["cradio_output_mode"], "dinov3_pooling":bbox_recipe.get("dinov3_pooling"),"cradio_pooling":bbox_recipe.get("cradio_pooling")}
        if missing_objects:
            proxy = api.ClassAnalysisJob(job_id=directory.name,cancel_event=event)
            missing_records = [records[i] for i in missing_objects]
            payload = {**bbox_recipe,"embedding_aggregation":"mean"}
            progress(.43,f"Encoding {len(missing_records)} missing bbox descriptions with the existing DQE pipeline")
            features = api._class_analysis_stream_encode_records(missing_records,payload,job=proxy,head=head,batch_size=32,out_dir=root)
            if bbox_recipe["aggregation"]=="sam3_mask_fusion_v1":
                sam,_ = api._class_analysis_extract_sam3_mask_embeddings(missing_records,job=proxy,output_path=root/"sam.npy")
            else: sam=None
            states = {im["image_path"]:im["image_state"] for im in images}
            if any(file_state(Path(r["_image_path"])) != states[r["_image_path"]] for r in missing_records):
                raise ValueError("Source image changed while encoding bbox descriptors")
            for start in range(0,len(missing_records),128):
                ids = missing_objects[start:start+128]
                values = features[start:start+128]
                if sam is not None: values=api.fuse_class_analysis_feature_branches([values,sam[start:start+128]],[.75,.25])
                for i,feature in zip(ids,values): cache.put(object_keys[i],feature)
            stats["bbox_encoded"] = len(missing_objects)
        if object_keys:
            first = cache.get(object_keys[0])
            objects = np.lib.format.open_memmap(root/"objects.npy",mode="w+",dtype=np.float32,shape=(len(object_keys),len(first)))
            for i,key in enumerate(object_keys):
                feature = cache.get(key)
                if feature is None or len(feature)!=objects.shape[1]: raise ValueError("Bbox descriptor is missing or incompatible")
                objects[i] = feature
            objects.flush()
        else: objects=np.empty((0,0),np.float32)
        return scenes,objects

    def neighbors(self,matrix,k):
        started = time.monotonic()
        if len(matrix)<2:
            self.last_neighbor_metadata = {'method': 'not_enough_rows', 'audit': {'sample_count': len(matrix), 'recall_at_k': None}}
            return np.empty((len(matrix),0),np.int64)
        # Reuse DQE's audited ANN and its existing bounded exact fallback.
        if len(matrix)>20000:
            job = self.api.ClassAnalysisJob(job_id="pruning-neighbors",cancel_event=getattr(self,"cancel_event",threading.Event()))
            budget = self.api.recommended_quality_memory_mb(self.api.available_quality_memory_mb())
            indices,_,metadata = self.api._class_analysis_compute_neighbors(matrix,neighbor_k=k,seed=42,job=job,
                                                                          memory_policy='budgeted', memory_budget_mb=budget)
            self.last_neighbor_metadata = {**metadata, 'memory_policy': 'budgeted', 'memory_budget_mb': budget}
            return indices
        indices = self.api._class_analysis_exact_neighbors_chunked(matrix,neighbor_k=k,chunk_size=128,cancel_event=getattr(self,"cancel_event",None))[0]
        self.last_neighbor_metadata = {'method': 'exact_chunked_cosine', 'chunk_size': 128,
                                       'elapsed_seconds': time.monotonic()-started,
                                       'audit': {'sample_count': len(matrix), 'recall_at_k': 1.}}
        return indices

    def group_neighbors(self, matrix, k, image_indices):
        job = self.api.ClassAnalysisJob(job_id='pruning-cross-image-neighbors', cancel_event=getattr(self, 'cancel_event', threading.Event()))
        budget = self.api.recommended_quality_memory_mb(self.api.available_quality_memory_mb())
        indices, _, metadata = self.api._class_analysis_compute_group_excluded_neighbors(
            matrix, image_indices, neighbor_k=k, seed=42, memory_budget_mb=budget, job=job)
        self.last_neighbor_metadata = metadata
        return indices

    def export_file_state(self,path): return self.api._annotation_export_file_state(path)
    def disk_capacity(self,directory,plan): return self.api._annotation_export_require_disk_capacity(plan,directory)
    def freeze_export(self,plan,directory,progress): return self.api._annotation_export_freeze_plan(plan,directory,progress)
    def write_export(self,plan,path,progress): return self.api._annotation_export_write_plan(plan,path,progress)
    def register_job(self,path):
        return self.api._get_storage_library_service().inventory.register_pruning(path)
    def register_export(self,path):
        return self.api._get_storage_library_service().inventory.register_pruning_export(path)


def read_json_job(directory):
    return json.loads((directory/"request.json").read_text())
