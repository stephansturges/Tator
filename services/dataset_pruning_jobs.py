"""Durable pruning jobs, revisioned selectors and frozen complete derivatives."""
from __future__ import annotations
import hashlib
import copy
import contextlib
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import threading
import time
import uuid
from typing import Any

import numpy as np
from fastapi import HTTPException
from PIL import Image
from starlette.responses import FileResponse

from services.dataset_pruning import PruningConfig, VERSION, SIMILARITY_RECIPE, build_evidence, build_order, digest, grouped_splits, parse_objects, removal_evidence, selection, coverage_for_retained, reevaluate_pair_evidence, pair_evidence
from services.dataset_pruning_cache import DescriptorCache, file_state
from services.dataset_pruning_evidence import PairStore
from services.dataset_pruning_review import ReviewLedger, FilteredEvidence, compute_plan, family_inventory, choose_family_keepers, directed_digest, family_edges, SupplementaryEvidence
from services.dataset_pruning_mask_evidence import collect_run_evidence, copy_sidecar, sam_recipe_digest, read_sidecar, object_mask_evidence
from services.dataset_pruning_serving import CompletedSnapshots, BoundedViews, ServingContext, states as artifact_states, verify as verify_artifacts, thaw
from services.dataset_pruning_planning import PlanningMixin, settings_for

DERIVED_SCHEMA = "pruning-derived-v2-candidate-admission"
ASSIGNMENT_SCHEMA = "labelled_assignments_v2_candidate_scope"
PRUNING_BLAS_LOCK = threading.RLock()


@contextlib.contextmanager
def pruning_blas_policy(cancelled=None):
    """Serialize pruning's process-wide BLAS limits and restore on every exit.

    This lock covers pruning matching and cached comparisons. Independent DQE
    callers that change the same process-wide library limits remain outside it.
    """
    from threadpoolctl import threadpool_info, threadpool_limits
    while not PRUNING_BLAS_LOCK.acquire(timeout=.1):
        if cancelled is not None and cancelled(): raise InterruptedError('Pruning calculation cancelled')
    try:
        if cancelled is not None and cancelled(): raise InterruptedError('Pruning calculation cancelled')
        execution = {'blas_threads': 2, 'before': threadpool_info(),
                     'scope': 'serialized pruning contexts; independent DQE contexts are not serialized'}
        try:
            with threadpool_limits(limits=2, user_api='blas'):
                execution['active'] = threadpool_info()
                yield execution
        finally:
            execution['restored'] = threadpool_info()
    finally:
        PRUNING_BLAS_LOCK.release()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(value,sort_keys=True,allow_nan=False))
        os.replace(temp,path)
    finally:
        temp.unlink(missing_ok=True)


class PruningDownloadResponse(FileResponse):
    def __init__(self, *args, adapter, validate, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter, self.validate = adapter, validate

    async def __call__(self, scope, receive, send):
        # Reuse storage admission through the complete existing FileResponse
        # stream, including disconnect/background handling.
        with self.adapter.guard():
            self.validate()
            await super().__call__(scope, receive, send)


class PruningService(PlanningMixin):
    def __init__(self, root: Path, adapter):
        self.root, self.adapter = root,adapter
        root.mkdir(parents=True,exist_ok=True)
        self.cache = DescriptorCache(root/"cache")
        self.lock = threading.RLock()
        self.feature_lock = threading.Lock()
        self.events = {}
        self.live = {}
        self.completed_snapshots = CompletedSnapshots()
        self.topology_views = BoundedViews(4)
        self.order_views = BoundedViews(4)
        self.plan_views = BoundedViews(4)
        self.family_views = BoundedViews(48)
        self.init_planning()

    def directory(self, job_id: str) -> Path:
        if not re.fullmatch(r"dp_[a-f0-9]{16}",str(job_id)):
            raise HTTPException(404,"Pruning analysis not found")
        directory = self.root/str(job_id)
        if directory.is_symlink() or not directory.is_dir():
            raise HTTPException(404,"Pruning analysis not found")
        return directory

    def update(self, identifier, **values):
        job_id = identifier
        with self.lock:
            directory = self.directory(job_id)
            state = self.live.get(job_id) or (read_json(directory/"state.json") if (directory/"state.json").exists() else {})
            state.update(values,updated_at=time.time())
            self.live[job_id] = state
            write_json(directory/"state.json",state)

    def status(self, job_id):
        with self.lock:
            directory = self.directory(job_id)
            if not (directory/"state.json").is_file(): raise HTTPException(404,"Saved pruning review was removed")
            state = dict(self.live.get(job_id) or read_json(directory/"state.json"))
            if state["status"] in {"queued","running"} and job_id not in self.events:
                state.update(status="interrupted",message="The backend restarted. Run the analysis again; cached features are retained.")
            return state

    def create(self, payload):
        token=payload.get("request_id")
        job_id="dp_"+(digest({"analysis_request":token})[:16] if token else uuid.uuid4().hex[:16])
        directory=self.root/job_id
        def existing():
            if directory.exists():
                if read_json(directory/"request.json")!=payload: raise HTTPException(409,"Request identity already belongs to another analysis")
                return self.status(job_id)
        with self.lock:
            found=existing()
            if found: return found
        self.adapter.validate_request(payload)
        with self.lock:
            found=existing()
            if found: return found
            staging=self.root/(".creating-"+uuid.uuid4().hex)
            staging.mkdir()
            try:
                write_json(staging/"request.json",payload)
                now=time.time()
                write_json(staging/"state.json",{"status":"queued","job_id":job_id,"progress":0.0,"message":"Preparing the complete dataset","created_at":now,"updated_at":now})
                os.replace(staging,directory)
            finally: shutil.rmtree(staging,ignore_errors=True)
            self.events[job_id]=threading.Event()
            threading.Thread(target=self.run,args=(job_id,),daemon=True,name=job_id).start()
            return self.status(job_id)

    @staticmethod
    def valid_derived(directory,images,similarity_recipe=SIMILARITY_RECIPE):
        try:
            commit=read_json(directory/"commit.json")
            if any(file_state(directory/name)!=state for name,state in commit["artifacts"].items()): return False
            required = {"result.json","images.json","scene.npy","evidence.sqlite3"}
            if not required <= set(commit['artifacts']) or set(commit['artifacts']) - required - {'components.json', 'dependencies.json'}: return False
            result=read_json(directory/"result.json")
            if result.get('recipe', {}).get('schema') == 'pruning-component-recipe-v1' and not {'components.json', 'dependencies.json'} <= set(commit['artifacts']): return False
            cached=read_json(directory/"images.json")
            if result.get("similarity_recipe")!=similarity_recipe or len(cached)!=len(images): return False
            if [im["image_id"] for im in cached]!=[im["image_id"] for im in images]: return False
            if len(result["curve"])!=result["max_removed_count"]+1 or [r["removed_count"] for r in result["curve"]]!=list(range(result["max_removed_count"]+1)): return False
            if sorted(result["order"])!=list(range(len(images))) or len(result["groups"])!=len(images): return False
            scene=np.load(directory/"scene.npy",mmap_mode="r",allow_pickle=False)
            if scene.ndim!=2 or len(scene)!=len(images) or scene.dtype!=np.float32: return False
            with sqlite3.connect(f"file:{directory/'evidence.sqlite3'}?mode=ro",uri=True) as db:
                if db.execute("PRAGMA quick_check").fetchone()[0]!="ok": return False
                row=db.execute("SELECT payload FROM pairs LIMIT 1").fetchone()
                if row: json.loads(row[0])
            return True
        except (OSError,ValueError,KeyError,TypeError,sqlite3.Error): return False

    def cancel(self, job_id):
        self.status(job_id)
        if job_id in self.events:
            self.events[job_id].set()
        return self.status(job_id)

    def reusable_assignments(self, assignment_key, images):
        for directory in (self.root / 'cache' / 'derived').glob('*'):
            if directory.is_symlink() or not (directory / 'result.json').is_file(): continue
            try:
                result = read_json(directory / 'result.json')
                if result.get('assignment_key') != assignment_key or not result.get('matcher_config'): continue
                if self.valid_derived(directory, images, result['similarity_recipe']): return directory, result
            except (OSError, ValueError, KeyError): continue
        return None

    def run(self, job_id):
        directory = self.directory(job_id)
        event = self.events[job_id]
        try:
            with self.adapter.guard(), self.feature_lock:
                if not self.cache.path.exists(): self.cache = DescriptorCache(self.root/"cache")
                if event.is_set(): raise RuntimeError("cancelled")
                self.update(job_id,status="running",message="Reading complete source annotations",progress=.02)
                request = read_json(directory/"request.json")
                source = self.adapter.source(request)
                images = []
                for idx,row in enumerate(source.pop("images")):
                    if event.is_set(): raise RuntimeError("cancelled")
                    path = Path(row["image_path"])
                    before = file_state(path)
                    def hash_pixels():
                        sha=hashlib.sha256()
                        with path.open("rb") as handle:
                            for chunk in iter(lambda:handle.read(1024*1024),b""): sha.update(chunk)
                        return sha.hexdigest()
                    pixel_sha = self.cache.pixel_hash(path,before,hash_pixels)
                    if before != file_state(path): raise ValueError("Source image changed during analysis")
                    objects,errors = parse_objects(row.get("label_lines",[]),len(source["labelmap"]))
                    if row.get("annotation_identity_status") == "identity_conflict": errors.append("Annotation identity conflict")
                    if not row.get("label_source_present",True): errors.append("No explicit annotation record")
                    with Image.open(path) as im:
                        width,height = im.size
                    images.append({**row,"image_id":digest({"source":source["identity"],"split":row["split"],"path":row["image_relpath"]})[:24],
                                   "image_index": idx,
                                   "image_sha256":pixel_sha,"image_state":before,"width":width,"height":height,
                                   "objects":objects,"protection_errors":errors})
                    if idx % 64 == 0:
                        self.update(job_id,progress=.03+.08*(idx+1)/max(1,source["image_count"]),message=f"Reading image {idx+1} of {source['image_count']}")
                if not images: raise ValueError("The source dataset contains no readable images")
                self.adapter.check_revision(source)
                self.adapter.check_inventory(source,images)
                source["inventory_digest"] = digest([{k:i[k] for k in ["image_id","image_sha256","label_lines","text_label","caption_records","protection_errors"]} for i in images])
                source["annotation_states_path"]=str(directory/"annotation_files.json")
                write_json(Path(source["annotation_states_path"]),source.pop("annotation_states"))
                config = PruningConfig.from_payload(request.get("config"))
                recipe = self.adapter.recipes(request,source)
                stats = {"scene_reused":0,"scene_encoded":0,"bbox_reused":0,"bbox_encoded":0,"heads_trained":0,"derived_reused":False}
                records = self.adapter.resolve_features(images, recipe, directory, event, stats)
                if request.get('config', {}).get('calibration_profile'):
                    from services.dataset_pruning_recipe_lab import prepare_calibrated_fits
                    prepare_calibrated_fits(self, request['config']['calibration_profile'], source, recipe, config)
                write_json(directory/"source.json",source)
                write_json(directory/"recipe.json",recipe)
                # Older completed derivatives can already contain assignments
                # reused from an incomplete scene-admitted pool. Invalidate the
                # complete derivative as well as its assignment reuse key.
                key = digest({"schema":DERIVED_SCHEMA,"version":VERSION,"source":source["inventory_digest"],"recipe":recipe,"similarity":config.recipe()})
                feature_recipe = {k: v for k, v in recipe.items() if k not in {'qualification', 'component_manifest'}}
                assignment_key = digest({'schema': ASSIGNMENT_SCHEMA, 'source': source['inventory_digest'],
                                         'retrieval_version': VERSION,
                                         'feature_recipe': feature_recipe, 'object_threshold': config.object_threshold,
                                         'evidence_mode': config.evidence_mode, 'scale_protection': config.scale_protection,
                                         'neighbor_k': config.neighbor_k,
                                         'candidate_scene_prefilter': config.recipe()['pair_scene_prefilter']})
                derived = self.root/"cache"/"derived"/key
                if self.valid_derived(derived,images,config.recipe()):
                    cached_images = read_json(derived/"images.json")
                    for image,previous in zip(images,cached_images): image["objects"] = previous["objects"]
                    result = read_json(derived/"result.json")
                    recipe = result["recipe"]
                    if (derived/"components.json").is_file(): shutil.copy2(derived/"components.json", directory/"components.json")
                    if (derived/'dependencies.json').is_file(): shutil.copy2(derived/'dependencies.json', directory/'dependencies.json')
                    copy_sidecar(derived, directory, sam_recipe_digest(recipe))
                    shutil.copy2(derived/"evidence.sqlite3",directory/"evidence.sqlite3")
                    evidence = PairStore(directory/"evidence.sqlite3")
                    stats.update(scene_reused=len(images),bbox_reused=sum(len(i["objects"]) for i in images),derived_reused=True)
                else:
                    previous = self.reusable_assignments(assignment_key, images)
                    retrieval = []
                    if previous:
                        cached, previous_result = previous
                        retrieval = previous_result.get('retrieval', [])
                        cached_images = read_json(cached / 'images.json')
                        for image, old in zip(images, cached_images): image['objects'] = old['objects']
                        scene = np.load(cached / 'scene.npy', mmap_mode='r', allow_pickle=False)
                        for name in ['components.json', 'dependencies.json']:
                            if (cached / name).is_file(): shutil.copy2(cached / name, directory / name)
                        copy_sidecar(cached, directory, sam_recipe_digest(recipe))
                        evidence = PairStore(directory / 'evidence.sqlite3'); (directory / 'evidence.sqlite3').unlink(missing_ok=True)
                        original = PruningConfig.from_payload(previous_result['matcher_config'])
                        try:
                            for edge in PairStore(cached / 'evidence.sqlite3'):
                                if event.is_set(): raise RuntimeError('cancelled')
                                evidence.append(reevaluate_pair_evidence(images[edge['left']], images[edge['right']], edge, config, assignment_config=original))
                            evidence.finish()
                        except BaseException:
                            evidence.abort(); raise
                        recipe['component_manifest'] = previous_result['recipe'].get('component_manifest', {})
                        stats.update(scene_reused=len(images), bbox_reused=sum(len(i['objects']) for i in images), assignments_reused=True)
                        self.update(job_id, message='Reusing exact object assignments; applying the new scene/index gates and weights', progress=.90)
                    else:
                        self.update(job_id,message="Reusing scene and bbox descriptors",progress=.12)
                        scene,obj = self.adapter.features(images,recipe,self.cache,directory,event,stats,
                            lambda p,m:self.update(job_id,progress=p,message=m), records=records)
                        if request.get('config', {}).get('calibration_profile'):
                            from services.dataset_pruning_recipe_lab import validate_analysis_profile
                            validate_analysis_profile(self, request['config']['calibration_profile'], source, recipe, config)
                        stats['sam_mask_evidence'] = collect_run_evidence(self.root, directory, source, recipe, images, records, stats)
                        (directory/"evidence.sqlite3").unlink(missing_ok=True)
                        def record_retrieval(record):
                            retrieval.append({**record, 'neighbor_execution': getattr(self.adapter, 'last_neighbor_metadata', {})})
                            write_json(directory/'retrieval.json', {'schema': 'pruning-retrieval-progress-v1', 'routes': retrieval,
                                       'candidate_count': None, 'at': time.time(), 'status': 'retrieving'})
                        def matching_progress(done, total):
                            if done <= 256 or done == total:
                                write_json(directory/'retrieval.json', {'schema': 'pruning-retrieval-progress-v1', 'routes': retrieval,
                                           'candidate_count': total, 'at': time.time(), 'status': 'matching' if done < total else 'matched'})
                            self.update(job_id,progress=.75+.15*done/max(1,total),message=f"Matching scene and objects: {done}/{total} candidate pairs")
                        execution = {}
                        try:
                            with pruning_blas_policy() as execution:
                                stats['matching_execution'] = execution
                                evidence = build_evidence(images,scene,obj,self.adapter.neighbors,
                                    matching_progress,event.is_set,PairStore(directory/"evidence.sqlite3"),config=config,
                                    retrieval_audit=record_retrieval, group_neighbors_fn=getattr(self.adapter, 'group_neighbors', None))
                        finally:
                            write_json(directory/'cpu-execution.json', {'matching': execution})
                    self.update(job_id,progress=.92,message="Protecting rare examples and computing the coverage curve")
                    with pruning_blas_policy() as execution:
                        result = build_order(images,scene,evidence,cancelled=event.is_set,config=config)
                        stats['selection_execution'] = execution
                    derived.mkdir(parents=True,exist_ok=True)
                    # The scene matrix is enough to recompute overrides: the
                    # pair evidence already contains all object matching work.
                    np.save(derived/"scene.npy",scene,allow_pickle=False)
                    shutil.copy2(directory/"evidence.sqlite3",derived/"evidence.sqlite3")
                    write_json(derived/"images.json",images)
                    result["recipe"] = recipe
                    result['derived_schema'] = DERIVED_SCHEMA
                    result['retrieval'] = retrieval
                    result['assignment_key'] = assignment_key
                    result['matcher_config'] = {name: getattr(config, name) for name in config.__dataclass_fields__}
                    if (directory/"components.json").is_file(): shutil.copy2(directory/"components.json", derived/"components.json")
                    if (directory/'dependencies.json').is_file(): shutil.copy2(directory/'dependencies.json', derived/'dependencies.json')
                    copy_sidecar(directory, derived, sam_recipe_digest(recipe))
                    write_json(derived/"result.json",result)
                    artifacts = ["result.json","images.json","scene.npy","evidence.sqlite3"] + [name for name in ['components.json', 'dependencies.json'] if (derived/name).is_file()]
                    write_json(derived/"commit.json",{"artifacts":{name:file_state(derived/name) for name in artifacts}})
                if event.is_set(): raise RuntimeError("cancelled")
                self.adapter.check_revision(source)
                self.adapter.check_inventory(source,images)
                if any(file_state(Path(image["image_path"])) != image["image_state"] for image in images):
                    raise ValueError("Source image changed during analysis; analyze again")
                stats['sam_mask_evidence'] = collect_run_evidence(self.root, directory, source, recipe, images, records, stats)
                calibration = None
                if request.get('config', {}).get('calibration_profile'):
                    from services.dataset_pruning_recipe_lab import validate_analysis_profile
                    calibration = validate_analysis_profile(self, request['config']['calibration_profile'], source, recipe, config)
                result.update(selection_revision=1,source_revision=source["revision"],image_count=len(images),
                              class_names=source["labelmap"],recipe=recipe,cache_stats=stats,derived_key=key,config=request.get("config") or {},
                              qualification=recipe.get("qualification", "unmeasured pruning calibration"))
                if calibration is not None:
                    result['calibration_profile'] = calibration
                write_json(directory/"recipe.json",recipe)
                write_json(directory/"images.json",images)
                write_json(directory/"result.json",result)
                self.inherit_manual_intent(directory, source, images)
                with ReviewLedger(directory / 'review.sqlite3').transaction() as initial_review:
                    initial_review.setdefault('budget', int(.20 * len(images)))
                shutil.copy2(derived/"scene.npy",directory/"scene.npy")
                self.cache.close_maps()
                self.adapter.register_job(directory)
                self.update(job_id,status="completed",progress=1.0,message="Pruning suggestions ready",summary={k:result[k] for k in ["image_count","max_removed_count","protected_count","cache_stats"]})
        except Exception as exc:
            detail = getattr(exc,"detail",str(exc))
            self.update(job_id,status="cancelled" if event.is_set() else "failed",message=str(detail),error=str(detail))
        finally:
            with self.feature_lock: self.cache.close_maps()
            # Assembly memmaps are disposable; durable shared descriptors and
            # derived pair/curve artifacts survive subsequent jobs.
            shutil.rmtree(directory/"features",ignore_errors=True)
            self.events.pop(job_id,None)

    @staticmethod
    def public_result(result):
        return {k:v for k,v in result.items() if k not in {"order","groups","derived_key","assignment_key","matcher_config","protection_reasons","keep_receipts"}}

    @staticmethod
    def dependency_status(directory):
        path = directory / 'dependencies.json'
        if not path.is_file(): return {'manifest_available': False, 'dependencies': [], 'optional_caches': []}
        manifest = read_json(path)
        dependencies = manifest.get('dependencies', [])
        if not isinstance(dependencies, list) or len(dependencies) > 128:
            raise HTTPException(409, 'Saved pruning dependencies cannot be read; preserve the review for inspection')
        rows = []
        for item in dependencies:
            target = Path(item['path'])
            try:
                status = 'available' if not target.is_symlink() and file_state(target) == item.get('state') else 'changed'
            except FileNotFoundError: status = 'missing'
            except OSError: status = 'unavailable'
            rows.append({'role': item.get('role'), 'required': bool(item.get('required')), 'status': status,
                         'sha256': item.get('sha256'), 'content_attestation': item.get('content_attestation')})
        caches = [{'role': item.get('role'), 'status': 'available' if Path(item['path']).is_dir() else 'missing_or_purged'}
                  for item in manifest.get('optional_caches', [])]
        return {'manifest_available': True, 'dependencies': rows, 'optional_caches': caches,
                'saved_review_self_contained': True, 'note': manifest.get('optional_cache_note')}

    def result(self,job_id):
        context, state, plan = self._current_review_plan(job_id)
        result = context[1]
        return {**self.public_result(result), **{key: plan['order_result'][key] for key in ['curve', 'max_removed_count', 'protected_count']},
                    **{key: plan[key] for key in ['review_revision', 'plan_revision', 'approved_count', 'plan_digest', 'budget']},
                    'active_settings': settings_for(result, state), 'active_policy': settings_for(result, state)['policy'],
                    'proposed_loss': plan.get('proposed_loss'), 'approved_loss': plan.get('approved_loss'),
                    'accounting_digest': plan.get('accounting_digest'),
                    'accounting_outdated': plan.get('accounting_outdated', False),
                    'appearance_group_count': plan.get('appearance_group_count'), 'total_appearance_weight': plan.get('total_appearance_weight'),
                    'index_weights': {'objects': settings_for(result, state)['object_weight'], 'scene': 1-settings_for(result, state)['object_weight']},
                    'dependency_status': self.dependency_status(context[0])}

    def load_selection(self,job_id,payload,check_selection=True):
        directory = self.directory(job_id)
        if self.status(job_id)["status"] != "completed": raise HTTPException(409,"Pruning analysis is not ready")
        snapshot = self.completed_snapshots.load(directory)
        result, source, images = snapshot.result, snapshot.source, snapshot.images
        expected_similarity = PruningConfig.from_payload(result.get("config")).recipe() if "config" in result else SIMILARITY_RECIPE
        if result.get("similarity_recipe")!=expected_similarity:
            raise HTTPException(409,"The similarity rules were updated. Analyze again; cached descriptors will be reused")
        if check_selection and payload.get("selection_revision") != result["selection_revision"]:
            raise HTTPException(409,"The pruning selection changed. Refresh its curve before continuing.")
        self.adapter.check_revision(source)
        self.adapter.check_inventory(source,images)
        verify_artifacts(snapshot.binding)
        try: chosen = selection(images,result,payload.get("removed_count",0))
        except ValueError as exc: raise HTTPException(400,str(exc)) from exc
        return directory,result,source,images,chosen

    def review(self,job_id,payload):
        if payload.get('view') == 'impact' or payload.get('preview_id') or ReviewLedger(self.directory(job_id) / 'review.sqlite3').read().get('plans', {}).get('active'):
            return self.planning_selection(job_id, payload)
        with self.lock:
            directory,result,source,images,chosen = self.load_selection(job_id,payload)
        offset = max(0,int(payload.get("offset",0)))
        limit = min(24,max(1,int(payload.get("limit",8))))
        evidence = PairStore(directory/"evidence.sqlite3")
        retained = set(chosen["retained"])
        rows = []
        for i in chosen["removed"][offset:offset+limit]:
            match = removal_evidence(i,retained,evidence)
            rows.append({"removed":self.public_image(images[i]),"retained":self.public_image(images[match["retained_index"]]),
                         "evidence":{k:v for k,v in match.items() if k not in {"left","right","retained_index"}}})
        splits = grouped_splits(images,chosen["retained"],result["groups"])
        return {"selection_revision":result["selection_revision"],"removed_count":len(chosen["removed"]),"retained_count":len(retained),
                "rows":rows,"offset":offset,"next_offset":offset+len(rows) if offset+len(rows)<len(chosen["removed"]) else None,
                "coverage":{k:v for k,v in chosen.items() if k not in {"retained","removed"}},"splits":{k:v for k,v in splits.items() if k!="assignments"}}

    @staticmethod
    def public_image(image):
        return {k:thaw(image[k]) for k in ["image_id","image_relpath","split","width","height","objects","protection_errors", "image_index", "display_id"] if k in image}

    def keep(self,job_id,payload):
        # Legacy selectors rewrote result.json outside the revisioned review
        # ledger. All current UI pins use one receipt/CAS/history transaction.
        raise HTTPException(409, 'Use the revisioned pruning review endpoint to change Always Keep images; this legacy selector cannot modify a saved review')

    def review_context(self, job_id, payload, check_selection=True):
        directory, result, source, images, _ = self.load_selection(job_id, {**payload, 'removed_count': 0}, check_selection)
        ledger = ReviewLedger(directory / 'review.sqlite3')
        snapshot = self.completed_snapshots.load(directory)
        paths = [directory / name for name in ('scene.npy', 'evidence.sqlite3', 'evidence.sqlite3-wal')]
        binding = snapshot.binding + artifact_states(paths)
        scene = np.load(directory / 'scene.npy', mmap_mode='r', allow_pickle=False)
        evidence = PairStore(directory / 'evidence.sqlite3')
        config = PruningConfig.from_payload(result['config']) if 'config' in result else None
        verify_artifacts(binding)
        context = ServingContext((directory, result, source, images, ledger, scene, evidence, config), binding)
        self.reconcile_manual_intent(context)
        return context

    @staticmethod
    def manual_intent_fields(state):
        """Retain Keep events even when the latest action was Allow removal.

        Earlier ledgers lacked last_keep_at. Their normal unpin history records
        the previous owned pin, so recover that timestamp without guessing a
        new user action or changing the older ledger.
        """
        fields = copy.deepcopy(state['pins'])
        def keep_at(field):
            return max(field.get('last_keep_at', 0), field.get('intent_at', 0) if field.get('value') else 0)
        for field in fields.values(): field['last_keep_at'] = keep_at(field)
        for operation in state.get('history', []):
            for touched in operation.get('touched_fields', []):
                if touched.get('collection') != 'pins': continue
                field = fields.get(touched.get('key'))
                before = touched.get('before') or {}
                if field and before.get('image_sha256') == field.get('image_sha256'):
                    field['last_keep_at'] = max(field['last_keep_at'], keep_at(before))
        return fields

    def latest_manual_intent(self, source, images):
        latest = {}
        current = {image['image_id']: image['image_sha256'] for image in images}
        if not source.get('identity'): return latest
        for previous in self.root.glob('dp_*'):
            if previous.is_symlink() or not (previous / 'review.sqlite3').is_file(): continue
            try:
                if read_json(previous / 'source.json').get('identity') != source['identity']: continue
                state = ReviewLedger(previous / 'review.sqlite3').read()
                for identifier, field in self.manual_intent_fields(state).items():
                    if field.get('image_sha256') != current.get(identifier): continue
                    last_keep = max(field.get('last_keep_at', 0), latest.get(identifier, {}).get('last_keep_at', 0))
                    if field.get('intent_at', 0) > latest.get(identifier, {}).get('intent_at', -1):
                        latest[identifier] = {**copy.deepcopy(field), 'intent_origin_job': field.get('intent_origin_job', previous.name)}
                    latest[identifier]['last_keep_at'] = last_keep
            except (OSError, ValueError, KeyError, sqlite3.Error) as error:
                raise HTTPException(503, 'Saved manual Keep intent needs repair before reviewing this source') from error
        return latest

    def reconcile_manual_intent(self, context):
        directory, _, source, images, ledger, _, _, _ = context
        latest = self.latest_manual_intent(source, images)
        if not latest: return
        with ledger.transaction() as state:
            previous = self.manual_intent_fields(state)
            changed = {key: field for key, field in latest.items()
                       if field.get('intent_at', 0) > previous.get(key, {}).get('intent_at', -1)
                       or field.get('last_keep_at', 0) > previous.get(key, {}).get('last_keep_at', 0)}
            if not changed: return
            missed_keeps = {key for key, field in changed.items()
                            if field.get('last_keep_at', 0) > previous.get(key, {}).get('last_keep_at', 0)}
            state['review_revision'] += 1; state['plan_revision'] += 1
            for key, field in changed.items():
                state['pins'][key] = {**field, 'owner': 'inherited:' + field['intent_origin_job'], 'version': state['review_revision']}
            state['inherited_manual_intent'] = {'source_identity': source['identity'], 'count': len(changed),
                                              'policy': 'latest committed exact image ID and pixel SHA, reconciled for every review'}
            active = bool(state.get('plans', {}).get('active'))
            projection = state
            if active and missed_keeps:
                projection = copy.deepcopy(state)
                for key in missed_keeps: projection['pins'][key]['value'] = True
            plan = self.plan_for(context, projection, state.get('budget', 0))
            if state.get('plans', {}).get('active') and plan.get('coverage_refresh_required'):
                # The reduced candidate set is committed with the inherited
                # Keep. A later unpin cannot reveal the older larger artifact.
                state['plans']['active'] = {'value': self._store_plan(context, plan),
                    'owner': 'inherited-keep', 'version': state['review_revision']}
            self.supersede_approvals(state, plan)

    @staticmethod
    def supersede_approvals(state, plan):
        for candidate, decision in state['decisions'].items():
            proposal = plan['proposals'].get(candidate)
            decision['active'] = bool(proposal and proposal['evidence_digest'] == decision.get('evidence_digest') and
                                      proposal['witness_id'] == decision.get('witness_id'))
            if decision.get('status') == 'approved' and not decision['active']:
                decision['status'] = 'superseded'
                decision['superseded_at_revision'] = state['review_revision']

    def inherit_manual_intent(self, directory, source, images):
        """Reconcile committed Keep intent, never old approvals/constraints.

        The old job ledger itself is the receipt authority. There is no second
        writable intent store that could diverge after a lost acknowledgement.
        Pixel binding prevents transferring a pin to replaced image bytes.
        """
        latest = {}
        current = {image['image_id']: image['image_sha256'] for image in images}
        for previous in self.root.glob('dp_*'):
            if previous == directory or previous.is_symlink() or not (previous / 'review.sqlite3').is_file(): continue
            try:
                if read_json(previous / 'source.json')['identity'] != source['identity']: continue
                state = ReviewLedger(previous / 'review.sqlite3').read()
                for identifier, field in self.manual_intent_fields(state).items():
                    if field.get('image_sha256') != current.get(identifier): continue
                    last_keep = max(field.get('last_keep_at', 0), latest.get(identifier, {}).get('last_keep_at', 0))
                    if field.get('intent_at', 0) > latest.get(identifier, {}).get('intent_at', -1): latest[identifier] = copy.deepcopy(field)
                    latest[identifier]['last_keep_at'] = last_keep
            except (OSError, ValueError, KeyError, sqlite3.Error) as error:
                raise HTTPException(503, 'Saved manual Keep intent needs repair before creating another review') from error
        if latest:
            with ReviewLedger(directory / 'review.sqlite3').transaction() as state:
                state['pins'] = latest
                state['inherited_manual_intent'] = {'source_identity': source['identity'], 'count': sum(bool(field['value']) for field in latest.values()), 'policy': 'exact_image_id_and_pixel_sha_only'}

    @staticmethod
    def review_public(state, plan, result):
        snapshot = {key: copy.deepcopy(value) for key, value in state.items() if key not in {'operations', 'previews', 'comparisons'}}
        snapshot['additional_pair_count'] = len(state.get('comparisons', {}))
        snapshot.update({key: plan[key] for key in ['plan_digest', 'approved_count', 'removed_count', 'retained_count']})
        snapshot.update(selection_revision=result['selection_revision'], removed_percent=100 * plan['removed_count'] / max(1, result['image_count']),
                        budget=plan['budget'], max_removed_count=plan['order_result']['max_removed_count'], curve=plan['order_result']['curve'])
        snapshot['coverage'] = {key: value for key, value in plan['chosen'].items() if key not in {'removed', 'retained'}}
        snapshot.update(active_settings=settings_for(result, state), active_policy=settings_for(result, state)['policy'],
                        proposed_loss=plan.get('proposed_loss'), approved_loss=plan.get('approved_loss'),
                        accounting_digest=plan.get('accounting_digest'), appearance_group_count=plan.get('appearance_group_count'),
                        total_appearance_weight=plan.get('total_appearance_weight'), coverage_refresh_required=plan.get('coverage_refresh_required', False),
                        accounting_outdated=plan.get('accounting_outdated', False))
        return snapshot

    def plan_for(self, context, state, budget):
        if state.get('plans', {}).get('active'):
            return self._load_active_plan(context, state)
        directory, result, source, images, _, scene, evidence, config = context
        binding = context.binding
        verify_artifacts(binding)
        base = digest({'artifacts': binding, 'source': source['inventory_digest'],
                       'recipe': result['recipe'], 'selection_revision': result['selection_revision']})
        supplementary = {key: field['edge'] for key, field in state.get('comparisons', {}).items()}
        topology_key = digest({'base': base, 'comparisons': supplementary})
        families = self.topology_views.get(topology_key)
        if families is None:
            families = family_inventory(images, SupplementaryEvidence(evidence, state.get('comparisons')))
            verify_artifacts(binding)
            self.topology_views.put(topology_key, families)
        pinned = {key for key, field in state['pins'].items() if field['value']}
        for field in state['constraints'].values(): pinned.update(field['value']['keeper_ids'])
        order_key = digest({'topology': topology_key, 'pins': sorted(pinned),
                            'rejected': sorted(state['rejected']), 'config': result.get('config')})
        plan_key = digest({'order': order_key, 'budget': budget, 'constraints': state['constraints'],
                           'pins': state['pins'], 'decisions': state['decisions'],
                           'review_revision': state['review_revision'], 'plan_revision': state['plan_revision']})
        cached = self.plan_views.get(plan_key)
        if cached is not None:
            verify_artifacts(binding)
            return cached
        order = self.order_views.get(order_key)
        if order is None:
            if not pinned and not state['rejected'] and not supplementary and not result.get('keep_ids'):
                fields = ('version', 'similarity_recipe', 'order', 'protected_count', 'max_removed_count', 'curve',
                          'protection_reasons', 'groups', 'keep_ids', 'coverage_note', 'index_weights', 'thresholds')
                order = {key: thaw(result[key]) for key in fields}
            else:
                order = build_order(images, scene, FilteredEvidence(evidence, state['rejected'], state.get('comparisons')), sorted(pinned), config=config)
            verify_artifacts(binding)
            self.order_views.put(order_key, order)
        plan = compute_plan(images, result, source, scene, evidence, state, budget, config,
                            family_population=families, order_result=order)
        verify_artifacts(binding)
        self.plan_views.put(plan_key, plan)
        return plan

    def review_snapshot(self, job_id):
        context, state, plan = self._current_review_plan(job_id)
        return self.review_public(state, plan, context[1])

    def families(self, job_id, payload):
        if payload.get('preview_id') or ReviewLedger(self.directory(job_id) / 'review.sqlite3').read().get('plans', {}).get('active'):
            return self.planning_families(job_id, payload)
        with self.adapter.guard(), self.lock:
            context = self.review_context(job_id, payload)
            directory, result, source, images, ledger, scene, evidence, _ = context
            state = ledger.read()
            if payload.get('plan_revision') is not None and payload['plan_revision'] != state['plan_revision']:
                raise HTTPException(409, 'Family plans changed. Refresh this review generation.')
            if payload.get('expected_review_revision') is not None and payload['expected_review_revision'] != state['review_revision']:
                raise HTTPException(409, 'Review changed. Refresh the filtered family population.')
            if 'budget' not in state:
                with ledger.transaction() as initial: initial['budget'] = payload.get('removed_count', 0)
                state = ledger.read()
            elif payload.get('removed_count', 0) != state['budget']:
                raise HTTPException(409, 'Commit the new removal ceiling before loading its authoritative families.')
            plan = self.plan_for(context, state, payload.get('removed_count', 0))
            family_rows = list(plan['families'].values())
            query = str(payload.get('query') or '').casefold().strip()
            filter_name, sort_name = payload.get('filter', 'all'), payload.get('sort', 'stable')
            if filter_name not in {'all', 'unreviewed', 'approved', 'exceptions'} or sort_name not in {'stable', 'size', 'score'}:
                raise HTTPException(400, 'Unknown family filter or order')
            if query:
                family_rows = [row for row in family_rows if query in row['id'].casefold() or any(query in images[i]['image_relpath'].casefold() or query in images[i]['image_id'] for i in row['member_indices'])]
            if filter_name == 'exceptions': family_rows = [row for row in family_rows if row['exception_reasons']]
            elif filter_name != 'all': family_rows = [row for row in family_rows if any(p['status'] == filter_name for p in row['proposals'])]
            def key(row):
                return ((-len(row['member_indices']) if sort_name == 'size' else -max((p['evidence']['combined'] for p in row['proposals']), default=0) if sort_name == 'score' else 0), row['id'])
            family_rows.sort(key=key)
            captured = [row['id'] for row in family_rows]
            offset, limit = max(0, int(payload.get('offset', 0))), min(24, max(1, int(payload.get('limit', 12))))
            rows = []
            for original in family_rows[offset:offset + limit]:
                row = copy.deepcopy(original)
                members = row.pop('member_indices')
                row['members'] = [self.public_image(images[i]) for i in members[:120]]
                row.update(self.family_navigation(context, state, members, row['requested_count']))
                rows.append(row)
            return {**self.review_public(state, plan, result), 'family_count': len(plan['families']), 'filtered_family_count': len(captured),
                    'filtered_family_ids': captured, 'families': rows, 'offset': offset,
                    'next_offset': offset + len(rows) if offset + len(rows) < len(captured) else None,
                    'generation': digest({'source': source['inventory_digest'], 'recipe': plan['recipe_digest'], 'plan': plan['plan_digest'], 'review_revision': state['review_revision']})}

    def family_navigation(self, context, state, members, requested_count):
        _, _, _, images, _, scene, evidence, config = context
        verify_artifacts(context.binding)
        key = digest({'artifacts': context.binding, 'members': members,
                      'pins': sorted(identifier for identifier, field in state['pins'].items() if field['value']),
                      'rejected': sorted(state['rejected']),
                      'comparisons': {identifier: field['edge'] for identifier, field in state.get('comparisons', {}).items()},
                      'config': config.recipe() if config else None,
                      'requested_count': requested_count if len(members) > 128 else 0})
        cached = self.family_views.get(key)
        if cached is not None:
            verify_artifacts(context.binding)
            return cached
        row = {'requested_count': requested_count}
        ids = set(members[:120])
        edges = []
        for i in members[:120]:
            for edge in FilteredEvidence(evidence, state['rejected'], state.get('comparisons', {})).edges_for(i):
                if edge['left'] != i or edge['right'] not in ids: continue
                for candidate, witness, side in [(edge['left'], edge['right'], 'left'), (edge['right'], edge['left'], 'right')]:
                    if not edge[side + '_valid']: continue
                    edges.append({'candidate_id': images[candidate]['image_id'], 'witness_id': images[witness]['image_id'],
                                  'combined': edge[side + '_combined'],
                                  'scene': edge['scene'] if edge.get('scene_evidence_used', True) else None,
                                  'object': edge[side + '_object'] if edge.get('object_evidence_used', True) else None,
                                  'minimum_object_appearance': edge[side + '_minimum_object_appearance'] if edge.get('object_evidence_used', True) else None})
                    if len(edges) > 500: break
                if len(edges) > 500: break
            if len(edges) > 500: break
        display_members = members[:120]
        vectors = np.asarray(scene[display_members], np.float32)
        centered = vectors - vectors.mean(axis=0, keepdims=True)
        if len(members) > 1:
            left, values, _ = np.linalg.svd(centered, full_matrices=False)
            xy = np.zeros((len(display_members), 2), np.float32)
            width = min(2, len(values)); xy[:, :width] = left[:, :width] * values[:width]
        else: xy = np.zeros((1, 2), np.float32)
        row['graph'] = {'edges': edges, 'points': [{'image_id': images[i]['image_id'], 'x': float(xy[j, 0]), 'y': float(xy[j, 1])} for j, i in enumerate(display_members)],
                        'layout_note': ('Navigation positions only: whole-image descriptors are disabled' if not scene.shape[1] else
                                        '2-D projection of the displayed scene descriptors for navigation; direct edge evidence decides removals; additional loaded members use navigation positions')}
        local_indices = {index: ordinal for ordinal, index in enumerate(members)}
        local_edges = [{**edge, 'left': local_indices[edge['left']], 'right': local_indices[edge['right']]}
                       for edge in family_edges(FilteredEvidence(evidence, state['rejected'], state.get('comparisons', {})), members)]
        counts = list(range(1, min(12, len(members)) + 1))
        counts += [count for count in [16, 24, 32, 48, 64, 96, 128, len(members)] if 12 < count <= len(members)]
        if len(members) > 128:
            counts = [1, min(128, max(1, row['requested_count'])), 128, len(members)]
        row['coverage_curve'] = []
        for count in sorted(set(counts)):
            selected_ids = choose_family_keepers(images, members, FilteredEvidence(evidence, state['rejected'], state.get('comparisons', {})), scene, count,
                                                 [identifier for identifier, value in state['pins'].items() if value['value']])
            selected = [local_indices[index] for index in members if images[index]['image_id'] in set(selected_ids)]
            coverage = coverage_for_retained([images[index] for index in members], selected, local_edges, context[-1])
            row['coverage_curve'].append({'keeper_count': len(selected), 'requested_keeper_count': count,
                                          **{key: coverage[key] for key in ['scene_coverage', 'object_coverage', 'worst_class_coverage', 'coverage']}})
        row['total_member_count'] = len(members)
        row['graph_truncated'] = len(members) > 120 or len(edges) > 500
        row['members_truncated'] = len(members) > 120
        row['member_next_offset'] = 120 if len(members) > 120 else None
        row['graph']['points'] = row['graph']['points'][:120]
        row['graph']['edges'] = row['graph']['edges'][:500]
        row.pop('requested_count')
        verify_artifacts(context.binding)
        self.family_views.put(key, row)
        return row

    def family_members(self, job_id, family_id, payload):
        if payload.get('preview_id'): return self.planning_preview_members(job_id, family_id, payload)
        with self.adapter.guard(), self.lock:
            context = self.review_context(job_id, payload)
            _, result, source, images, ledger, _, _, _ = context
            state = ledger.read()
            if payload.get('expected_review_revision') != state['review_revision'] or payload.get('plan_revision') != state['plan_revision']:
                raise HTTPException(409, 'The family generation changed. Refresh before loading more members.')
            if payload.get('removed_count') != state.get('budget', 0):
                raise HTTPException(409, 'The committed removal ceiling changed.')
            offset, limit = payload.get('offset', 0), payload.get('limit', 120)
            if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 120:
                raise HTTPException(400, 'Invalid member continuation bounds')
            plan = self.plan_for(context, state, state['budget'])
            generation = digest({'source': source['inventory_digest'], 'recipe': plan['recipe_digest'], 'plan': plan['plan_digest'], 'review_revision': state['review_revision']})
            if payload.get('generation') != generation: raise HTTPException(409, 'This member cursor belongs to another review generation')
            family = plan['families'].get(family_id)
            if not family: raise HTTPException(404, 'This family is unavailable in the current plan')
            members = family['member_indices']
            page = members[offset:offset + limit]
            return {'generation': generation, 'family_id': family_id, 'members': [self.public_image(images[index]) for index in page],
                    'offset': offset, 'next_offset': offset + len(page) if offset + len(page) < len(members) else None,
                    'total_member_count': len(members), 'review_revision': state['review_revision'], 'plan_revision': state['plan_revision']}

    def plan_preview(self, job_id, payload):
        if payload.get('kind') == 'settings': return self.settings_preview(job_id, payload)
        if ReviewLedger(self.directory(job_id) / 'review.sqlite3').read().get('plans', {}).get('active'):
            return self.planning_family_preview(job_id, payload)
        with self.adapter.guard(), self.lock:
            context = self.review_context(job_id, payload)
            _, result, source, images, ledger, scene, evidence, _ = context
            with ledger.transaction() as state:
                if payload.get('expected_review_revision') != state['review_revision']:
                    raise HTTPException(409, 'Review changed. Refresh before previewing this captured population.')
                population = {key: value['member_indices'] for key, value in self.plan_for(context, state, state.get('budget', payload.get('removed_count', 0)))['families'].items()}
                family_ids = [payload['family_id']] if payload.get('family_id') else payload.get('family_ids')
                family_ids = sorted(population) if family_ids is None else list(family_ids)
                if len(set(family_ids)) != len(family_ids) or set(family_ids) - set(population):
                    raise HTTPException(400, 'The captured family population is invalid')
                requested = payload.get('keep_count', 1)
                draft = copy.deepcopy(state)
                changed = {}
                pinned = [key for key, field in state['pins'].items() if field['value']]
                filtered = FilteredEvidence(evidence, state['rejected'], state.get('comparisons', {}))
                for family_id in family_ids:
                    count = min(requested, len(population[family_id])) if type(requested) is int and requested > 0 else requested
                    keeper_ids = choose_family_keepers(images, population[family_id], filtered, scene, count, pinned,
                                                       payload.get('keeper_ids') if payload.get('family_id') == family_id else None)
                    value = {'keeper_ids': keeper_ids, 'requested_count': requested}
                    changed[family_id] = value
                    draft['constraints'][family_id] = {'value': value, 'owner': 'preview', 'version': state['review_revision'] + 1}
                budget = payload.get('removed_count', 0)
                if 'budget' in state and state['budget'] != budget:
                    raise HTTPException(409, 'Commit the new removal ceiling before previewing a keeper plan.')
                previous = self.plan_for(context, state, budget)
                plan = self.plan_for(context, draft, budget)
                for family_id, value in changed.items():
                    value['confirmed_keeper_ids'] = plan['families'][family_id]['keeper_ids']
                plan = self.plan_for(context, draft, budget)
                member_ids = {images[i]['image_id'] for family_id in family_ids for i in population[family_id]}
                proposals = {key: value for key, value in plan['proposals'].items() if key in member_ids}
                superseded = sum(decision.get('status') == 'approved' and (key not in plan['proposals'] or plan['proposals'][key]['evidence_digest'] != decision.get('evidence_digest')) for key, decision in state['decisions'].items())
                source_counts = Counter(obj['class_id'] for image in images for obj in image['objects'])
                retained_counts = Counter(obj['class_id'] for i in plan['chosen']['retained'] for obj in images[i]['objects'])
                preview = {'source_revision': source['revision'], 'recipe_digest': plan['recipe_digest'], 'selection_revision': result['selection_revision'],
                           'plan_revision': state['plan_revision'], 'review_revision': state['review_revision'], 'family_ids': family_ids,
                           'requested_keep_count': requested, 'retained_count': plan['retained_count'], 'removed_count': plan['removed_count'],
                           'captured_eligible_count': len(proposals),
                           'captured_member_count': sum(len(plan['families'][key]['member_indices']) for key in family_ids),
                           'captured_retained_count': sum(len(plan['families'][key]['member_indices']) for key in family_ids) - len(proposals),
                           'removed_percent': 100 * plan['removed_count'] / max(1, len(images)), 'budget': budget,
                           'additional_keeper_count': sum(max(0, len(plan['families'][key]['keeper_ids']) - requested) for key in family_ids),
                           'new_approval_count': sum(value['status'] != 'approved' for value in proposals.values()), 'superseded_count': superseded,
                           'exceptions': [{'family_id': key, 'reasons': plan['families'][key]['exception_reasons']} for key in family_ids if plan['families'][key]['exception_reasons']],
                           'class_impact': {str(cls): {'source': count, 'retained': retained_counts[cls]} for cls, count in source_counts.items()},
                           'constraints': changed, 'proposals': proposals, 'plan_digest': plan['plan_digest'],
                           'plan': {key: {field: value for field, value in plan['families'][key].items() if field != 'member_indices'} for key in family_ids}}
                preview_digest = digest(preview)
                state['previews'][preview_digest] = preview
                return {'preview_digest': preview_digest, **{key: value for key, value in preview.items() if key not in {'constraints', 'proposals'}}}

    def measure_direct(self, job_id, payload):
        """Stage cache-only family comparisons, then commit one ledger CAS.

        Keep/source changes during calculation reject publication. There is no
        second evidence DB commit, inference or fitting behind this action.
        """
        token = str(payload.get('request_id') or '')
        if not re.fullmatch(r'[a-f0-9]{32}', token): raise HTTPException(400, 'A stable review request identity is required')
        identity = digest(payload)
        staged_path = self.directory(job_id)/('.direct-'+token+'-'+uuid.uuid4().hex+'.sqlite3')
        staged = PairStore(staged_path)
        with self.adapter.guard():
            with self.lock:
                ledger = ReviewLedger(self.directory(job_id)/'review.sqlite3')
                state = ledger.read(); receipt = state['operations'].get(token)
                if receipt:
                    if receipt['identity'] != identity: raise HTTPException(409, 'This identity belongs to another operation')
                    try:
                        context = self.review_context(job_id, payload, False)
                        state = ledger.read()
                        return {'outcome': 'committed', 'receipt': receipt,
                                'snapshot': self.review_public(state, self.plan_for(context, state, state.get('budget', 0)), context[1])}
                    except HTTPException as error:
                        return {'outcome': 'committed', 'receipt': receipt, 'snapshot': None, 'source_stale': True,
                                'message': 'Direct comparisons committed; the current source must be refreshed.', 'current_source_error': error.detail}
                context = self.review_context(job_id, payload, False)
                directory, result, source, images, ledger, scene, evidence, config = context
                state = ledger.read()
                if payload.get('selection_revision') != result['selection_revision'] or payload.get('expected_review_revision') != state['review_revision'] or payload.get('plan_revision') != state['plan_revision']:
                    raise HTTPException(409, 'The family review changed; refresh before calculating direct comparisons')
                if payload.get('removed_count', 0) != state.get('budget', 0): raise HTTPException(409, 'The removal ceiling changed')
                plan = self.plan_for(context, state, state.get('budget', 0))
                family = plan['families'].get(payload.get('family_id'))
                keepers = payload.get('keeper_ids')
                members = set(family['member_indices']) if family else set()
                ids = {images[i]['image_id']: i for i in members}
                if not family or not isinstance(keepers, list) or not keepers or len(set(keepers)) != len(keepers) or set(keepers)-set(ids):
                    raise HTTPException(400, 'Select distinct keepers from one current family')
                captured = (state['review_revision'], state['plan_revision'], result['selection_revision'], source['inventory_digest'], digest(result['recipe']))
                measured = SupplementaryEvidence(evidence, state.get('comparisons'))
                manifest = read_json(directory/'components.json') if (directory/'components.json').is_file() else {}
                if not result['recipe'].get('component_manifest') or digest(manifest) != digest(result['recipe']['component_manifest']):
                    raise HTTPException(409, 'Direct comparison is unmeasured: its saved transform/component manifest is absent or changed. Recompute this analysis before adding comparisons.')
            count = 0
            try:
                from services.dataset_pruning_comparison import reconstruct_cached_object_rows
                with self.feature_lock:
                    with pruning_blas_policy():
                        for witness in sorted(ids[key] for key in keepers):
                            for candidate in sorted(members):
                                if candidate == witness or measured.pair(candidate, witness) is not None: continue
                                a, b = sorted((candidate, witness))
                                if staged.writer is not None and staged.writer.execute('SELECT 1 FROM pairs WHERE a=? AND b=?', (a, b)).fetchone(): continue
                                features = reconstruct_cached_object_rows(self.adapter.api, self.cache, [images[a], images[b]], result['recipe'], manifest)
                                similarity = float(scene[a]@scene[b]) if config.evidence_mode != 'objects_only' else 0.
                                staged.append({'left': a, 'right': b, **pair_evidence(images[a], images[b], similarity, features, config)})
                                count += 1
                    staged.finish()
                with self.lock:
                    current = self.review_context(job_id, payload, False)
                    current_result, current_source = current[1], current[2]
                    if digest(read_json(directory/'components.json')) != digest(manifest):
                        raise HTTPException(409, 'Saved component transforms changed while computing. No comparisons were published; recompute this analysis.')
                    with ledger.transaction() as latest:
                        now = (latest['review_revision'], latest['plan_revision'], current_result['selection_revision'], current_source['inventory_digest'], digest(current_result['recipe']))
                        if now != captured: raise HTTPException(409, 'The source, Keep choices or review changed while computing. No comparisons were published; refresh and retry.')
                        revision = latest['review_revision']+1
                        comparisons = latest.setdefault('comparisons', {})
                        for edge in staged:
                            comparisons[SupplementaryEvidence.key(edge['left'], edge['right'])] = {'edge': edge, 'owner': token, 'version': revision,
                                'source_digest': source['inventory_digest'], 'recipe_digest': captured[-1], 'producer': 'cache_only_exact_saved_transform_v1'}
                        latest['review_revision'] = revision
                        latest['plan_revision'] += int(count > 0)
                        final = self.plan_for(current, latest, latest.get('budget', 0))
                        self.supersede_approvals(latest, final)
                        final = self.plan_for(current, latest, latest.get('budget', 0))
                        receipt = {'request_id': token, 'identity': identity, 'revision': revision, 'plan_revision': latest['plan_revision'],
                                   'action': 'measure_direct', 'family_id': payload['family_id'], 'computed_pair_count': count,
                                   'at': time.time(), 'batch_id': None, 'conflicts': []}
                        latest['operations'][token] = receipt
                        latest.setdefault('history', []).append({**receipt, 'keeper_ids': keepers, 'source_digest': source['inventory_digest'], 'recipe_digest': captured[-1]})
                        self.adapter.check_revision(source); self.adapter.check_inventory(source, images)
                        if digest(read_json(directory/'components.json')) != digest(manifest):
                            raise HTTPException(409, 'Saved component transforms changed before publication. No comparisons were published; recompute this analysis.')
                        return {'outcome': 'committed', 'receipt': receipt, 'snapshot': self.review_public(latest, final, result),
                                'message': f'{count} previously unmeasured direct comparisons computed from cached descriptors and exact saved transforms. Preview and confirm the resulting keeper plan.'}
            finally:
                staged.abort(); staged_path.unlink(missing_ok=True)

    def review_mutation(self, job_id, payload):
        if payload.get('action') in {'apply_plan', 'pin'} or ReviewLedger(self.directory(job_id) / 'review.sqlite3').read().get('plans', {}).get('active'):
            return self.planning_review_mutation(job_id, payload)
        if payload.get('action') == 'measure_direct': return self.measure_direct(job_id, payload)
        token = str(payload.get('request_id') or '')
        if not re.fullmatch(r'[a-f0-9]{32}', token): raise HTTPException(400, 'A stable review request identity is required')
        identity = digest(payload)
        with self.adapter.guard(), self.lock:
            saved = ReviewLedger(self.directory(job_id) / 'review.sqlite3').read()['operations'].get(token)
            if saved and saved['identity'] != identity:
                raise HTTPException(409, 'This request identity belongs to another review operation')
            try:
                context = self.review_context(job_id, payload, False)
            except HTTPException as error:
                if not saved: raise
                return {'outcome': 'committed', 'receipt': saved, 'snapshot': None, 'source_stale': True,
                        'message': 'The operation committed. The current source must be refreshed before further review.',
                        'current_source_error': error.detail}
            _, result, source, images, ledger, _, evidence, _ = context
            with ledger.transaction() as state:
                receipt = state['operations'].get(token)
                if receipt:
                    if receipt['identity'] != identity: raise HTTPException(409, 'This request identity belongs to another review operation')
                    plan = self.plan_for(context, state, state.get('budget', 0))
                    return {'outcome': 'committed', 'receipt': receipt, 'snapshot': self.review_public(state, plan, result)}
                if payload.get('selection_revision') != result['selection_revision'] or payload.get('expected_review_revision') != state['review_revision']:
                    raise HTTPException(409, 'Review generation changed. Refresh before applying this operation.')
                budget = payload.get('removed_count', 0)
                action = payload.get('action')
                previous_decision = copy.deepcopy(state['decisions'].get(payload.get('candidate_id')))
                if action != 'set_budget' and 'budget' in state and state['budget'] != budget:
                    raise HTTPException(409, 'The removal ceiling changed. Refresh the committed plan before reviewing.')
                plan = self.plan_for(context, state, budget)
                revision = state['review_revision'] + 1
                conflicts = []
                plan_changed = False
                touched = []
                def assign(collection, key, value):
                    prior = copy.deepcopy(state[collection].get(key))
                    state[collection][key] = {**value, 'owner': token, 'version': revision}
                    touched.append({'collection': collection, 'key': key, 'before': prior})
                if action == 'set_budget':
                    plan_changed = True
                elif action == 'pin':
                    identifier = payload.get('image_id')
                    if identifier not in {image['image_id'] for image in images} or type(payload.get('keep')) is not bool:
                        raise HTTPException(400, 'Pin a specific image using an explicit keep value')
                    image = next(image for image in images if image['image_id'] == identifier)
                    intent_at = time.time_ns()
                    previous = self.manual_intent_fields(state).get(identifier, {})
                    assign('pins', identifier, {'value': payload['keep'], 'note': str(payload.get('note') or '')[:2000],
                                               'image_sha256': image['image_sha256'], 'intent_at': intent_at, 'intent_origin_job': job_id,
                                               'last_keep_at': intent_at if payload['keep'] else previous.get('last_keep_at', 0)})
                    plan_changed = True
                elif action in {'approve', 'reject'}:
                    candidate = payload.get('candidate_id')
                    proposal = plan['proposals'].get(candidate)
                    if not proposal or proposal['witness_id'] != payload.get('witness_id') or proposal['evidence_digest'] != payload.get('evidence_digest'):
                        raise HTTPException(409, 'The proposed removal/witness changed. Review the current comparison.')
                    if action == 'approve' and not any(family['confirmed'] and any(row['candidate_id'] == candidate for row in family['proposals'])
                                                       for family in plan['families'].values()):
                        raise HTTPException(409, 'Confirm the family keeper plan before approving a removal.')
                    assign('decisions', candidate, {'status': 'approved' if action == 'approve' else 'rejected', 'witness_id': proposal['witness_id'],
                                                   'evidence_digest': proposal['evidence_digest'], 'note': str(payload.get('note') or '')[:2000]})
                    if action == 'reject':
                        id_index = {image['image_id']: i for i, image in enumerate(images)}
                        assign('rejected', f"{id_index[candidate]}:{id_index[proposal['witness_id']]}", {'value': True})
                        plan_changed = True
                elif action == 'undo_decision':
                    candidate = payload.get('candidate_id')
                    decision = state['decisions'].pop(candidate, None)
                    if decision and decision['status'] == 'rejected':
                        id_index = {image['image_id']: i for i, image in enumerate(images)}
                        key = f"{id_index[candidate]}:{id_index[decision['witness_id']]}"
                        rejected = state['rejected'].get(key)
                        if rejected and rejected['owner'] == decision['owner']: state['rejected'].pop(key)
                        plan_changed = True
                elif action in {'confirm_plan', 'bulk_commit'}:
                    preview = state['previews'].get(payload.get('preview_digest'))
                    if not preview or preview['review_revision'] != state['review_revision'] or preview['plan_revision'] != state['plan_revision'] or preview['selection_revision'] != result['selection_revision'] or preview['budget'] != budget:
                        raise HTTPException(409, 'This preview is stale. Preview the captured family population again.')
                    for key, value in preview['constraints'].items(): assign('constraints', key, {'value': value})
                    updated_plan = self.plan_for(context, state, budget)
                    if updated_plan['plan_digest'] != preview['plan_digest']:
                        raise HTTPException(409, 'Family evidence changed. Preview again before approving.')
                    if action == 'bulk_commit':
                        for candidate, proposal in preview['proposals'].items():
                            actual = updated_plan['proposals'].get(candidate)
                            if not actual or actual['evidence_digest'] != proposal['evidence_digest']:
                                raise HTTPException(409, 'A captured removal changed. No part of this batch was applied.')
                            assign('decisions', candidate, {'status': 'approved', 'witness_id': actual['witness_id'], 'evidence_digest': actual['evidence_digest'], 'note': ''})
                    state['batches'][token] = {'batch_id': token, 'fields': touched, 'revision': revision, 'at': time.time(), 'action': action, 'undone': False}
                    plan_changed = True
                elif action == 'bulk_undo':
                    batch = state['batches'].get(payload.get('batch_id'))
                    if not batch or batch['undone']: raise HTTPException(409, 'This batch cannot be undone again')
                    conflicts = [field for field in batch['fields'] if not state[field['collection']].get(field['key']) or
                                 state[field['collection']][field['key']].get('owner') != batch['batch_id'] or
                                 state[field['collection']][field['key']].get('version') != batch['revision']]
                    if conflicts:
                        raise HTTPException(409, f"This batch has{len(conflicts)}fields changed by later independent edits. No undo changes were applied; review those decisions first.")
                    for field in reversed(batch['fields']):
                        collection, key = field['collection'], field['key']
                        current = state[collection].get(key)
                        if not current or current.get('owner') != batch['batch_id']:
                            conflicts.append({'collection': collection, 'key': key, 'reason': 'A later independent edit owns this field'})
                            continue
                        if collection == 'pins':
                            image = next(image for image in images if image['image_id'] == key)
                            prior = copy.deepcopy(field['before'] or {'value': False, 'note': ''})
                            intent_at = time.time_ns()
                            last_keep_at = self.manual_intent_fields(state).get(key, {}).get('last_keep_at', 0)
                            prior.update(owner=token, version=revision, image_sha256=image['image_sha256'],
                                         intent_at=intent_at, intent_origin_job=job_id,
                                         last_keep_at=intent_at if prior['value'] else last_keep_at)
                            state[collection][key] = prior
                        elif field['before'] is None: state[collection].pop(key)
                        else: state[collection][key] = field['before']
                    batch['undone'] = True
                    plan_changed = True
                else: raise HTTPException(400, 'Unknown review operation')
                state['review_revision'] = revision
                state['plan_revision'] += int(plan_changed)
                state['budget'] = budget
                final = self.plan_for(context, state, budget)
                # Old approvals are retained as superseded evidence, never silently
                # resurrected by later changes or exported against another witness.
                self.supersede_approvals(state, final)
                final = self.plan_for(context, state, budget)
                receipt = {'request_id': token, 'identity': identity, 'revision': revision, 'plan_revision': state['plan_revision'],
                           'action': action, 'batch_id': token if action in {'bulk_commit', 'confirm_plan'} else None,
                           'conflicts': conflicts, 'at': time.time()}
                state['operations'][token] = receipt
                state.setdefault('history', []).append({'request_id': token, 'action': action, 'revision': revision,
                    'at': receipt['at'], 'candidate_id': payload.get('candidate_id'), 'image_id': payload.get('image_id'),
                    'note': str(payload.get('note') or '')[:2000], 'previous_decision': previous_decision,
                    'touched_fields': copy.deepcopy(touched), 'source_digest': source['inventory_digest'],
                    'recipe_digest': final['recipe_digest'], 'plan_digest': final['plan_digest']})
                self.adapter.check_revision(source)
                self.adapter.check_inventory(source, images)
                return {'outcome': 'committed', 'receipt': receipt, 'snapshot': self.review_public(state, final, result)}

    def comparison(self, job_id, payload):
        with self.adapter.guard(), self.lock:
            context = self.review_context(job_id, payload)
            directory, result, source, images, ledger, _, evidence, _ = context
            evidence = SupplementaryEvidence(evidence, ledger.read().get('comparisons'))
            ids = {image['image_id']: i for i, image in enumerate(images)}
            candidate, witness = ids.get(payload.get('candidate_id')), ids.get(payload.get('witness_id'))
            if candidate is None or witness is None or candidate == witness: raise HTTPException(400, 'Select two distinct images')
            edge = evidence.pair(candidate, witness)
            if edge is None: raise HTTPException(409, 'This pair has no computed direct evidence; it cannot authorize a removal')
            try: directed = removal_evidence(candidate, {witness}, [edge])
            except ValueError:
                side = 'left' if edge['left'] == candidate else 'right'
                directed = {**edge, 'reverse': side == 'right', 'valid': False, 'object': edge[side + '_object'], 'combined': edge[side + '_combined']}
            directed.pop('retained_index', None)
            crops = []
            raw_sam = sam_recipe_digest(result['recipe'])
            mask_sidecar = read_sidecar(directory, raw_sam, source_digest=source['inventory_digest']) if raw_sam else None
            for match in edge['matches']:
                left_row, right_row = match['left_row'], match['right_row']
                row_a, row_b = (left_row, right_row) if edge['left'] == candidate else (right_row, left_row)
                crops.append({'candidate_row': row_a, 'witness_row': row_b, 'class_id': match['class_id'],
                              'appearance': match['appearance'], 'scale': match['scale'],
                              'candidate_mask_evidence': object_mask_evidence(mask_sidecar, images[candidate], row_a, sam_used=bool(raw_sam)),
                              'witness_mask_evidence': object_mask_evidence(mask_sidecar, images[witness], row_b, sam_used=bool(raw_sam)),
                              'candidate_url': f'/dataset_pruning/jobs/{job_id}/images/{images[candidate]["image_id"]}/objects/{row_a}',
                              'witness_url': f'/dataset_pruning/jobs/{job_id}/images/{images[witness]["image_id"]}/objects/{row_b}'})
            return {'candidate': self.public_image(images[candidate]), 'witness': self.public_image(images[witness]), 'evidence': directed,
                    'evidence_digest': directed_digest(images[candidate]['image_id'], images[witness]['image_id'], directed, source['inventory_digest'], digest(result['recipe'])), 'crops': crops}

    def recipe_lab(self, job_id):
        from services.dataset_pruning_recipe_lab import metadata
        return metadata(self, job_id)

    def recipe_lab_compare(self, job_id, payload):
        from services.dataset_pruning_recipe_lab import compare
        return compare(self, job_id, payload)

    def recipe_lab_pair(self, job_id, payload):
        from services.dataset_pruning_recipe_lab import add_pair
        return add_pair(self, job_id, payload)

    def recipe_lab_calibrate(self, job_id, payload):
        from services.dataset_pruning_recipe_lab import calibrate
        return calibrate(self, job_id, payload)

    def crop(self, job_id, image_id, label_row):
        directory = self.directory(job_id)
        if self.status(job_id)['status'] != 'completed': raise HTTPException(409, 'Pruning analysis is not ready')
        snapshot = self.completed_snapshots.load(directory)
        index = snapshot.indices.get(image_id)
        image = snapshot.images[index] if index is not None else None
        obj = next((obj for obj in image['objects'] if obj['label_row'] == label_row), None) if image else None
        if obj is None: raise HTTPException(404, 'Object crop not found')
        path = Path(image['image_path'])
        if file_state(path) != image['image_state']: raise HTTPException(409, 'Source image changed')
        target = directory / 'thumbnails' / f'{image_id}-object-{label_row}.jpg'
        if not target.is_file():
            target.parent.mkdir(exist_ok=True)
            with Image.open(path) as loaded:
                box = obj['box']; width, height = loaded.size
                bounds = (max(0, int((box[0]-box[2]/2)*width)), max(0, int((box[1]-box[3]/2)*height)),
                          min(width, int(np.ceil((box[0]+box[2]/2)*width))), min(height, int(np.ceil((box[1]+box[3]/2)*height))))
                preview = loaded.convert('RGB').crop(bounds); preview.thumbnail((320, 320))
                temp = target.with_name('.' + uuid.uuid4().hex + '.jpg')
                try:
                    preview.save(temp, 'JPEG', quality=90)
                    if file_state(path) != image['image_state']: raise HTTPException(409, 'Source image changed')
                    os.replace(temp, target)
                finally: temp.unlink(missing_ok=True)
        return FileResponse(target, media_type='image/jpeg')

    def thumbnail(self,job_id,image_id):
        directory = self.directory(job_id)
        if self.status(job_id)['status'] != 'completed': raise HTTPException(409, 'Pruning analysis is not ready')
        snapshot = self.completed_snapshots.load(directory)
        index = snapshot.indices.get(image_id)
        image = snapshot.images[index] if index is not None else None
        if image is None: raise HTTPException(404,"Image not found")
        path = Path(image["image_path"])
        if file_state(path) != image["image_state"]: raise HTTPException(409,"Source image changed")
        target = directory/"thumbnails"/(image_id+".jpg")
        if not target.exists():
            target.parent.mkdir(exist_ok=True)
            with Image.open(path) as im:
                preview = im.convert("RGB")
                preview.thumbnail((960,960))
                temp=target.with_name("."+uuid.uuid4().hex+".jpg")
                try:
                    preview.save(temp,"JPEG",quality=85)
                    if file_state(path)!=image["image_state"]: raise HTTPException(409,"Source image changed")
                    os.replace(temp,target)
                finally: temp.unlink(missing_ok=True)
        return FileResponse(target,media_type="image/jpeg")

    def export(self,job_id,payload):
        if payload.get('preview_id'): raise HTTPException(409, 'Apply and review a planning preview before exporting its approved removals')
        prepared = self.prepare_active_export(job_id, payload)
        with self.adapter.guard(),self.lock:
            context = self.review_context(job_id, payload)
            directory, result, source, images, ledger, _, evidence, config = context
            review = ledger.read()
            if payload.get('removed_count', 0) != review.get('budget', payload.get('removed_count', 0)):
                raise HTTPException(409, 'The removal ceiling changed. Refresh the committed plan before exporting.')
            if payload.get('expected_review_revision') != review['review_revision'] or payload.get('plan_revision') != review['plan_revision']:
                raise HTTPException(409, 'Approvals or family plans changed. Refresh before preparing this variant.')
            authoritative = self.plan_for(context, review, payload.get('removed_count', 0))
            removed = authoritative['approved_indices']
            if not removed: raise HTTPException(400, 'Approve at least one exact proposed removal before exporting')
            retained = sorted(set(range(len(images))) - set(removed))
            if prepared:
                self._check_capture(prepared['context'], prepared['capture'])
                if prepared['plan']['plan_digest'] != authoritative['plan_digest']:
                    raise HTTPException(409, 'The approved export plan changed')
                chosen = {'removed': removed, 'retained': retained, **prepared['metrics'],
                          'planning_settings': authoritative['settings'], 'accounting_digest': authoritative.get('accounting_digest'),
                          'split_group_digest': digest(prepared['groups']),
                          'split_group_policy': 'preserve_existing_groups_and_recorded_scene_video_relations_v1'}
            else:
                chosen = {'removed': removed, 'retained': retained,
                          **coverage_for_retained(images, retained, FilteredEvidence(evidence, review['rejected'], review.get('comparisons', {})), config)}
            chosen.update(review_revision=review['review_revision'], plan_revision=review['plan_revision'],
                          plan_digest=authoritative['plan_digest'], split_policy=payload.get('split_policy', 'grouped'),
                          approval_evidence=[copy.deepcopy(authoritative['proposals'][images[i]['image_id']]) for i in removed])
            result = {**result, 'keep_ids': authoritative['order_result']['keep_ids'],
                      'protection_reasons': authoritative['order_result']['protection_reasons']}
            if prepared: result['groups'] = prepared['groups']
            token=payload.get("request_id")
            receipt_path=directory/"export_requests"/(str(token)+".json") if token else None
            request_identity=digest(payload)
            if receipt_path and receipt_path.is_file():
                receipt=read_json(receipt_path)
                if receipt["identity"]!=request_identity: raise HTTPException(409,"Export request identity belongs to another variant")
                status=self.export_status(job_id,receipt["export_id"])
                if status["status"] not in {"interrupted","removed","failed"}: return status
            retained=set(chosen["retained"])
            try:
                if not prepared:
                    for index in chosen["removed"]: removal_evidence(index,retained,PairStore(directory/"evidence.sqlite3"))
            except (ValueError,KeyError,sqlite3.Error) as exc:
                raise HTTPException(409,"Saved matching evidence is incomplete. Analyze again; cached descriptors will be reused") from exc
            identity_fields = {'selection_revision': result['selection_revision'], 'review_revision': review['review_revision'],
                               'plan_revision': review['plan_revision'], 'plan_digest': authoritative['plan_digest'],
                               'removed': chosen['removed'], 'source': source['inventory_digest'], 'split_policy': chosen['split_policy']}
            if prepared: identity_fields['split_groups'] = chosen['split_group_digest']
            identity = digest(identity_fields)[:20]
            export_dir = directory/"exports"/identity
            export_dir.mkdir(parents=True,exist_ok=True)
            state_path = export_dir/"state.json"
            if state_path.exists():
                existing = read_json(state_path)
                if (existing["status"] == "completed" and (export_dir/"dataset.zip").is_file()) or (job_id,identity) in self.events:
                    if receipt_path: write_json(receipt_path,{"identity":request_identity,"export_id":identity})
                    return existing
            state = {"export_id":identity,"job_id":job_id,"status":"queued","progress":0,
                     "selection_revision":result["selection_revision"],"removed_count":len(chosen["removed"]),
                     'review_revision': review['review_revision'], 'plan_revision': review['plan_revision'], 'plan_digest': authoritative['plan_digest']}
            write_json(state_path,state)
            if receipt_path: write_json(receipt_path,{"identity":request_identity,"export_id":identity})
            self.events[(job_id,identity)] = threading.Event()
            threading.Thread(target=self.run_export,args=(export_dir,result,source,images,chosen),daemon=True).start()
            return state

    def export_directory(self,job_id,export_id):
        if not re.fullmatch(r"[a-f0-9]{20}",str(export_id)): raise HTTPException(404,"Export not found")
        directory = self.directory(job_id)/"exports"/export_id
        if not (directory/"state.json").is_file(): raise HTTPException(404,"Export not found")
        return directory

    def export_status(self,job_id,export_id):
        state = read_json(self.export_directory(job_id,export_id)/"state.json")
        try: review = self.export_review_state(job_id)
        except HTTPException as error:
            state.update(status='stale', message=str(error.detail))
            return state
        if state.get('review_revision') != review['review_revision'] or state.get('plan_revision') != review['plan_revision']:
            state.update(status='stale', message='Approvals or keepers changed. Prepare the currently approved variant again.')
        if state["status"] == "completed" and not (self.export_directory(job_id,export_id)/"dataset.zip").is_file():
            state.update(status="removed",message="The prepared export was removed. Prepare the variant again.")
        if state["status"] in {"queued","running"} and (job_id,export_id) not in self.events:
            state.update(status="interrupted",message="Export interrupted by backend restart. Prepare it again.")
        return state

    def export_review_state(self, job_id):
        # A Keep committed in another analysis is authoritative even when this
        # job has not been opened since its ZIP was prepared. Reconcile the
        # cheap durable candidate reduction; export admission needs no replan.
        with self.adapter.guard(), self.lock:
            context = self.review_context(job_id, {'removed_count': 0}, False)
            state = context[4].read()
            active = self._load_active_plan(context, state)
            if active and active.get('accounting_outdated'):
                raise HTTPException(409, 'Direct coverage qualification changed. Apply and review a new cached preview before exporting.')
            return state

    def download(self,job_id,export_id):
        directory = self.export_directory(job_id,export_id)
        def validate():
            if self.export_status(job_id,export_id)['status'] != 'completed': raise HTTPException(409,'Dataset variant is not ready')
            job = self.directory(job_id)
            source = read_json(job / 'source.json')
            self.adapter.check_revision(source)
            self.adapter.check_inventory(source, read_json(job / 'images.json'))
        with self.adapter.guard(): validate()
        return PruningDownloadResponse(directory/'dataset.zip', media_type='application/zip', filename='pruned_dataset.zip',
                                       adapter=self.adapter, validate=validate)

    def run_export(self,directory,result,source,images,chosen):
        state = read_json(directory/"state.json")
        def update(**values):
            state.update(values)
            write_json(directory/"state.json",state)
        try:
            with self.adapter.guard():
                self.adapter.check_revision(source)
                self.adapter.check_inventory(source,images)
                splits = grouped_splits(images,chosen["retained"],result["groups"])
                if chosen.get('split_policy') == 'preserve':
                    assignments = {str(i): images[i]['split'] if images[i]['split'] in {'train', 'val', 'test'} else 'train' for i in chosen['retained']}
                    group_splits = {}
                    for i in chosen['retained']: group_splits.setdefault(result['groups'][i], set()).add(assignments[str(i)])
                    crossing = sum(len(values) > 1 for values in group_splits.values())
                    splits = {'assignments': assignments, 'train_count': sum(value == 'train' for value in assignments.values()),
                              'val_count': sum(value == 'val' for value in assignments.values()), 'test_count': sum(value == 'test' for value in assignments.values()),
                              'group_count': len(group_splits), 'method': 'preserve_source_splits',
                              'warnings': [f'{crossing} near-duplicate groups still cross source splits'] if crossing else []}
                # Disk and stream operations use the existing export machinery.
                plan, retained_manifest,caption_records = [], [],[]
                for i in chosen["retained"]:
                    image = images[i]
                    path = Path(image["image_path"])
                    if file_state(path) != image["image_state"]: raise ValueError("Source image changed; analyze again before exporting")
                    split = splits["assignments"][str(i)]
                    # Stable IDs prevent equal filenames, stems and extensions
                    # from colliding after the grouped split rebuild.
                    filename = image["image_id"]+path.suffix.lower()
                    base = image["image_id"]+".txt"
                    plan.append({"arcname":f"{split}/images/{filename}","path":path,"data":None,"state":self.adapter.export_file_state(path)})
                    plan.append({"arcname":f"{split}/labels/{base}","data":("\n".join(image["label_lines"])+("\n" if image["label_lines"] else "")).encode()})
                    if image.get("text_label"):
                        plan.append({"arcname":f"{split}/text_labels/{base}","data":image["text_label"].encode()})
                    for caption in image.get("caption_records",[]):
                        caption_records.append({**caption,"image_name":filename,"image":filename,"image_key":f"{split}/{filename}","split":split})
                    retained_manifest.append({"image_id":image["image_id"],"source_split":image["split"],"source_path":image["image_relpath"],
                                              "output_path":f"{split}/images/{filename}","sha256":image["image_sha256"],"duplicate_group":result["groups"][i]})
                if caption_records:
                    plan.append({"arcname":"text_labels/captions.jsonl","data":("\n".join(json.dumps(c,ensure_ascii=False) for c in caption_records)+"\n").encode()})
                names = source["labelmap"]
                yaml = "path: .\ntrain: train/images\nval: "+("val/images" if splits["val_count"] else "null")+"\nnc: "+str(len(names))+"\nnames: "+json.dumps(names,ensure_ascii=False)+"\n"
                if splits.get('test_count'): yaml += 'test: test/images\n'
                manifest = {"version":VERSION,"source_identity":source["identity"],"source_revision":source["revision"],
                            "inventory_digest":source["inventory_digest"],"feature_recipe":result["recipe"],"selection_revision":result["selection_revision"],
                            "coverage":{k:v for k,v in chosen.items() if k not in {"retained","removed"}},"thresholds":result["thresholds"],
                            "similarity_recipe":result["similarity_recipe"],
                            "keep_ids":result["keep_ids"],"retained":retained_manifest,"removed":[images[i]["image_id"] for i in chosen["removed"]],
                            "protected":[{"image_id":images[int(i)]["image_id"],"reasons":r} for i,r in result["protection_reasons"].items()],
                            "splits":{k:v for k,v in splits.items() if k!="assignments"},"coverage_note":result["coverage_note"]}
                manifest.update(review_revision=chosen['review_revision'], plan_revision=chosen['plan_revision'], plan_digest=chosen['plan_digest'],
                                approvals=chosen['approval_evidence'], split_policy=chosen['split_policy'], unreviewed_policy='retained')
                for name,data in [("data.yaml",yaml.encode()),("labelmap.txt",("\n".join(names)+"\n").encode()),("pruning_manifest.json",json.dumps(manifest,indent=2).encode())]:
                    plan.append({"arcname":name,"data":data})
                self.adapter.disk_capacity(directory,plan)
                update(status="running",message="Freezing original images and current annotations",progress=.05)
                frozen = self.adapter.freeze_export(plan,directory/"snapshot",lambda done,total:update(progress=.05+.5*done/max(1,total)))
                # Hashes bind the copied bytes, not just filesystem timestamps.
                frozen_by_name = {p["arcname"]:p for p in frozen}
                for item in retained_manifest:
                    if frozen_by_name[item["output_path"]]["content_sha256"] != item["sha256"]: raise ValueError("Source image bytes changed during export")
                self.adapter.check_revision(source)
                self.adapter.check_inventory(source,images)
                current_review = self.export_review_state(state['job_id'])
                if current_review['review_revision'] != chosen['review_revision'] or current_review['plan_revision'] != chosen['plan_revision']:
                    raise ValueError('Review changed while preparing this variant. Prepare the current approvals again.')
                update(message="Writing complete dataset variant",progress=.55)
                self.adapter.write_export(frozen,directory/"dataset.zip",lambda done,total,*args:update(progress=.55+.45*done/max(1,total)))
                self.adapter.check_revision(source)
                self.adapter.check_inventory(source, images)
                with self.lock:
                    final_review = self.export_review_state(state['job_id'])
                    if final_review['review_revision'] != chosen['review_revision'] or final_review['plan_revision'] != chosen['plan_revision']:
                        raise ValueError('Review changed while writing this variant. Prepare the current approvals again.')
                    self.adapter.register_export(directory/"dataset.zip")
                    update(status="completed",message="Dataset variant ready",progress=1.0,splits=manifest["splits"],bytes=(directory/"dataset.zip").stat().st_size)
        except Exception as exc:
            (directory/"dataset.zip").unlink(missing_ok=True)
            update(status="failed",message=str(getattr(exc,"detail",exc)))
        finally:
            shutil.rmtree(directory/"snapshot",ignore_errors=True)
            self.events.pop((state["job_id"],state["export_id"]),None)
