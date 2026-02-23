"""APIRouter for agent mining endpoints."""


from typing import Any, Callable, Optional, Type

from fastapi import APIRouter, File, UploadFile


def build_agent_mining_router(
    *,
    start_job_fn: Callable[[Any], Any],
    list_jobs_fn: Callable[[], Any],
    get_job_fn: Callable[[str], Any],
    cancel_job_fn: Callable[[str], Any],
    latest_result_fn: Callable[[], Any],
    cache_size_fn: Callable[[], Any],
    cache_purge_fn: Callable[[], Any],
    apply_image_fn: Callable[[Any], Any],
    apply_chain_fn: Callable[[Any], Any],
    save_recipe_fn: Callable[[Any], Any],
    list_recipes_fn: Callable[[Optional[str]], Any],
    get_recipe_fn: Callable[[str], Any],
    export_recipe_fn: Callable[[str], Any],
    import_recipe_fn: Callable[[UploadFile], Any],
    delete_recipe_fn: Callable[[str], Any],
    save_cascade_fn: Callable[[Any], Any],
    list_cascades_fn: Callable[[], Any],
    get_cascade_fn: Callable[[str], Any],
    delete_cascade_fn: Callable[[str], Any],
    export_cascade_fn: Callable[[str], Any],
    import_cascade_fn: Callable[[UploadFile], Any],
    job_request_cls: Type[Any],
    apply_image_request_cls: Type[Any],
    apply_chain_request_cls: Type[Any],
    recipe_request_cls: Type[Any],
    cascade_request_cls: Type[Any],
    sam3_response_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.post("/agent_mining/jobs")
    def start_agent_mining_job(payload: job_request_cls):  # type: ignore[valid-type]
        return start_job_fn(payload)

    @router.get("/agent_mining/jobs")
    def list_agent_mining_jobs():
        return list_jobs_fn()

    @router.get("/agent_mining/jobs/{job_id}")
    def get_agent_mining_job(job_id: str):
        return get_job_fn(job_id)

    @router.post("/agent_mining/jobs/{job_id}/cancel")
    def cancel_agent_mining_job(job_id: str):
        return cancel_job_fn(job_id)

    @router.get("/agent_mining/results/latest")
    def get_latest_agent_mining_result():
        return latest_result_fn()

    @router.get("/agent_mining/cache_size")
    def agent_mining_cache_size():
        return cache_size_fn()

    @router.post("/agent_mining/cache/purge")
    def agent_mining_cache_purge():
        return cache_purge_fn()

    @router.post("/agent_mining/apply_image", response_model=sam3_response_cls)
    def agent_mining_apply_image(payload: apply_image_request_cls):  # type: ignore[valid-type]
        return apply_image_fn(payload)

    @router.post("/agent_mining/apply_image_chain", response_model=sam3_response_cls)
    def agent_mining_apply_image_chain(payload: apply_chain_request_cls):  # type: ignore[valid-type]
        return apply_chain_fn(payload)

    @router.post("/agent_mining/recipes", response_model=dict)
    def agent_mining_save_recipe(payload: recipe_request_cls):  # type: ignore[valid-type]
        return save_recipe_fn(payload)

    @router.get("/agent_mining/recipes", response_model=list)
    def agent_mining_list_recipes(dataset_id: Optional[str] = None):
        return list_recipes_fn(dataset_id)

    @router.get("/agent_mining/recipes/{recipe_id}", response_model=dict)
    def agent_mining_get_recipe(recipe_id: str):
        return get_recipe_fn(recipe_id)

    @router.get("/agent_mining/recipes/{recipe_id}/export")
    def agent_mining_export_recipe(recipe_id: str):
        return export_recipe_fn(recipe_id)

    @router.post("/agent_mining/recipes/import", response_model=dict)
    async def agent_mining_import_recipe(file: UploadFile = File(...)):
        return await import_recipe_fn(file)

    @router.delete("/agent_mining/recipes/{recipe_id}")
    def agent_mining_delete_recipe(recipe_id: str):
        return delete_recipe_fn(recipe_id)

    @router.post("/agent_mining/cascades", response_model=dict)
    def agent_mining_save_cascade(payload: cascade_request_cls):  # type: ignore[valid-type]
        return save_cascade_fn(payload)

    @router.get("/agent_mining/cascades", response_model=list)
    def agent_mining_list_cascades():
        return list_cascades_fn()

    @router.get("/agent_mining/cascades/{cascade_id}", response_model=dict)
    def agent_mining_get_cascade(cascade_id: str):
        return get_cascade_fn(cascade_id)

    @router.delete("/agent_mining/cascades/{cascade_id}")
    def agent_mining_delete_cascade(cascade_id: str):
        return delete_cascade_fn(cascade_id)

    @router.get("/agent_mining/cascades/{cascade_id}/export")
    def agent_mining_export_cascade(cascade_id: str):
        return export_cascade_fn(cascade_id)

    @router.post("/agent_mining/cascades/import", response_model=dict)
    async def agent_mining_import_cascade(file: UploadFile = File(...)):
        return await import_cascade_fn(file)

    return router
