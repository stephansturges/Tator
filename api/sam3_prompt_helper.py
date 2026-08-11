"""APIRouter for SAM3 prompt helper endpoints."""


from typing import Any, Callable, Type

from fastapi import APIRouter, Form


def build_sam3_prompt_helper_router(
    *,
    suggest_fn: Callable[[Any], Any],
    expand_fn: Callable[[Any], Any],
    create_job_fn: Callable[[Any], Any],
    search_fn: Callable[[Any], Any],
    recipe_fn: Callable[[Any], Any],
    list_presets_fn: Callable[[], Any],
    get_preset_fn: Callable[[str], Any],
    save_preset_fn: Callable[[str, str, str], Any],
    list_jobs_fn: Callable[[], Any],
    get_job_fn: Callable[[str], Any],
    request_suggest_cls: Type[Any],
    request_expand_cls: Type[Any],
    request_job_cls: Type[Any],
    request_search_cls: Type[Any],
    request_recipe_cls: Type[Any],
) -> APIRouter:
    router = APIRouter()

    @router.post("/sam3/prompt_helper/suggest")
    def sam3_prompt_suggest(payload: request_suggest_cls):  # type: ignore[valid-type]
        return suggest_fn(payload)

    @router.post("/sam3/prompt_helper/expand")
    def sam3_prompt_expand(payload: request_expand_cls):  # type: ignore[valid-type]
        return expand_fn(payload)

    @router.post("/sam3/prompt_helper/jobs")
    def sam3_prompt_helper_jobs(payload: request_job_cls):  # type: ignore[valid-type]
        return create_job_fn(payload)

    @router.post("/sam3/prompt_helper/search")
    def sam3_prompt_helper_search(payload: request_search_cls):  # type: ignore[valid-type]
        return search_fn(payload)

    @router.post("/sam3/prompt_helper/recipe")
    def sam3_prompt_helper_recipe(payload: request_recipe_cls):  # type: ignore[valid-type]
        return recipe_fn(payload)

    @router.get("/sam3/prompt_helper/presets")
    def list_prompt_helper_presets():
        return list_presets_fn()

    @router.get("/sam3/prompt_helper/presets/{preset_id}")
    def get_prompt_helper_preset(preset_id: str):
        return get_preset_fn(preset_id)

    @router.post("/sam3/prompt_helper/presets")
    def save_prompt_helper_preset(
        dataset_id: str = Form(...),
        label: str = Form(""),
        prompts_json: str = Form(...),
    ):
        return save_preset_fn(dataset_id, label, prompts_json)

    @router.get("/sam3/prompt_helper/jobs")
    def list_prompt_helper_jobs():
        return list_jobs_fn()

    @router.get("/sam3/prompt_helper/jobs/{job_id}")
    def get_prompt_helper_job(job_id: str):
        return get_job_fn(job_id)

    return router
