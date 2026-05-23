from collections import deque
import threading

from services.qwen_runtime import _ensure_qwen_ready_for_caption_impl


class _CudaUnavailable:
    @staticmethod
    def is_available():
        return False


class _TorchStub:
    cuda = _CudaUnavailable()
    float16 = "float16"
    float32 = "float32"


class _ModelStub:
    def __init__(self, model_id):
        self.model_id = model_id
        self.eval_called = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True


class _ProcessorStub:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return {"model_id": model_id, "kwargs": kwargs}


class _LoggerStub:
    def warning(self, *args, **kwargs):
        return None


def _load_caption_runtime_with_fallback(cache_limit):
    state = {
        "qwen_caption_cache": {},
        "qwen_caption_order": deque(),
        "qwen_device": None,
        "qwen_last_error": None,
    }
    evicted = []

    def load_model(model_id, load_kwargs, *, local_files_only):
        if model_id == "Qwen/test-GPTQ-Int4":
            raise OSError("primary missing")
        return _ModelStub(model_id)

    model, processor = _ensure_qwen_ready_for_caption_impl(
        "Qwen/test-GPTQ-Int4",
        state=state,
        qwen_lock=threading.Lock(),
        import_error=None,
        qwen_model_cls=object(),
        qwen_processor_cls=_ProcessorStub,
        process_vision_info=object(),
        packaging_version=None,
        min_transformers="0",
        resolve_device_fn=lambda: "cpu",
        device_pref="cpu",
        torch_module=_TorchStub(),
        load_qwen_model_fn=load_model,
        hf_offline_enabled_fn=lambda: False,
        set_hf_offline_fn=lambda enabled: None,
        enable_hf_offline_defaults_fn=lambda: None,
        strip_model_suffix_fn=lambda model_id: model_id.removesuffix("-GPTQ-Int4"),
        format_load_error_fn=str,
        min_pixels=1,
        max_pixels=2,
        caption_cache_limit=cache_limit,
        evict_entry_fn=lambda key, entry: evicted.append((key, entry)),
        logger=_LoggerStub(),
    )
    return state, evicted, model, processor


def test_caption_runtime_fallback_cache_keeps_single_lru_entry():
    state, evicted, model, processor = _load_caption_runtime_with_fallback(cache_limit=1)

    assert model.model_id == "Qwen/test"
    assert processor["model_id"] == "Qwen/test"
    assert list(state["qwen_caption_order"]) == ["caption:Qwen/test-GPTQ-Int4:cpu"]
    assert list(state["qwen_caption_cache"]) == ["caption:Qwen/test-GPTQ-Int4:cpu"]
    assert evicted == []


def test_caption_runtime_fallback_respects_disabled_cache():
    state, evicted, model, _processor = _load_caption_runtime_with_fallback(cache_limit=0)

    assert model.model_id == "Qwen/test"
    assert list(state["qwen_caption_order"]) == []
    assert state["qwen_caption_cache"] == {}
    assert evicted == []
