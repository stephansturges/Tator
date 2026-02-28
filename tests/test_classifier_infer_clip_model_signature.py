from services.classifier import _infer_clip_model_from_embedding_dim_impl


def test_infer_clip_model_accepts_positional_active_name():
    # Regression guard: callers pass active clip model as positional arg.
    assert _infer_clip_model_from_embedding_dim_impl(512, "ViT-B/16") == "ViT-B/16"
    assert _infer_clip_model_from_embedding_dim_impl(768, "ViT-B/16") == "ViT-L/14"

