"""Load registered MLX SAM3 weights into one private hosted worker."""
from pathlib import Path

from services.hosted.config import file_digest, regular_file


def build_predictor(definition):
    # The source is the installed, immutable package, never whichever checkpoint
    # happens to be active in the legacy server or the Hugging Face cache.
    directory = Path(definition["model_path"]).resolve(strict=True)
    for name, expected in definition["files"].items():
        path = regular_file(directory / name, [directory])
        if file_digest(path) != expected:
            raise ValueError("hosted_model_artifact_changed")
    import mlx.core as mx
    from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
    from mlx_vlm.utils import load_model
    from services.mlx_sam3 import MlxSam3PredictorAdapter

    model = load_model(directory, lazy=False, strict=True)
    model.eval()
    mx.eval(model.parameters())
    processor = Sam3Processor.from_pretrained(str(directory), local_files_only=True)
    return MlxSam3PredictorAdapter(model, processor, model_path=directory,
        runtime_notice="Apple Silicon MLX SAM3: point and box prompts. Text and find-similar prompts need native SAM3.")
