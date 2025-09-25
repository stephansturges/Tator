from functools import lru_cache
from typing import List, Optional

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    # CLIP
    clip_model_name: str = Field("ViT-B/32", env="CLIP_MODEL_NAME")

    # Logistic regression path
    logreg_path: str = Field("./my_logreg_model.pkl", env="LOGREG_PATH")

    # SAM
    sam_model_type: str = Field("vit_h", env="SAM_MODEL_TYPE")
    sam_checkpoint_path: str = Field("./sam_vit_h_4b8939.pth", env="SAM_CHECKPOINT_PATH")
    sam_variant: str = Field("sam1", env="SAM_VARIANT")
    sam2_config_path: Optional[str] = Field(None, env="SAM2_CONFIG_PATH")
    sam2_checkpoint_path: Optional[str] = Field(None, env="SAM2_CHECKPOINT_PATH")

    # Runtime
    force_device: Optional[str] = Field(None, env="FORCE_DEVICE")  # e.g., "cpu" or "cuda"

    # API
    cors_allow_origins: str = Field("*", env="CORS_ALLOW_ORIGINS")  # comma-separated
    max_body_size_mb: Optional[int] = Field(None, env="MAX_BODY_SIZE_MB")

    # Observability
    enable_metrics: bool = Field(False, env="ENABLE_METRICS")

    @validator("cors_allow_origins")
    def _strip(cls, v: str) -> str:
        return v.strip()

    def origins_list(self) -> List[str]:
        raw = self.cors_allow_origins
        if raw == "*":
            return ["*"]
        return [x.strip() for x in raw.split(",") if x.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
