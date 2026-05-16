import pytest
from fastapi import HTTPException

import localinferenceapi as api
from models.schemas import SamPreloadRequest


def test_sam_preload_rejects_disabled_background_slot():
    original_capacity = api.predictor_manager.get_capacity()
    api.predictor_manager.set_capacity(1)
    try:
        payload = SamPreloadRequest(
            image_token="missing-token",
            image_name="next.jpg",
            slot="next",
            sam_variant="sam1",
        )
        with pytest.raises(HTTPException) as exc:
            api.sam_preload(payload)
        assert exc.value.status_code == 409
        assert "slot_disabled:next" in str(exc.value.detail)
    finally:
        api.predictor_manager.set_capacity(original_capacity)
