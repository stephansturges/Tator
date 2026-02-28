from __future__ import annotations

import asyncio
import base64
import io
import zipfile

from fastapi import HTTPException
from PIL import Image

import localinferenceapi as api
from models.schemas import BboxModel, CropImage, CropZipRequest


def _png_b64(size: int = 8) -> str:
    img = Image.new("RGB", (size, size), color=(200, 10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def _stream_body(resp) -> bytes:
    chunks = []
    async for chunk in resp.body_iterator:
        chunks.append(chunk)
    return b"".join(chunks)


def test_crop_zip_sanitizes_archive_member_names() -> None:
    init = api.crop_zip_init()
    job_id = init["jobId"]
    payload = CropZipRequest(
        images=[
            CropImage(
                image_base64=_png_b64(),
                originalName="../../unsafe/..\\evil name.png",
                bboxes=[BboxModel(className="../veh\\icle", x=0, y=0, width=6, height=6)],
            )
        ]
    )
    api.crop_zip_chunk(payload, jobId=job_id)
    response = api.crop_zip_finalize(job_id)
    raw_zip = asyncio.run(_stream_body(response))
    with zipfile.ZipFile(io.BytesIO(raw_zip), "r") as zf:
        names = zf.namelist()
    assert len(names) == 1
    member = names[0]
    assert ".." not in member
    assert "/" not in member
    assert "\\" not in member
    assert member.endswith(".jpg")


def test_crop_zip_invalid_base64_raises_400_and_cleans_job() -> None:
    init = api.crop_zip_init()
    job_id = init["jobId"]
    payload = CropZipRequest(
        images=[
            CropImage(
                image_base64="not-base64",
                originalName="test.png",
                bboxes=[BboxModel(className="car", x=0, y=0, width=1, height=1)],
            )
        ]
    )
    api.crop_zip_chunk(payload, jobId=job_id)
    try:
        api.crop_zip_finalize(job_id)
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 400
        assert str(exc.detail).startswith("invalid_base64")
    assert job_id not in api.job_store
