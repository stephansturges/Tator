import base64, hashlib, io, zipfile, math, uuid
import numpy as np
from typing import Optional, List, Dict
import torch, clip, joblib
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from segment_anything import sam_model_registry, SamPredictor

# 1) Add threading for the lock
import threading

job_store: Dict[str, List["CropImage"]] = {}
MODEL_PATH = "./my_logreg_model.pkl"
device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("Loading CLIP model...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
print("Loading logistic regression...")
clf = joblib.load(MODEL_PATH)

MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"
sam_model = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
predictor = SamPredictor(sam_model)

# Cache for repeated calls
last_image_hash = None
last_image_np = None

# 2) Create a lock for synchronizing set_image calls
sam_lock = threading.Lock()

def set_image_if_needed(np_img: np.ndarray):
    global last_image_hash, last_image_np
    new_hash = hashlib.md5(np_img.tobytes()).hexdigest()
    with sam_lock:  # 3) Acquire lock before changing global state
        if new_hash != last_image_hash:
            predictor.set_image(np_img)
            last_image_np = np_img
            last_image_hash = new_hash

class Base64Payload(BaseModel):
    image_base64: str
    uuid: Optional[str] = None

class PredictResponse(BaseModel):
    prediction: str
    uuid: Optional[str] = None

class BboxModel(BaseModel):
    className: str
    x: float
    y: float
    width: float
    height: float

class CropImage(BaseModel):
    image_base64: str
    originalName: str
    bboxes: List[BboxModel]

class CropZipRequest(BaseModel):
    images: List[CropImage]

class PointPrompt(BaseModel):
    image_base64: str
    point_x: float
    point_y: float
    uuid: Optional[str] = None

class BboxPrompt(BaseModel):
    image_base64: str
    bbox_left: float
    bbox_top: float
    bbox_width: float
    bbox_height: float
    uuid: Optional[str] = None

class YoloBboxOutput(BaseModel):
    class_id: str
    bbox: List[float]
    uuid: Optional[str] = None

class YoloBboxClassOutput(BaseModel):
    class_id: int
    bbox: List[float]
    uuid: Optional[str] = None

def mask_to_bounding_box(mask: np.ndarray) -> tuple[int,int,int,int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return (0,0,0,0)
    y_min,y_max = np.where(rows)[0][[0,-1]]
    x_min,x_max = np.where(cols)[0][[0,-1]]
    return (int(x_min), int(y_min), int(x_max), int(y_max))

def to_yolo(w: int, h: int, left: int, top: int, right: int, bottom: int) -> List[float]:
    w_abs = float(right - left)
    h_abs = float(bottom - top)
    cx_abs = left + w_abs/2
    cy_abs = top + h_abs/2
    cx = cx_abs / w
    cy = cy_abs / h
    ww = w_abs / w
    hh = h_abs / h
    return [cx, cy, ww, hh]

@app.post("/predict_base64", response_model=PredictResponse)
def predict_base64(payload: Base64Payload):
    data = base64.b64decode(payload.image_base64)
    pil_img = Image.open(BytesIO(data)).convert("RGB")
    inp = clip_preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = clip_model.encode_image(inp)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    feats_np = feats.squeeze(0).cpu().numpy().reshape(1, -1)
    pred_cls = clf.predict(feats_np)[0]
    return PredictResponse(prediction=pred_cls, uuid=payload.uuid)



@app.post("/predict_crop", response_model=PredictResponse)
def predict_crop(file: UploadFile = File(...), x: int = Form(...), y: int = Form(...),
                 w: int = Form(...), h: int = Form(...)):
    image_bytes = file.file.read()
    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    cropped = pil_img.crop((x, y, x+w, y+h))
    inp = clip_preprocess(cropped).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = clip_model.encode_image(inp)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    feats_np = feats.squeeze(0).cpu().numpy().reshape(1, -1)
    pred_cls = clf.predict(feats_np)[0]
    return PredictResponse(prediction=pred_cls, uuid=None)

@app.post("/sam2_point", response_model=YoloBboxOutput)
def sam2_point(prompt: PointPrompt):
    data = base64.b64decode(prompt.image_base64)
    pil_img = Image.open(BytesIO(data)).convert("RGB")
    np_img = np.array(pil_img)
    set_image_if_needed(np_img)
    coords = np.array([[prompt.point_x, prompt.point_y]])
    labels = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=coords,
        point_labels=labels,
        multimask_output=False
    )
    mask = masks[0]
    left, top, right, bottom = mask_to_bounding_box(mask)
    yolo_box = to_yolo(pil_img.width, pil_img.height, left, top, right, bottom)
    return YoloBboxOutput(class_id="0", bbox=yolo_box, uuid=prompt.uuid)

class SamPointAutoResponse(BaseModel):
    prediction: str
    bbox: List[float]
    uuid: Optional[str] = None
@app.post("/sam2_bbox_auto", response_model=SamPointAutoResponse)
def sam2_bbox_auto(prompt: BboxPrompt):
    data = base64.b64decode(prompt.image_base64)
    pil_img = Image.open(BytesIO(data)).convert("RGB")
    np_img = np.array(pil_img)

    # Ensure the SAM predictor has the correct image
    set_image_if_needed(np_img)

    full_h, full_w = pil_img.height, pil_img.width

    # Build the bounding box from the prompt
    left = max(0, prompt.bbox_left)
    top = max(0, prompt.bbox_top)
    right = min(full_w, left + prompt.bbox_width)
    bottom = min(full_h, top + prompt.bbox_height)

    # If it's clearly invalid, return "unknown"
    if right <= left or bottom <= top:
        return SamPointAutoResponse(
            prediction="unknown",
            bbox=[0, 0, 0, 0],
            uuid=prompt.uuid
        )

    # Use the bounding box to get a mask from the SAM predictor
    sub_box = np.array([left, top, right, bottom], dtype=np.float32)
    masks, _, _ = predictor.predict(box=sub_box, multimask_output=False)

    # Convert the mask to a bounding box, then to YOLO format
    mask = masks[0]
    x_min, y_min, x_max, y_max = mask_to_bounding_box(mask)
    yolo_box = to_yolo(full_w, full_h, x_min, y_min, x_max, y_max)

    # Clamp and check
    gx_min_i = max(0, int(x_min))
    gy_min_i = max(0, int(y_min))
    gx_max_i = min(full_w, int(x_max))
    gy_max_i = min(full_h, int(y_max))

    # If the mask bounding box is invalid, label as "unknown"
    if gx_max_i <= gx_min_i or gy_max_i <= gy_min_i:
        return SamPointAutoResponse(
            prediction="unknown",
            bbox=yolo_box,
            uuid=prompt.uuid
        )

    # Crop for CLIP classification
    subarr = np_img[gy_min_i:gy_max_i, gx_min_i:gx_max_i, :]
    final_pil = Image.fromarray(subarr)

    # Same CLIP logic as sam2_point_auto
    inp = clip_preprocess(final_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = clip_model.encode_image(inp)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    feats_np = feats.squeeze(0).cpu().numpy().reshape(1, -1)

    # Classify
    pred_cls = clf.predict(feats_np)[0]

    # Return as string, just like in sam2_point_auto
    return SamPointAutoResponse(
        prediction=str(pred_cls),
        bbox=yolo_box,
        uuid=prompt.uuid
    )

@app.post("/sam2_bbox", response_model=YoloBboxOutput)
def sam2_bbox(prompt: BboxPrompt):
    data = base64.b64decode(prompt.image_base64)
    pil_img = Image.open(BytesIO(data)).convert("RGB")
    np_img = np.array(pil_img)
    set_image_if_needed(np_img)
    left = prompt.bbox_left
    top = prompt.bbox_top
    w_box = prompt.bbox_width
    h_box = prompt.bbox_height
    right = left + w_box
    bottom = top + h_box
    cx = (left+right)/2
    cy = (top+bottom)/2
    new_w = w_box*2
    new_h = h_box*2
    exp_left = max(0, cx - new_w/2)
    exp_top = max(0, cy - new_h/2)
    exp_right = min(pil_img.width, exp_left+new_w)
    exp_bottom = min(pil_img.height, exp_top+new_h)
    if exp_right<=exp_left or exp_bottom<=exp_top:
        return YoloBboxOutput(class_id="0", bbox=[0,0,0,0], uuid=prompt.uuid)
    sli = int(exp_left)
    sri = int(exp_right)
    sti = int(exp_top)
    sbi = int(exp_bottom)
    subarr = np_img[sti:sbi, sli:sri, :]
    sub_left = left - exp_left
    sub_top = top - exp_top
    sub_box = np.array([sub_left, sub_top, sub_left+w_box, sub_top+h_box])
    masks, scores, logits = predictor.predict(box=sub_box, multimask_output=False)
    mask = masks[0]
    x_sub, y_sub, xx_sub, yy_sub = mask_to_bounding_box(mask)
    gx_min = exp_left + x_sub
    gy_min = exp_top + y_sub
    gx_max = exp_left + xx_sub
    gy_max = exp_top + yy_sub
    yolo_box = to_yolo(pil_img.width, pil_img.height, gx_min, gy_min, gx_max, gy_max)
    return YoloBboxOutput(class_id="0", bbox=yolo_box, uuid=prompt.uuid)



@app.post("/sam2_bbox_auto", response_model=YoloBboxClassOutput)
def sam2_bbox_auto(prompt: BboxPrompt):
    data = base64.b64decode(prompt.image_base64)
    pil_img = Image.open(BytesIO(data)).convert("RGB")
    np_img = np.array(pil_img)
    set_image_if_needed(np_img)

    # ----- BBOX LOGIC (unchanged) -----
    full_h, full_w = pil_img.height, pil_img.width
    left = max(0, prompt.bbox_left)
    top = max(0, prompt.bbox_top)
    right = min(full_w, left + prompt.bbox_width)
    bottom = min(full_h, top + prompt.bbox_height)

    if right <= left or bottom <= top:
        return YoloBboxClassOutput(
            class_id=class_map.get("unknown", 0),
            bbox=[0, 0, 0, 0],
            uuid=prompt.uuid
        )

    sub_box = np.array([left, top, right, bottom], dtype=np.float32)
    masks, _, _ = predictor.predict(
        box=sub_box,
        multimask_output=False
    )

    # Convert mask to a bounding box, then to YOLO format
    mask = masks[0]
    x_min, y_min, x_max, y_max = mask_to_bounding_box(mask)
    yolo_box = to_yolo(full_w, full_h, x_min, y_min, x_max, y_max)

    gx_min_i = max(0, int(x_min))
    gy_min_i = max(0, int(y_min))
    gx_max_i = min(full_w, int(x_max))
    gy_max_i = min(full_h, int(y_max))

    if gx_max_i <= gx_min_i or gy_max_i <= gy_min_i:
        return YoloBboxClassOutput(
            class_id=class_map.get("unknown", 0),
            bbox=yolo_box,
            uuid=prompt.uuid
        )
    # -----------------------------------

    # ----- CLIP CLASSIFICATION -----
    subarr = np_img[gy_min_i:gy_max_i, gx_min_i:gx_max_i, :]
    final_pil = Image.fromarray(subarr)
    inp = clip_preprocess(final_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = clip_model.encode_image(inp)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    feats_np = feats.squeeze(0).cpu().numpy().reshape(1, -1)
    
    # predict() returns a string like "Building"
    pred_label = clf.predict(feats_np)[0]

    # Convert string label to integer ID
    class_id = class_map.get(pred_label, 0)

    # Return in the YoloBboxClassOutput format
    return YoloBboxClassOutput(
        class_id=class_id,
        bbox=yolo_box,
        uuid=prompt.uuid
    )

@app.post("/crop_zip_init")
def crop_zip_init():
    jobId = str(uuid.uuid4())
    job_store[jobId] = []
    return {"jobId": jobId}

@app.post("/crop_zip_chunk")
def crop_zip_chunk(request: CropZipRequest, jobId: str = Query(...)):
    if jobId not in job_store:
        raise HTTPException(status_code=400, detail="Invalid jobId")
    job_store[jobId].extend(request.images)
    return {"status": "ok", "count": len(request.images)}

@app.get("/crop_zip_finalize")
def crop_zip_finalize(jobId: str):
    if jobId not in job_store:
        raise HTTPException(status_code=400, detail="Invalid jobId")
    all_images = job_store[jobId]
    if len(all_images)==0:
        empty_buffer = io.BytesIO()
        with zipfile.ZipFile(empty_buffer, mode="w") as zf:
            pass
        empty_buffer.seek(0)
        del job_store[jobId]
        return StreamingResponse(
            empty_buffer,
            media_type="application/x-zip-compressed",
            headers={"Content-Disposition": "attachment; filename=crops.zip"}
        )
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, cropImage in enumerate(all_images):
            img_data = base64.b64decode(cropImage.image_base64)
            pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
            for bindex, bbox in enumerate(cropImage.bboxes):
                left = bbox.x
                top = bbox.y
                right = left + bbox.width
                bottom = top + bbox.height
                left = max(0, min(left, pil_img.width))
                right = max(0, min(right, pil_img.width))
                top = max(0, min(top, pil_img.height))
                bottom = max(0, min(bottom, pil_img.height))
                if right<=left or bottom<=top:
                    continue
                sub_img = pil_img.crop((left, top, right, bottom))
                stem = cropImage.originalName.rsplit(".",1)[0]
                out_name = f"{stem}-{bbox.className}-{bindex}.jpg"
                crop_buffer = io.BytesIO()
                sub_img.save(crop_buffer, format="JPEG")
                crop_buffer.seek(0)
                zf.writestr(out_name, crop_buffer.read())
    zip_buffer.seek(0)
    del job_store[jobId]
    return StreamingResponse(
        zip_buffer,
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": "attachment; filename=crops.zip"}
    )