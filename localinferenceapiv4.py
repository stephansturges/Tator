import base64, io, zipfile, math, uuid
import numpy as np
from typing import Optional
from typing import List, Dict
import math
from fastapi.middleware.cors import CORSMiddleware

# If you installed the segment-anything library and SAM2 from the repo:
from segment_anything import sam_model_registry, SamPredictor

import torch
import clip
import joblib


from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse


from pydantic import BaseModel
import zipfile
from pathlib import Path



# In-memory store: {jobId: [list of CropImage data from all chunks]}
job_store: Dict[str, List["CropImage"]] = {}

# Paths to model files
MODEL_PATH = "./my_logreg_model.pkl"
LABELS_PATH = "./my_label_list.pkl"  # optional if you saved label references

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create the FastAPI app + enable CORS for cross-origin requests.
app = FastAPI()


# Load CLIP once, at startup
print("Loading CLIP model...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Load logistic regression from joblib
print("Loading logistic regression...")
clf = joblib.load(MODEL_PATH)
# If needed, load label list with joblib.load(LABELS_PATH)

# A pydantic model to parse JSON { "image_base64": "..." }
class Base64Payload(BaseModel):
    image_base64: str

# The response
class PredictResponse(BaseModel):
    prediction: str

# Models describing the incoming JSON

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

@app.post("/predict_base64", response_model=PredictResponse)
def predict_base64(payload: Base64Payload):
    """
    Receives JSON: { "image_base64": "..." }
    Decodes the base64 -> image -> CLIP embedding -> logistic regression => label
    Returns { "prediction": "some_class" }
    """
    # 1) Convert base64 -> bytes -> PIL image
    image_data = base64.b64decode(payload.image_base64)
    pil_img = Image.open(BytesIO(image_data)).convert("RGB")

    # 2) Preprocess + embed with CLIP
    tensor_input = clip_preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(tensor_input)
    # normalize
    features = features / features.norm(dim=-1, keepdim=True)
    features_np = features.squeeze(0).cpu().numpy().reshape(1, -1)

    # 3) Predict with logistic regression
    pred_cls = clf.predict(features_np)[0] 
    return PredictResponse(prediction=pred_cls)


@app.post("/predict_crop", response_model=PredictResponse)
def predict_crop(
    file: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    w: int = Form(...),
    h: int = Form(...)
):
    """
    Expects a multipart/form-data with:
      - file: the original image
      - x, y, w, h: bounding box int coords

    We'll crop server-side, run CLIP + logistic regression => predicted label
    """
    image_bytes = file.file.read()
    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Crop region
    left = x
    top = y
    right = x + w
    bottom = y + h
    cropped = pil_img.crop((left, top, right, bottom))

    tensor_input = clip_preprocess(cropped).unsqueeze(0).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(tensor_input)
    features = features / features.norm(dim=-1, keepdim=True)
    features_np = features.squeeze(0).cpu().numpy().reshape(1, -1)

    pred_cls = clf.predict(features_np)[0]
    return PredictResponse(prediction=pred_cls)

# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

################################################################################
# 1) SAM2 Model Initialization
################################################################################
# You need a model checkpoint from Segment Anything, e.g. "sam_vit_h_4b8939.pth"
# Then register or build the predictor. Adjust MODEL_TYPE, CHECKPOINT_PATH, etc.
MODEL_TYPE = "vit_b"  # e.g. "vit_h", "vit_l", "vit_b"
CHECKPOINT_PATH = "./sam_vit_b_01ec64.pth"

sam_model = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
predictor = SamPredictor(sam_model)

################################################################################
# 2) Pydantic input models
################################################################################
class PointPrompt(BaseModel):
    image_base64: str
    point_x: float
    point_y: float

class BboxPrompt(BaseModel):
    image_base64: str
    bbox_left: float
    bbox_top: float
    bbox_width: float
    bbox_height: float

# Output: YOLO bounding box + class label
class YoloBboxOutput(BaseModel):
    class_id: str
    bbox: List[float]  # [cx, cy, w, h] in normalized coords

################################################################################
# 3) Helper: bounding box from a mask
################################################################################
def mask_to_bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """
    Find bounding box in absolute pixel coords (left, top, right, bottom)
    from a binary mask.
    """
    # mask is HxW, bool or 0/1
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        # empty mask => return a dummy or raise error
        return (0, 0, 0, 0)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    # x_min => left, x_max => right, y_min => top, y_max => bottom
    return (int(x_min), int(y_min), int(x_max), int(y_max))

################################################################################
# 4) Converting bounding box to YOLO format
################################################################################
def to_yolo(image_width: int, image_height: int,
            left: int, top: int, right: int, bottom: int) -> list[float]:
    """
    Convert absolute bounding box coords to YOLO format:
    (center_x, center_y, width, height), normalized to [0..1].
    """
    w_abs = float(right - left)
    h_abs = float(bottom - top)
    cx_abs = left + w_abs / 2.0
    cy_abs = top + h_abs / 2.0

    cx = cx_abs / image_width
    cy = cy_abs / image_height
    w = w_abs / image_width
    h = h_abs / image_height

    return [cx, cy, w, h]

################################################################################
# 5) Single-Point Prompt (sam2_point)
################################################################################
@app.post("/sam2_point", response_model=YoloBboxOutput)
def sam2_point(prompt: PointPrompt):
    """
    Body (JSON):
      {
        "image_base64": "...",  # entire image in base64
        "point_x": <float pixel X>,
        "point_y": <float pixel Y>
      }
    Returns class_id="0" and YOLO bbox [cx, cy, w, h].
    """
    # 1) Decode base64 => PIL
    image_data = base64.b64decode(prompt.image_base64)
    pil_img = Image.open(BytesIO(image_data)).convert("RGB")

    # 2) Convert to NumPy for SAM
    img_np = np.array(pil_img)  # shape (H, W, 3) in RGB
    predictor.set_image(img_np)

    # 3) Single-point coords for SAM
    point_coords = np.array([[prompt.point_x, prompt.point_y]])  # shape (1,2)
    point_labels = np.array([1])  # 1 => foreground

    # 4) Actually run SAM to get a mask
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False
    )
    # masks shape: (1, H, W)
    mask = masks[0]

    # 5) bounding box from mask
    left, top, right, bottom = mask_to_bounding_box(mask)

    # 6) Convert to YOLO
    w_img, h_img = pil_img.width, pil_img.height
    yolo_box = to_yolo(w_img, h_img, left, top, right, bottom)

    print(yolo_box)

    return YoloBboxOutput(class_id="0", bbox=yolo_box)

################################################################################
# 6) Bounding-Box Prompt (sam2_bbox)
################################################################################
@app.post("/sam2_bbox", response_model=YoloBboxOutput)
def sam2_bbox(prompt: BboxPrompt):
    """
    Body (JSON):
      {
        "image_base64": "...",
        "bbox_left": <float pixel X>,
        "bbox_top": <float pixel Y>,
        "bbox_width": <float>,
        "bbox_height": <float>
      }

    1) Expands the bounding box to 2× around the same center.
    2) Sub-crops that region from the image (in NumPy form).
    3) Runs SAM with bounding box prompt in sub-image coords.
    4) Maps the resulting mask bounding box back to global coords, 
       then converts to YOLO format with class_id="0".
    """
    # 1) Decode base64 -> NumPy array (RGB)
    image_data = base64.b64decode(prompt.image_base64)
    np_img = np.array(Image.open(io.BytesIO(image_data)).convert("RGB"))
    full_h, full_w, _ = np_img.shape

    # 2) Original bounding box
    left = prompt.bbox_left
    top = prompt.bbox_top
    w_box = prompt.bbox_width
    h_box = prompt.bbox_height
    right = left + w_box
    bottom = top + h_box

    # 3) Expand bounding box 2× around same center
    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0
    new_w = w_box * 2.0
    new_h = h_box * 2.0

    exp_left = cx - new_w / 2
    exp_top  = cy - new_h / 2
    exp_right  = exp_left + new_w
    exp_bottom = exp_top  + new_h

    # 4) Clamp to image boundaries
    exp_left   = max(0, exp_left)
    exp_top    = max(0, exp_top)
    exp_right  = min(full_w, exp_right)
    exp_bottom = min(full_h, exp_bottom)

    # If degenerate => fallback
    if exp_right <= exp_left or exp_bottom <= exp_top:
        return YoloBboxOutput(class_id="0", bbox=[0,0,0,0])

    # 5) Crop sub-array
    sub_left_i   = int(exp_left)
    sub_top_i    = int(exp_top)
    sub_right_i  = int(exp_right)
    sub_bottom_i = int(exp_bottom)

    sub_array = np_img[sub_top_i:sub_bottom_i, sub_left_i:sub_right_i, :]

    # 6) Convert original bounding box to sub-image coords
    sub_left = left - exp_left
    sub_top  = top  - exp_top
    sub_w    = w_box
    sub_h    = h_box
    sub_box  = np.array([sub_left, sub_top, sub_left + sub_w, sub_top + sub_h])

    # 7) Run SAM in that sub-array
    predictor.set_image(sub_array)
    masks, scores, logits = predictor.predict(
        box=sub_box,
        multimask_output=False
    )
    mask = masks[0]

    # 8) Bbox from mask in sub-array coords
    x_min_sub, y_min_sub, x_max_sub, y_max_sub = mask_to_bounding_box(mask)

    # map back to global coords
    global_x_min = exp_left + x_min_sub
    global_y_min = exp_top  + y_min_sub
    global_x_max = exp_left + x_max_sub
    global_y_max = exp_top  + y_max_sub

    # 9) Convert final bounding box to YOLO
    yolo_box = to_yolo(full_w, full_h, global_x_min, global_y_min, global_x_max, global_y_max)
    
    # class_id remains "0" for now
    return YoloBboxOutput(class_id="0", bbox=yolo_box)



# Example new response: includes an int "class_id" plus the YOLO bounding box
class YoloBboxClassOutput(BaseModel):
    class_id: int
    bbox: list[float]  # [cx, cy, w, h] in normalized coords

@app.post("/sam2_bbox_auto", response_model=YoloBboxClassOutput)
def sam2_bbox_auto(prompt: BboxPrompt):
    """
    Minimally expanded version that doubles the bounding box area 
    around the input, clamps to image boundaries, 
    uses that sub-array for SAM, 
    then does a final bounding box from mask, 
    and does logistic regression on that final sub-array.
    
    Returns an int class_id and final YOLO bounding box [cx, cy, w, h].
    """

    # Decode base64 => entire NumPy array
    image_data = base64.b64decode(prompt.image_base64)
    np_img = np.array(Image.open(io.BytesIO(image_data)).convert("RGB"))  
    full_h, full_w, _ = np_img.shape  # H, W, 3

    # Original bounding box
    left = prompt.bbox_left
    top = prompt.bbox_top
    w_box = prompt.bbox_width
    h_box = prompt.bbox_height
    right = left + w_box
    bottom = top + h_box

    # Expand 2×
    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0
    new_w = w_box * 2.0
    new_h = h_box * 2.0
    exp_left = max(0, cx - new_w / 2)
    exp_top = max(0, cy - new_h / 2)
    exp_right = min(full_w, exp_left + new_w)
    exp_bottom = min(full_h, exp_top + new_h)

    # If degenerate => fallback
    if exp_right <= exp_left or exp_bottom <= exp_top:
        return YoloBboxClassOutput(class_id=0, bbox=[0,0,0,0])

    # Crop from np array for SAM
    # sub-array shape => subH, subW
    sub_left_i   = int(exp_left)
    sub_right_i  = int(exp_right)
    sub_top_i    = int(exp_top)
    sub_bottom_i = int(exp_bottom)

    sub_array = np_img[sub_top_i:sub_bottom_i, sub_left_i:sub_right_i, :]

    # We convert bounding box from global -> sub array coords for SAM
    # sub_box = [ (left-exp_left), (top-exp_top), (right-exp_left), (bottom-exp_top) ]
    box_for_sam = np.array([
        left - exp_left, 
        top  - exp_top, 
        right - exp_left, 
        bottom - exp_top
    ])

    # Run SAM on the sub_array
    predictor.set_image(sub_array)
    masks, scores, logits = predictor.predict(
        box=box_for_sam,
        multimask_output=False
    )
    mask = masks[0]

    # bounding box from mask in sub-array coords
    x_min_sub, y_min_sub, x_max_sub, y_max_sub = mask_to_bounding_box(mask)
    # map back to global coords
    global_x_min = exp_left + x_min_sub
    global_y_min = exp_top  + y_min_sub
    global_x_max = exp_left + x_max_sub
    global_y_max = exp_top  + y_max_sub

    # YOLO final bounding box
    yolo_box = to_yolo(full_w, full_h, global_x_min, global_y_min, global_x_max, global_y_max)

    # logistic regression on final bounding box area
    gx_min_i = max(0, int(global_x_min))
    gy_min_i = max(0, int(global_y_min))
    gx_max_i = min(full_w, int(global_x_max))
    gy_max_i = min(full_h, int(global_y_max))

    if gx_max_i <= gx_min_i or gy_max_i <= gy_min_i:
        # degenerate => fallback
        class_id = 0
    else:
        # 1) slice from original np_img
        final_sub = np_img[gy_min_i:gy_max_i, gx_min_i:gx_max_i, :]
        # 2) convert once to PIL to use clip_preprocess
        final_pil = Image.fromarray(final_sub)
        tensor_input = clip_preprocess(final_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = clip_model.encode_image(tensor_input)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats_np = feats.squeeze(0).cpu().numpy().reshape(1, -1)

        pred_cls = clf.predict(feats_np)[0]
        try:
            class_id = int(pred_cls)
        except:
            class_id = 0

    return YoloBboxClassOutput(class_id=class_id, bbox=yolo_box)


@app.post("/crop_zip_init")
def crop_zip_init():
    """
    Initialize a new job. Return a unique jobId for subsequent chunk requests.
    """
    jobId = str(uuid.uuid4())
    job_store[jobId] = []
    return {"jobId": jobId}

@app.post("/crop_zip_chunk")
def crop_zip_chunk(
    request: CropZipRequest,
    jobId: str = Query(...)
):
    """
    Append images from the chunk to the in-memory store for the given jobId.
    """
    if jobId not in job_store:
        raise HTTPException(status_code=400, detail="Invalid jobId")
    
    # Append these images to the job
    job_store[jobId].extend(request.images)
    return {"status": "ok", "count": len(request.images)}

@app.get("/crop_zip_finalize")
def crop_zip_finalize(
    jobId: str
):
    """
    Build a single final 'crops.zip' from all images in job_store[jobId], then remove the job from memory.
    """
    if jobId not in job_store:
        raise HTTPException(status_code=400, detail="Invalid jobId")

    all_images = job_store[jobId]
    if len(all_images) == 0:
        # empty
        empty_buffer = io.BytesIO()
        with zipfile.ZipFile(empty_buffer, mode="w") as zf:
            pass
        empty_buffer.seek(0)
        # remove job
        del job_store[jobId]
        return StreamingResponse(
            empty_buffer,
            media_type="application/x-zip-compressed",
            headers={"Content-Disposition": "attachment; filename=crops.zip"}
        )

    # Build final zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, cropImage in enumerate(all_images):
            # decode base64 -> PIL
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

                if right <= left or bottom <= top:
                    continue

                sub_img = pil_img.crop((left, top, right, bottom))
                stem = cropImage.originalName.rsplit(".", 1)[0]
                out_name = f"{stem}-{bbox.className}-{bindex}.jpg"

                crop_buffer = io.BytesIO()
                sub_img.save(crop_buffer, format="JPEG")
                crop_buffer.seek(0)
                zf.writestr(out_name, crop_buffer.read())

    zip_buffer.seek(0)

    # Remove this job from memory now that we're done
    del job_store[jobId]

    return StreamingResponse(
        zip_buffer,
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": "attachment; filename=crops.zip"}
    )