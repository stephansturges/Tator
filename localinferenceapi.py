import os
import torch
import clip
import joblib
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Paths to model files
MODEL_PATH = "./my_logreg_model.pkl"
LABELS_PATH = "./my_label_list.pkl"  # optional if you saved label references

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create the FastAPI app + enable CORS for cross-origin requests.
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # or specify e.g. ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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