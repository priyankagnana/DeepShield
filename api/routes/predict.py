"""
REST endpoints for image and video deepfake prediction.

POST /predict/image  — analyze a single uploaded image
POST /predict/video  — analyze an uploaded video (samples N frames)
"""

import os
import sys
import tempfile

from fastapi import APIRouter, File, UploadFile, HTTPException, Query

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api.schemas import PredictionResponse, VideoPredictionResponse

router = APIRouter()

ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO = {".mp4", ".avi", ".mov", ".mkv"}


def _check_ext(filename: str, allowed: set, kind: str):
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported {kind} format '{ext}'. Allowed: {sorted(allowed)}",
        )
    return ext


@router.post("/predict/image", response_model=PredictionResponse)
async def predict_image_endpoint(
    file: UploadFile = File(...),
    gradcam: bool = Query(False, description="Also compute Grad-CAM (slower)"),
):
    """Upload an image and get Real/Fake label with confidence score."""
    from api.main import get_model
    model, device = get_model()

    ext = _check_ext(file.filename, ALLOWED_IMAGE, "image")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        path = tmp.name

    try:
        from inference.predict import get_transform, preprocess_image, predict
        import torch

        transform = get_transform()
        tensor = preprocess_image(path, transform)
        with torch.no_grad():
            label, confidence = predict(model, tensor, device)
        import torch as _torch
        logit = model(tensor.to(device))[0, 0].item()
        prob_real = _torch.sigmoid(_torch.tensor(logit)).item()

        return PredictionResponse(
            label=label,
            confidence=round(confidence, 4),
            prob_real=round(prob_real, 4),
        )
    finally:
        os.unlink(path)


@router.post("/predict/video", response_model=VideoPredictionResponse)
async def predict_video_endpoint(
    file: UploadFile = File(...),
    num_frames: int = Query(16, ge=1, le=64,
                            description="Number of frames to sample from the video"),
):
    """Upload a video and get an aggregated Real/Fake verdict across sampled frames."""
    ext = _check_ext(file.filename, ALLOWED_VIDEO, "video")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        path = tmp.name

    try:
        from inference.predict import predict_video
        from api.main import MODEL_PATH

        result = predict_video(MODEL_PATH, path, num_frames=num_frames)
        return VideoPredictionResponse(**result)
    finally:
        os.unlink(path)
