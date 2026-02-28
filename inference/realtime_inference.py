"""
Real-time inference: webcam or video file, same preprocessing as training, display Real/Fake (+ optional Grad-CAM).
Run from project root: python -m inference.realtime_inference [--video path.mp4] [--gradcam]
"""

import argparse
import os
import sys

import cv2
import torch
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.cnn_model import DeepfakeCNN
from inference.predict import get_transform, load_model, predict, preprocess_image

# Optional Grad-CAM (can be disabled if not needed for speed)
def _optional_gradcam(model, frame_bgr, tensor, device, show_overlay):
    if not show_overlay:
        return frame_bgr
    try:
        from explainability.gradcam import generate_gradcam
        from explainability.heatmap_utils import overlay_heatmap
        t = tensor.clone().detach().to(device).requires_grad_(True)
        heatmap = generate_gradcam(model, t, target_layer=model.conv2)
        return overlay_heatmap(heatmap, frame_bgr.copy(), alpha=0.5)
    except Exception:
        return frame_bgr


def run_realtime(video_path=None, model_path="saved_models/best_model.pth", show_gradcam=False):
    """
    video_path: None = webcam (0), else path to video file.
    show_gradcam: if True, overlay Grad-CAM on frame (slower).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, device = load_model(model_path, device)
    transform = get_transform()

    cap = cv2.VideoCapture(video_path if video_path else 0)
    if not cap.isOpened():
        print("Could not open video source (webcam or file).")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess: BGR -> RGB, then same as training
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = preprocess_image(frame_rgb, transform)

        with torch.no_grad():
            label, confidence = predict(model, tensor, device)

        # Optional Grad-CAM overlay
        if show_gradcam:
            frame = _optional_gradcam(model, frame, tensor, device, True)

        # Draw label and confidence on frame
        text = f"{label} ({confidence:.2f})"
        color = (0, 255, 0) if label == "Real" else (0, 0, 255)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("DeepShield", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time deepfake detection (webcam or video)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file; omit for webcam")
    parser.add_argument("--model", type=str, default="saved_models/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--gradcam", action="store_true", help="Show Grad-CAM overlay on frame")
    args = parser.parse_args()
    run_realtime(video_path=args.video, model_path=args.model, show_gradcam=args.gradcam)
