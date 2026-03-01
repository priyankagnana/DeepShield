"""
Real-time inference: webcam or video file, same preprocessing as training, display Real/Fake (+ optional Grad-CAM).
Run from project root: python -m inference.realtime_inference [--video path.mp4] [--gradcam]

Optimizations:
  - MPS device support (Apple Silicon)
  - Frame skipping: inference runs every INFERENCE_EVERY frames to reduce lag
  - Grad-CAM target layer correctly set to model.backbone.features[-1]
"""

import argparse
import os
import sys

import cv2
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inference.predict import get_transform, load_model, predict, preprocess_image

# Run model inference every N frames — keeps display smooth while reducing lag
INFERENCE_EVERY = 3


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _optional_gradcam(model, frame_bgr, tensor, device):
    try:
        from explainability.gradcam import generate_gradcam
        from explainability.heatmap_utils import overlay_heatmap
        t = tensor.clone().detach().to(device).requires_grad_(True)
        heatmap = generate_gradcam(model, t, target_layer=model.backbone.features[-1])
        return overlay_heatmap(heatmap, frame_bgr.copy(), alpha=0.5)
    except Exception:
        return frame_bgr


def run_realtime(video_path=None, model_path="saved_models/best_model.pth", show_gradcam=False):
    """
    video_path: None = webcam (0), else path to video file.
    show_gradcam: if True, overlay Grad-CAM every INFERENCE_EVERY frames (slower).
    """
    device = _get_device()
    print(f"Using device: {device}")
    model, device = load_model(model_path, device)
    transform = get_transform()

    cap = cv2.VideoCapture(video_path if video_path else 0)
    if not cap.isOpened():
        print("Could not open video source (webcam or file).")
        return

    label, confidence = "...", 0.0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only run inference every INFERENCE_EVERY frames to reduce lag
        if frame_idx % INFERENCE_EVERY == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = preprocess_image(frame_rgb, transform)

            with torch.no_grad():
                label, confidence = predict(model, tensor, device)

            if show_gradcam:
                frame = _optional_gradcam(model, frame, tensor, device)

        # Overlay label on every displayed frame using the last known prediction
        text = f"{label} ({confidence:.2f})"
        color = (0, 255, 0) if label == "Real" else (0, 0, 255)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Frame {frame_idx}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("DeepShield – press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time deepfake detection (webcam or video)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file; omit for webcam")
    parser.add_argument("--model", type=str, default="saved_models/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--gradcam", action="store_true", help="Show Grad-CAM overlay on frame")
    args = parser.parse_args()
    run_realtime(video_path=args.video, model_path=args.model, show_gradcam=args.gradcam)
