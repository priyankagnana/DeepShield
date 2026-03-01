"""
Single-image inference: load model, same preprocessing as training, output Real/Fake + confidence.
Run from project root: python -m inference.predict path/to/image.jpg
"""

import os
import sys

import torch
from PIL import Image
from torchvision import transforms

# Allow running from project root or from inference/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.cnn_model import DeepfakeCNN

# Match training preprocessing (dataset.py)
IMAGE_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def load_model(model_path="saved_models/best_model.pth", device=None):
    """Load trained DeepfakeCNN from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepfakeCNN().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def preprocess_image(image_input, transform=None):
    """
    image_input: path (str), PIL Image, or numpy array (BGR/RGB).
    Returns: tensor (1, 3, 224, 224) on CPU (call .to(device) before model).
    """
    if transform is None:
        transform = get_transform()

    if isinstance(image_input, str):
        pil = Image.open(image_input).convert("RGB")
    elif hasattr(image_input, "convert"):
        pil = image_input.convert("RGB") if image_input.mode != "RGB" else image_input
    else:
        import numpy as np
        arr = image_input if isinstance(image_input, np.ndarray) else np.array(image_input)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        pil = Image.fromarray(arr)
    tensor = transform(pil).unsqueeze(0)
    return tensor


def predict(model, image_tensor, device):
    """
    Run model forward; image_tensor shape (1, 3, 224, 224).
    Returns: label ("Real" or "Fake"), confidence in [0, 1].
    """
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logit = model(image_tensor)[0, 0].item()
    prob = torch.sigmoid(torch.tensor(logit)).item()
    label = "Real" if prob >= 0.5 else "Fake"
    confidence = prob if label == "Real" else (1.0 - prob)
    return label, confidence


def predict_image(model_path, image_path, device=None):
    """
    Load model, load image, preprocess, predict.
    Returns: dict with keys: label, confidence, prob_real.
    """
    model, device = load_model(model_path, device)
    transform = get_transform()
    tensor = preprocess_image(image_path, transform)
    label, confidence = predict(model, tensor, device)
    with torch.no_grad():
        logit = model(tensor.to(device))[0, 0].item()
    prob_real = torch.sigmoid(torch.tensor(logit)).item()
    return {"label": label, "confidence": confidence, "prob_real": prob_real}


def predict_with_gradcam(model_path, image_path, device=None):
    """
    Same as predict_image but also returns Grad-CAM heatmap and overlay.
    Returns: dict with label, confidence, prob_real, heatmap (H,W), overlay (numpy BGR).
    """
    from explainability.gradcam import generate_gradcam
    from explainability.heatmap_utils import overlay_heatmap
    import numpy as np
    import cv2

    model, device = load_model(model_path, device)
    transform = get_transform()
    tensor = preprocess_image(image_path, transform).to(device)
    tensor.requires_grad_(True)

    heatmap = generate_gradcam(model, tensor, target_layer=model.backbone.features[-1])
    with torch.no_grad():
        logit = model(tensor.detach())[0, 0].item()
    prob_real = torch.sigmoid(torch.tensor(logit)).item()
    label = "Real" if prob_real >= 0.5 else "Fake"
    confidence = prob_real if label == "Real" else (1.0 - prob_real)

    if isinstance(image_path, str):
        img_bgr = cv2.imread(image_path)
    else:
        img_bgr = cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    if img_bgr is None:
        img_bgr = (tensor[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
    overlay = overlay_heatmap(heatmap, img_bgr, alpha=0.5)

    return {
        "label": label,
        "confidence": confidence,
        "prob_real": prob_real,
        "heatmap": heatmap,
        "overlay": overlay,
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path or not os.path.isfile(path):
        print("Usage: python -m inference.predict <image_path>")
        sys.exit(1)
    out = predict_image("saved_models/best_model.pth", path)
    print(f"Label: {out['label']}  Confidence: {out['confidence']:.3f}  P(Real): {out['prob_real']:.3f}")
