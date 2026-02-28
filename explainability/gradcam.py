"""
Grad-CAM: gradient-weighted class activation mapping using the last conv layer.
Uses the spatial branch's last conv (e.g. conv2) to build a heatmap of where the model looks.
"""

import torch
import torch.nn.functional as F
import numpy as np


def generate_gradcam(model, image_tensor, target_layer=None):
    """
    Compute Grad-CAM heatmap for a single image (batch size 1).

    Args:
        model: nn.Module (e.g. DeepfakeCNN), in eval mode.
        image_tensor: (1, C, H, W) tensor, requires_grad will be set.
        target_layer: Conv2d layer to use (e.g. model.conv2). If None, use model.conv2.

    Returns:
        heatmap: (H, W) numpy array, values in [0, 1].
    """
    if target_layer is None:
        target_layer = getattr(model, "conv2", None)
    if target_layer is None:
        raise ValueError("target_layer must be provided or model must have 'conv2'")

    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    activations = []

    def save_activation(module, input, output):
        activations.append(output)

    handle = target_layer.register_forward_hook(save_activation)

    model.zero_grad()
    output = model(image_tensor)

    # Backward from the single logit (batch index 0)
    score = output[0, 0]
    score.backward()

    handle.remove()

    if not activations:
        raise RuntimeError("No activations captured; check target_layer.")

    act = activations[0]
    grad = act.grad
    if grad is None:
        raise RuntimeError("Gradients not available; ensure backward() was called.")

    # Grad-CAM: channel weights = global average of gradients
    weights = grad.mean(dim=(2, 3))
    cam = (weights.unsqueeze(-1).unsqueeze(-1) * act).sum(dim=1, keepdim=True)
    cam = F.relu(cam).squeeze(0).squeeze(0).cpu().numpy()

    # Resize to input spatial size
    h, w = image_tensor.shape[2], image_tensor.shape[3]
    cam = _resize_heatmap(cam, (h, w))

    # Normalize to [0, 1]
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam.astype(np.float32)


def _resize_heatmap(cam, size):
    """Resize 2D heatmap to (height, width) using bilinear-style interpolation."""
    cam_t = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0)
    cam_t = F.interpolate(cam_t, size=size, mode="bilinear", align_corners=False)
    return cam_t.squeeze().numpy()


# Backward compatibility: keep old name pointing to new implementation
def generate_heatmap(model, image_tensor, target_layer=None):
    """Legacy name for generate_gradcam."""
    return generate_gradcam(model, image_tensor, target_layer)
