"""
Utilities to overlay a Grad-CAM heatmap on an image (colormap + blend).
"""

import cv2
import numpy as np


def overlay_heatmap(heatmap, image, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """
    Overlay a heatmap on an image: colormap the heatmap and blend with the image.

    Red (hot) = model focused here; blue (cold) = less important.

    Args:
        heatmap: (H, W) float array, values in [0, 1].
        image: (H, W, 3) uint8 BGR image (e.g. from cv2.imread or frame).
        colormap: OpenCV colormap (default COLORMAP_JET).
        alpha: blend factor; overlay = alpha * colormap(heatmap) + (1 - alpha) * image.

    Returns:
        overlay: (H, W, 3) uint8 BGR image.
    """
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    heatmap_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

    overlay = cv2.addWeighted(heatmap_colored, alpha, image, 1 - alpha, 0)
    return overlay


def heatmap_to_colormap(heatmap, colormap=cv2.COLORMAP_JET):
    """
    Convert a [0, 1] float heatmap to a BGR colormap image.

    Args:
        heatmap: (H, W) float array in [0, 1].
        colormap: OpenCV colormap.

    Returns:
        (H, W, 3) uint8 BGR image.
    """
    heatmap_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(heatmap_uint8, colormap)
