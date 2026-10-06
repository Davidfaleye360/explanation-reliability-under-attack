"""Heatmap similarity metrics.

SSIM is the primary metric and the only one used to decide whether an
explanation is "deceptive". IoU and correlation are added in Checkpoint 4.
"""

import numpy as np
from skimage.metrics import structural_similarity


def ssim_heatmaps(heatmap_a, heatmap_b):
    """SSIM between two (H, W) Grad-CAM heatmaps with values in [0, 1]."""
    return float(structural_similarity(heatmap_a, heatmap_b, data_range=1.0))


def ssim_batch(heatmaps_a, heatmaps_b):
    """Pairwise SSIM between two (N, H, W) arrays of heatmaps; returns an (N,) array."""
    return np.array([ssim_heatmaps(a, b) for a, b in zip(heatmaps_a, heatmaps_b)])
