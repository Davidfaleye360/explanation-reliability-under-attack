"""Heatmap similarity metrics.

SSIM is the primary metric and the only one used to decide whether an
explanation is "deceptive". IoU of the top salient regions and Pearson
correlation are supporting metrics that corroborate it; they never decide
"deceptive" status on their own.

All functions take Grad-CAM heatmaps as (H, W) arrays with values in [0, 1].
Degenerate cases return NaN instead of a made-up number, and callers count them.
"""

import numpy as np
from skimage.metrics import structural_similarity

TOP_FRACTION = 0.2  # fixed for every heatmap; never tuned per image


def ssim_heatmaps(heatmap_a, heatmap_b):
    """SSIM between two (H, W) heatmaps (primary metric)."""
    return float(structural_similarity(heatmap_a, heatmap_b, data_range=1.0))


def salient_mask(heatmap, top_fraction=TOP_FRACTION):
    """Boolean mask of the heatmap's top `top_fraction` pixels by intensity.

    A pixel is kept if it is at or above the (1 - top_fraction) percentile of that heatmap's own values.
    If that percentile is 0 (more than 80% of pixels are exactly zero), only pixels with positive
    intensity are kept, so zero-importance pixels are never counted as salient.
    """
    threshold = np.percentile(heatmap, 100 * (1 - top_fraction))
    return heatmap >= threshold if threshold > 0 else heatmap > 0


def iou_top_regions(heatmap_a, heatmap_b, top_fraction=TOP_FRACTION):
    """Intersection over union of the two heatmaps' top-fraction salient regions. NaN if both are empty."""
    mask_a = salient_mask(heatmap_a, top_fraction)
    mask_b = salient_mask(heatmap_b, top_fraction)
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(mask_a, mask_b).sum() / union)


def pearson_heatmaps(heatmap_a, heatmap_b):
    """Pearson correlation between the full flattened heatmaps. NaN if either heatmap is constant."""
    a, b = heatmap_a.ravel().astype(np.float64), heatmap_b.ravel().astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def ssim_batch(heatmaps_a, heatmaps_b):
    """Pairwise SSIM between two (N, H, W) arrays; returns an (N,) array."""
    return np.array([ssim_heatmaps(a, b) for a, b in zip(heatmaps_a, heatmaps_b)])


def iou_batch(heatmaps_a, heatmaps_b, top_fraction=TOP_FRACTION):
    return np.array([iou_top_regions(a, b, top_fraction) for a, b in zip(heatmaps_a, heatmaps_b)])


def pearson_batch(heatmaps_a, heatmaps_b):
    return np.array([pearson_heatmaps(a, b) for a, b in zip(heatmaps_a, heatmaps_b)])
