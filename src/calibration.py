"""Noise-floor calibration for the deceptive-explanation threshold.

Builds a held-out set of clean, correctly classified test images, pairs each
with a copy perturbed by small random (non-adversarial) noise at the smallest
attack magnitude (eps = 0.01), and measures the SSIM between the two Grad-CAM
heatmaps. The deceptive-explanation threshold is fixed at the 25th percentile
of that noise-floor distribution.

This must run BEFORE any adversarial explanation is evaluated. The threshold
is written once to results/calibration/threshold.json; the script refuses to
overwrite it, so the value cannot drift after attack results exist.

Usage:
    python -m src.calibration --device mps
"""

import argparse
import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam import GradCAM
from torchvision import datasets, transforms

from src.gradcam_utils import compute_gradcam_batch, get_target_layers, normalize_images
from src.metrics import ssim_batch
from src.model import load_trained_model
from src.train import CHECKPOINT_DIR, DATA_DIR, ROOT

CALIBRATION_DIR = ROOT / "results" / "calibration"
THRESHOLD_PATH = CALIBRATION_DIR / "threshold.json"
SPLIT_PATH = CALIBRATION_DIR / "split_indices.json"

SPLIT_SEED = 0
NUM_CALIBRATION_IMAGES = 1000
NOISE_EPS = 0.01
NOISE_SEED = 0
THRESHOLD_PERCENTILE = 25
BATCH_SIZE = 100


def load_test_set_01():
    """CIFAR-10 test set with pixels in [0, 1] (no normalization), as attacks and noise operate in pixel space."""
    return datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transforms.ToTensor())


def split_test_indices(num_test_images):
    """Deterministically split test-set indices into (calibration, evaluation) so the two never overlap."""
    permutation = np.random.default_rng(SPLIT_SEED).permutation(num_test_images)
    return np.sort(permutation[:NUM_CALIBRATION_IMAGES]), np.sort(permutation[NUM_CALIBRATION_IMAGES:])


def add_random_sign_noise(images_01, eps, generator):
    """Add +eps or -eps to every pixel channel at random, then clip to [0, 1].

    Per-pixel magnitude matches an FGSM step at the same eps, but the direction is random rather than adversarial.
    """
    signs = torch.randint(0, 2, images_01.shape, generator=generator).float() * 2 - 1
    return (images_01 + eps * signs).clamp(0, 1)


def predict(model, images_01, device):
    predictions = []
    with torch.no_grad():
        for start in range(0, len(images_01), BATCH_SIZE):
            batch = normalize_images(images_01[start : start + BATCH_SIZE].to(device))
            predictions.append(model(batch).argmax(dim=1).cpu())
    return torch.cat(predictions)


def gradcam_heatmaps(model, images_01, device):
    heatmaps = []
    with GradCAM(model=model, target_layers=get_target_layers(model)) as cam:
        for start in range(0, len(images_01), BATCH_SIZE):
            heatmaps.append(compute_gradcam_batch(cam, images_01[start : start + BATCH_SIZE], device))
    return np.concatenate(heatmaps)


def plot_distribution(ssim_values, threshold, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ssim_values, bins=40, color="#4C78A8", edgecolor="white")
    ax.set_yscale("log")
    ax.axvline(threshold, color="#E45756", linestyle="--", linewidth=2,
               label=f"threshold (25th percentile) = {threshold:.4f}")
    ax.set_xlabel("SSIM between clean and noisy Grad-CAM heatmaps")
    ax.set_ylabel("Number of image pairs (log scale)")
    ax.set_title(f"Noise-floor distribution (random noise, eps = {NOISE_EPS}, n = {len(ssim_values)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    if THRESHOLD_PATH.exists():
        print(f"{THRESHOLD_PATH} already exists. The threshold is locked and will not be recalculated.")
        print("(Delete the file by hand only if calibration itself was buggy and no adversarial results exist yet.)")
        return

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_trained_model(CHECKPOINT_DIR / "resnet18_cifar10.pt", device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, test_acc={checkpoint['test_acc']:.4f}")

    test_set = load_test_set_01()
    calibration_idx, evaluation_idx = split_test_indices(len(test_set))
    SPLIT_PATH.write_text(json.dumps({
        "split_seed": SPLIT_SEED,
        "calibration_indices": calibration_idx.tolist(),
        "evaluation_indices": evaluation_idx.tolist(),
    }))
    print(f"Split test set: {len(calibration_idx)} calibration / {len(evaluation_idx)} evaluation images")

    images = torch.stack([test_set[i][0] for i in calibration_idx])
    labels = torch.tensor([test_set[i][1] for i in calibration_idx])

    clean_pred = predict(model, images, device)
    correct = clean_pred == labels
    images, labels, clean_pred = images[correct], labels[correct], clean_pred[correct]
    kept_idx = calibration_idx[correct.numpy()]
    print(f"Calibration set: {len(images)} of {len(calibration_idx)} images are correctly classified and kept")

    generator = torch.Generator().manual_seed(NOISE_SEED)
    noisy_images = add_random_sign_noise(images, NOISE_EPS, generator)
    noisy_pred = predict(model, noisy_images, device)
    num_pred_changed = int((noisy_pred != clean_pred).sum())
    print(f"Random noise changed the predicted class for {num_pred_changed} of {len(images)} images")

    clean_heatmaps = gradcam_heatmaps(model, images, device)
    noisy_heatmaps = gradcam_heatmaps(model, noisy_images, device)
    ssim_values = ssim_batch(clean_heatmaps, noisy_heatmaps)

    threshold = float(np.percentile(ssim_values, THRESHOLD_PERCENTILE))

    pd.DataFrame({
        "test_index": kept_idx,
        "label": labels.numpy(),
        "pred_clean": clean_pred.numpy(),
        "pred_noisy": noisy_pred.numpy(),
        "ssim": ssim_values,
    }).to_csv(CALIBRATION_DIR / "noise_floor_ssim.csv", index=False)
    plot_distribution(ssim_values, threshold, CALIBRATION_DIR / "noise_floor_ssim_hist.png")

    THRESHOLD_PATH.write_text(json.dumps({
        "threshold": threshold,
        "percentile": THRESHOLD_PERCENTILE,
        "rule": "an adversarial example's explanation is 'deceptive' if SSIM(clean heatmap, adversarial heatmap) >= threshold",
        "noise_type": "random_sign",
        "noise_eps": NOISE_EPS,
        "noise_seed": NOISE_SEED,
        "split_seed": SPLIT_SEED,
        "num_calibration_pool": int(len(calibration_idx)),
        "num_pairs": int(len(ssim_values)),
        "num_pred_changed_by_noise": num_pred_changed,
        "ssim_mean": float(ssim_values.mean()),
        "ssim_median": float(np.median(ssim_values)),
        "ssim_min": float(ssim_values.min()),
        "ssim_max": float(ssim_values.max()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_test_acc": float(checkpoint["test_acc"]),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))

    print(f"SSIM over {len(ssim_values)} pairs: mean={ssim_values.mean():.4f} median={np.median(ssim_values):.4f} "
          f"min={ssim_values.min():.4f} max={ssim_values.max():.4f}")
    print(f"Threshold (25th percentile) = {threshold:.4f}  -> saved to {THRESHOLD_PATH}")


if __name__ == "__main__":
    main()
