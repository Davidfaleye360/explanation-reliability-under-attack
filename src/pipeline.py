"""Attack -> explain -> compare pipeline.

For each of the six attack configurations:
  1. attack the first N evaluation images (the calibration images are never used),
  2. keep the images that were classified correctly before the attack and misclassified after it,
  3. compute a Grad-CAM heatmap for each clean image and for its attacked version, each explaining the
     model's own top-1 prediction (the clean class for the clean image, the wrong class for the attacked one),
  4. compare the two heatmaps with SSIM (primary), IoU of the top 20% pixels and Pearson correlation,
  5. call the attacked explanation "deceptive" if SSIM >= the threshold locked in results/calibration/threshold.json.

IoU and correlation never decide "deceptive"; only SSIM against the locked threshold does.

Usage:
    python -m src.pipeline --subset-size 1000 --device mps
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.attacks import ATTACK_SEED, attack_configs, build_attack, generate_adversarial, load_evaluation_split
from src.calibration import gradcam_heatmaps, load_test_set_01
from src.metrics import iou_batch, pearson_batch, ssim_batch
from src.model import NormalizedModel, load_trained_model
from src.train import CHECKPOINT_DIR, CIFAR10_MEAN, CIFAR10_STD, ROOT

TABLES_DIR = ROOT / "results" / "tables"
GALLERY_DIR = ROOT / "results" / "gallery"
THRESHOLD_PATH = ROOT / "results" / "calibration" / "threshold.json"


def load_locked_threshold():
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError("Run `python -m src.calibration` first: the threshold must be locked before any comparison.")
    return float(json.loads(THRESHOLD_PATH.read_text())["threshold"])


def predict(model, images_01, device, batch_size=250):
    predictions = []
    with torch.no_grad():
        for start in range(0, len(images_01), batch_size):
            predictions.append(model(images_01[start : start + batch_size].to(device)).argmax(dim=1).cpu())
    return torch.cat(predictions)


def compare_config(name, eps, attack_model, raw_model, images, labels, clean_pred, clean_heatmaps, threshold, device):
    torch.manual_seed(ATTACK_SEED)
    attack = build_attack(name, attack_model, eps)
    adv_images, adv_pred, _ = generate_adversarial(attack, images, labels, device, eps)

    success = ((clean_pred == labels) & (adv_pred != labels)).numpy()
    positions = np.nonzero(success)[0]
    adv_heatmaps = gradcam_heatmaps(raw_model, adv_images[positions], device)
    clean_hm = clean_heatmaps[positions]

    ssim = ssim_batch(clean_hm, adv_heatmaps)
    per_image = pd.DataFrame({
        "attack": name,
        "epsilon": eps,
        "subset_position": positions,
        "label": labels.numpy()[positions],
        "adv_pred": adv_pred.numpy()[positions],
        "ssim": ssim,
        "iou_top20": iou_batch(clean_hm, adv_heatmaps),
        "pearson": pearson_batch(clean_hm, adv_heatmaps),
        "deceptive": ssim >= threshold,
    })
    kept = {
        "positions": positions,
        "clean_images": images[positions],
        "adv_images": adv_images[positions],
        "clean_heatmaps": clean_hm,
        "adv_heatmaps": adv_heatmaps,
        "ssim": ssim,
        "label": labels.numpy()[positions],
        "adv_pred": adv_pred.numpy()[positions],
    }
    return per_image, kept


def summarize(per_image, n_subset, n_clean_correct, name, eps, threshold):
    n = len(per_image)
    return {
        "attack": name,
        "epsilon": eps,
        "subset_images": n_subset,
        "clean_correct": n_clean_correct,
        "attacked_images": n,
        "ssim_mean": per_image["ssim"].mean(),
        "ssim_median": per_image["ssim"].median(),
        "ssim_min": per_image["ssim"].min(),
        "ssim_max": per_image["ssim"].max(),
        "iou_mean": per_image["iou_top20"].mean(),
        "iou_median": per_image["iou_top20"].median(),
        "pearson_mean": per_image["pearson"].mean(),
        "pearson_median": per_image["pearson"].median(),
        "iou_undefined": int(per_image["iou_top20"].isna().sum()),
        "pearson_undefined": int(per_image["pearson"].isna().sum()),
        "threshold": threshold,
        "deceptive_count": int(per_image["deceptive"].sum()),
        "deceptive_rate": float(per_image["deceptive"].mean()),
    }


def overlay(image_chw, heatmap):
    return show_cam_on_image(image_chw.permute(1, 2, 0).numpy().astype(np.float32), heatmap, use_rgb=True)


def pick_examples(kept_by_config):
    """Rule-based picks (no hand selection): overall most and least similar, and the median example of three configs."""
    picks = []
    all_rows = [(key, i, s) for key, k in kept_by_config.items() for i, s in enumerate(k["ssim"])]
    key, i, _ = max(all_rows, key=lambda r: r[2])
    picks.append(("Highest SSIM overall", key, i))
    for key in [("FGSM", 0.01), ("PGD", 0.01), ("PGD", 0.05)]:
        ssim = kept_by_config[key]["ssim"]
        picks.append(("Median SSIM", key, int(np.argmin(np.abs(ssim - np.median(ssim))))))
    key, i, _ = min(all_rows, key=lambda r: r[2])
    picks.append(("Lowest SSIM overall", key, i))
    return picks


def save_examples(kept_by_config, class_names, threshold, path):
    picks = pick_examples(kept_by_config)
    fig, axes = plt.subplots(len(picks), 4, figsize=(8.5, 2.35 * len(picks)))
    for row, (kind, key, i) in enumerate(picks):
        k = kept_by_config[key]
        ssim = k["ssim"][i]
        verdict = "deceptive" if ssim >= threshold else "not deceptive"
        panels = [
            (k["clean_images"][i].permute(1, 2, 0).numpy(), f"clean: {class_names[k['label'][i]]}"),
            (overlay(k["clean_images"][i], k["clean_heatmaps"][i]), "clean Grad-CAM"),
            (k["adv_images"][i].permute(1, 2, 0).numpy(), f"attacked: {class_names[k['adv_pred'][i]]}"),
            (overlay(k["adv_images"][i], k["adv_heatmaps"][i]), f"{key[0]} eps={key[1]}  SSIM={ssim:.3f}\n{verdict}"),
        ]
        for col, (picture, title) in enumerate(panels):
            axes[row, col].imshow(picture)
            axes[row, col].set_title(title, fontsize=8)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(kind, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-size", type=int, default=1000, help="number of evaluation images to attack")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    threshold = load_locked_threshold()
    print(f"Locked deceptive-explanation threshold: SSIM >= {threshold:.4f}")

    raw_model, checkpoint = load_trained_model(CHECKPOINT_DIR / "resnet18_cifar10.pt", device)
    attack_model = NormalizedModel(raw_model, CIFAR10_MEAN, CIFAR10_STD).to(device).eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, test_acc={checkpoint['test_acc']:.4f}")

    images, labels = load_evaluation_split(args.subset_size)
    class_names = load_test_set_01().classes
    print(f"Subset: first {len(labels)} evaluation images")

    clean_pred = predict(attack_model, images, device)
    correct = (clean_pred == labels).numpy()
    clean_heatmaps = np.full((len(images), 32, 32), np.nan, dtype=np.float32)
    clean_heatmaps[correct] = gradcam_heatmaps(raw_model, images[correct], device)
    print(f"Clean heatmaps computed for {int(correct.sum())} correctly classified images")

    per_image_frames, summaries, kept_by_config = [], [], {}
    for name, eps in attack_configs():
        per_image, kept = compare_config(
            name, eps, attack_model, raw_model, images, labels, clean_pred, clean_heatmaps, threshold, device
        )
        per_image_frames.append(per_image)
        kept_by_config[(name, eps)] = kept
        summary = summarize(per_image, len(labels), int(correct.sum()), name, eps, threshold)
        summaries.append(summary)
        print(
            f"{name:5s} eps={eps:.2f} | attacked={summary['attacked_images']:4d} "
            f"SSIM mean={summary['ssim_mean']:.4f} median={summary['ssim_median']:.4f} | "
            f"IoU mean={summary['iou_mean']:.3f} | Pearson mean={summary['pearson_mean']:.3f} | "
            f"deceptive={summary['deceptive_count']} ({summary['deceptive_rate']:.1%})"
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"n{len(labels)}"
    pd.concat(per_image_frames, ignore_index=True).to_csv(TABLES_DIR / f"pipeline_{tag}_per_image.csv", index=False)
    pd.DataFrame(summaries).to_csv(TABLES_DIR / f"pipeline_{tag}_summary.csv", index=False)
    save_examples(kept_by_config, class_names, threshold, GALLERY_DIR / f"attack_examples_{tag}.png")
    print(f"Saved tables to {TABLES_DIR} and examples to {GALLERY_DIR / f'attack_examples_{tag}.png'}")


if __name__ == "__main__":
    main()
