"""FGSM and PGD attacks (torchattacks) for the six project configurations.

Attacks run in [0, 1] pixel space so epsilon means what the project plan says;
the classifier is wrapped so normalization happens inside the model and
gradients flow through it. Running this file evaluates all six
configurations on the 9,000-image evaluation split (the 1,000 calibration
images are excluded) and writes results/tables/attack_results.csv.

Usage:
    python -m src.attacks --device mps
"""

import argparse
import json
import time

import pandas as pd
import torch
import torchattacks

from src.calibration import load_test_set_01
from src.model import NormalizedModel, load_trained_model
from src.train import CHECKPOINT_DIR, CIFAR10_MEAN, CIFAR10_STD, ROOT

TABLES_DIR = ROOT / "results" / "tables"
SPLIT_PATH = ROOT / "results" / "calibration" / "split_indices.json"

EPSILONS = (0.01, 0.03, 0.05)
ATTACK_NAMES = ("FGSM", "PGD")
PGD_STEPS = 10
PGD_ALPHA_FACTOR = 2.5  # step size = 2.5 * eps / steps, the standard Madry et al. choice
ATTACK_SEED = 0
BATCH_SIZE = 250


def attack_configs():
    return [(name, eps) for name in ATTACK_NAMES for eps in EPSILONS]


def build_attack(name, model, eps):
    if name == "FGSM":
        attack = torchattacks.FGSM(model, eps=eps)
    elif name == "PGD":
        attack = torchattacks.PGD(
            model, eps=eps, alpha=PGD_ALPHA_FACTOR * eps / PGD_STEPS, steps=PGD_STEPS, random_start=True
        )
    else:
        raise ValueError(f"Unknown attack: {name}")
    return attack


def predict_batches(model, images_01, device):
    """Predicted classes for a set of [0, 1] images (model is a NormalizedModel)."""
    predictions = []
    with torch.no_grad():
        for start in range(0, len(images_01), BATCH_SIZE):
            predictions.append(model(images_01[start : start + BATCH_SIZE].to(device)).argmax(dim=1).cpu())
    return torch.cat(predictions)


def generate_adversarial(attack, images_01, labels, device, eps):
    """Attack every image. Returns (adversarial images on CPU, adversarial predictions, largest per-pixel change)."""
    adv_images, adv_predictions = [], []
    max_linf = 0.0
    for start in range(0, len(images_01), BATCH_SIZE):
        batch = images_01[start : start + BATCH_SIZE].to(device)
        batch_labels = labels[start : start + BATCH_SIZE].to(device)
        adv = attack(batch, batch_labels)
        max_linf = max(max_linf, float((adv - batch).abs().max()))
        if adv.min() < 0 or adv.max() > 1:
            raise RuntimeError("Adversarial images left the valid [0, 1] pixel range")
        with torch.no_grad():
            adv_predictions.append(attack.model(adv).argmax(dim=1).cpu())
        adv_images.append(adv.cpu())
    if max_linf > eps + 1e-5:
        raise RuntimeError(f"Attack exceeded its budget: max |change| = {max_linf:.5f} > eps = {eps}")
    return torch.cat(adv_images), torch.cat(adv_predictions), max_linf


def run_attack(attack, images_01, labels, device, eps):
    """Adversarial predictions and largest per-pixel change (images are discarded)."""
    _, adv_predictions, max_linf = generate_adversarial(attack, images_01, labels, device, eps)
    return adv_predictions, max_linf


def evaluate_config(name, eps, model, images_01, labels, clean_pred, device):
    torch.manual_seed(ATTACK_SEED)
    attack = build_attack(name, model, eps)
    start_time = time.time()
    adv_pred, max_linf = run_attack(attack, images_01, labels, device, eps)
    elapsed = time.time() - start_time

    initially_correct = clean_pred == labels
    still_correct = adv_pred == labels
    num_flipped = int((initially_correct & ~still_correct).sum())
    num_initially_correct = int(initially_correct.sum())
    return {
        "attack": name,
        "epsilon": eps,
        "num_images": len(labels),
        "clean_accuracy": float(initially_correct.float().mean()),
        "num_initially_correct": num_initially_correct,
        "num_successfully_attacked": num_flipped,
        "attack_success_rate": num_flipped / num_initially_correct,
        "adversarial_accuracy": float(still_correct.float().mean()),
        "max_linf_observed": max_linf,
        "seconds": round(elapsed, 1),
    }


def load_evaluation_split(limit=None):
    test_set = load_test_set_01()
    evaluation_idx = json.loads(SPLIT_PATH.read_text())["evaluation_indices"]
    if limit is not None:
        evaluation_idx = evaluation_idx[:limit]
    images = torch.stack([test_set[i][0] for i in evaluation_idx])
    labels = torch.tensor([test_set[i][1] for i in evaluation_idx])
    return images, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N evaluation images (smoke test)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    model, checkpoint = load_trained_model(CHECKPOINT_DIR / "resnet18_cifar10.pt", device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, test_acc={checkpoint['test_acc']:.4f}")
    model = NormalizedModel(model, CIFAR10_MEAN, CIFAR10_STD).to(device).eval()

    images, labels = load_evaluation_split(args.limit)
    print(f"Evaluating on {len(labels)} images")
    clean_pred = predict_batches(model, images, device)

    rows = []
    for name, eps in attack_configs():
        row = evaluate_config(name, eps, model, images, labels, clean_pred, device)
        rows.append(row)
        print(
            f"{name:5s} eps={eps:.2f} | success_rate={row['attack_success_rate']:.4f} "
            f"adv_acc={row['adversarial_accuracy']:.4f} attacked={row['num_successfully_attacked']} "
            f"max_linf={row['max_linf_observed']:.4f} | {row['seconds']}s"
        )

    if args.limit is None:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame(rows)
        table.to_csv(TABLES_DIR / "attack_results.csv", index=False)
        print(f"Saved attack results to {TABLES_DIR / 'attack_results.csv'}")
    else:
        print("Smoke test only: results table not written.")


if __name__ == "__main__":
    main()
