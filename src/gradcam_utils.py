"""Grad-CAM heatmap generation, targeting the final conv block of the CIFAR ResNet-18.

Reusable functions here are meant to be shared with the later attack/compare
pipeline (Checkpoint 4). Running this file directly generates a sanity-check
gallery of Grad-CAM overlays for one correctly classified test image per
class (Checkpoint 2's "done when" requirement).

Usage:
    python -m src.gradcam_utils --device mps
"""

import argparse

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import datasets, transforms

from src.model import build_resnet18_cifar
from src.train import CHECKPOINT_DIR, DATA_DIR, CIFAR10_MEAN, CIFAR10_STD

GALLERY_DIR_NAME = "gallery"


def get_target_layers(model):
    """The last residual block of layer4 is the standard Grad-CAM target for ResNet: the final
    convolutional feature map before global average pooling."""
    return [model.layer4[-1]]


def unnormalize_image(image_tensor, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    """Convert a normalized CHW tensor (as fed to the model) back to an HWC numpy array in [0, 1]."""
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    image = image_tensor.detach().cpu() * std_t + mean_t
    image = image.clamp(0, 1)
    return image.permute(1, 2, 0).numpy().astype(np.float32)


def compute_gradcam(cam, image_tensor, device, target_category=None):
    """Compute a Grad-CAM heatmap for a single image using an already-built GradCAM instance.

    image_tensor: normalized CHW tensor, no batch dimension.
    target_category: class index to explain; None uses the model's own top prediction.
    Returns a (H, W) grayscale heatmap in [0, 1].
    """
    targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
    input_batch = image_tensor.unsqueeze(0).to(device)
    grayscale_cam = cam(input_tensor=input_batch, targets=targets)
    return grayscale_cam[0]


def overlay_heatmap(image_tensor, heatmap, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    """Overlay a grayscale Grad-CAM heatmap on the original (unnormalized) image."""
    rgb_image = unnormalize_image(image_tensor, mean, std)
    return show_cam_on_image(rgb_image, heatmap, use_rgb=True)


def _load_model(device):
    model = build_resnet18_cifar(num_classes=10)
    checkpoint = torch.load(CHECKPOINT_DIR / "resnet18_cifar10.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _pick_one_correct_image_per_class(model, test_set, device):
    """Scan the test set and return the first correctly classified image found for each class."""
    picks = {}
    for image, label in test_set:
        if label in picks:
            continue
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            predicted = output.argmax(dim=1).item()
        if predicted == label:
            picks[label] = image
        if len(picks) == len(test_set.classes):
            break
    return picks


def main():
    import matplotlib.pyplot as plt

    from src.train import CHARTS_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    model, checkpoint = _load_model(device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, test_acc={checkpoint['test_acc']:.4f}")

    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    test_set = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)

    picks = _pick_one_correct_image_per_class(model, test_set, device)
    print(f"Found a correctly classified example for {len(picks)}/{len(test_set.classes)} classes")

    gallery_dir = CHARTS_DIR.parent / GALLERY_DIR_NAME
    gallery_dir.mkdir(parents=True, exist_ok=True)

    num_classes = len(test_set.classes)
    fig, axes = plt.subplots(num_classes, 2, figsize=(4, 2 * num_classes))

    with GradCAM(model=model, target_layers=get_target_layers(model)) as cam:
        for class_idx in range(num_classes):
            image_tensor = picks[class_idx]
            heatmap = compute_gradcam(cam, image_tensor, device)
            overlay = overlay_heatmap(image_tensor, heatmap)
            original = unnormalize_image(image_tensor)

            axes[class_idx, 0].imshow(original)
            axes[class_idx, 0].set_ylabel(test_set.classes[class_idx], fontsize=9)
            axes[class_idx, 0].set_xticks([])
            axes[class_idx, 0].set_yticks([])

            axes[class_idx, 1].imshow(overlay)
            axes[class_idx, 1].set_xticks([])
            axes[class_idx, 1].set_yticks([])

    axes[0, 0].set_title("Original", fontsize=10)
    axes[0, 1].set_title("Grad-CAM overlay", fontsize=10)
    fig.tight_layout()
    fig.savefig(gallery_dir / "gradcam_samples.png", dpi=150)
    print(f"Saved sample gallery to {gallery_dir / 'gradcam_samples.png'}")


if __name__ == "__main__":
    main()
