"""ResNet-18 adapted for 32x32 CIFAR-10 input.

Standard torchvision ResNet-18 assumes 224x224 input (7x7 stride-2 stem conv
+ maxpool), which throws away too much spatial resolution for 32x32 images.
This version uses the common CIFAR adaptation: a 3x3 stride-1 stem conv and
no initial maxpool.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def load_trained_model(checkpoint_path, device):
    """Load a saved checkpoint into a fresh CIFAR ResNet-18 and put it in eval mode."""
    model = build_resnet18_cifar(num_classes=10)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
