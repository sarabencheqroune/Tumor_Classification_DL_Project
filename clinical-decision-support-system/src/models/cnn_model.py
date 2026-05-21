"""
ResNet18-based brain tumor classifier.

The final fully-connected layer is replaced with a 4-class head:
  glioma | meningioma | pituitary | no_tumor
"""
import torch
import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES


def build_resnet18(pretrained: bool = False, num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Return a ResNet18 with a custom classification head.

    Args:
        pretrained: load ImageNet weights for the backbone (only used during training)
        num_classes: number of output classes (default 4)
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features          # 512 for ResNet18
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_trained_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Load a fine-tuned ResNet18 from a .pt checkpoint.

    Raises FileNotFoundError if the checkpoint doesn't exist.
    """
    import os
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at '{model_path}'. "
            "Run the training notebook first."
        )

    model = build_resnet18(pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
