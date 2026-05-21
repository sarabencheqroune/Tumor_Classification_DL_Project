"""
Model evaluation script — produces accuracy, per-class metrics,
confusion matrix PNG, and a JSON summary.

Usage:
    python -m src.models.evaluate \\
        --test_dir data/raw/Testing \\
        --model    src/checkpoints/resnet18_braintumor.pt \\
        --output   model_evaluation.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend for CI/servers
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import CLASS_NAMES, IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD
from src.models.cnn_model import load_trained_model


def evaluate(
    test_dir: str,
    model_path: str,
    output_json: str = "model_evaluation.json",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_trained_model(model_path, device)

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )
    dataset = datasets.ImageFolder(test_dir, transform=transform)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy                         = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _         = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
    prec_pc, rec_pc, f1_pc, _        = precision_recall_fscore_support(all_labels, all_preds, average=None)
    cm                               = confusion_matrix(all_labels, all_preds)

    print(f"\nOverall Accuracy : {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall   : {recall:.4f}")
    print(f"Weighted F1       : {f1:.4f}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=CLASS_NAMES)}")

    # Save confusion-matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Brain Tumor Classification")
    for r in range(len(CLASS_NAMES)):
        for c in range(len(CLASS_NAMES)):
            ax.text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=12)
    plt.tight_layout()
    cm_path = Path(output_json).parent / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrix saved → {cm_path}")

    # Build metrics dict
    metrics = {
        "overall_accuracy":   float(accuracy),
        "weighted_precision": float(precision),
        "weighted_recall":    float(recall),
        "weighted_f1":        float(f1),
        "confusion_matrix":   cm.tolist(),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(prec_pc[i]),
                "recall":    float(rec_pc[i]),
                "f1":        float(f1_pc[i]),
            }
            for i in range(len(CLASS_NAMES))
        },
    }

    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Metrics saved → {out}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate brain tumor ResNet18")
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--model",    default="src/checkpoints/resnet18_braintumor.pt")
    parser.add_argument("--output",   default="model_evaluation.json")
    args = parser.parse_args()

    evaluate(args.test_dir, args.model, args.output)
