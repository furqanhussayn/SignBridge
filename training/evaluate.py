"""Evaluate trained ASL CNN on the held-out test set; save metrics and plots."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.classifier import ASLCNN, CLASS_INDEX_TO_LETTER, LABEL_TO_CLASS_INDEX  # noqa: E402
from training.train import SignMNISTDataset  # noqa: E402


@torch.no_grad()
def collect_predictions(model, loader, device):
    y_true, y_pred = [], []
    model.eval()
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1).cpu().numpy()
        y_pred.extend(pred.tolist())
        y_true.extend(yb.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def main() -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {config.MODEL_PATH}")

    test_ds = SignMNISTDataset(config.TEST_CSV)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    model = ASLCNN(num_classes=config.NUM_CLASSES).to(device)
    ckpt = torch.load(config.MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    y_true, y_pred = collect_predictions(model, test_loader, device)
    letters = [CLASS_INDEX_TO_LETTER[i] for i in range(config.NUM_CLASSES)]

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report = classification_report(
        y_true, y_pred, target_names=letters, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "n_test": int(len(test_ds)),
        "best_val_acc_from_training": ckpt.get("best_val_acc"),
    }
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (config.RESULTS_DIR / "classification_report.txt").write_text(report)
    np.save(config.RESULTS_DIR / "confusion_matrix.npy", cm)

    # Confusion matrix plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=letters,
        yticklabels=letters,
        cbar=True,
    )
    plt.title(f"ASL Alphabet Confusion Matrix (acc={accuracy:.4f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # Training history if present
    hist_path = config.RESULTS_DIR / "train_history.json"
    if hist_path.exists():
        history = json.loads(hist_path.read_text())
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(history["train_loss"], label="train")
        ax[0].plot(history["val_loss"], label="val")
        ax[0].set_title("Loss")
        ax[0].legend()
        ax[1].plot(history["val_acc"], label="val_acc", color="#3DDC97")
        ax[1].set_title("Validation Accuracy")
        ax[1].legend()
        plt.tight_layout()
        plt.savefig(config.RESULTS_DIR / "training_curves.png", dpi=150)
        plt.close()

    print("=== Test Metrics ===")
    print(json.dumps(metrics, indent=2))
    print(report)
    print(f"Saved results to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
