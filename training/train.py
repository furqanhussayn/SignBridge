"""Train a lightweight CNN on ASL MNIST and save model + label map."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.classifier import ASLCNN, CLASS_INDEX_TO_LABEL, LABEL_TO_CLASS_INDEX  # noqa: E402
from training.download_dataset import ensure_dataset  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SignMNISTDataset(Dataset):
    def __init__(self, csv_path: Path, augment: bool = False):
        df = pd.read_csv(csv_path)
        label_col = "label" if "label" in df.columns else df.columns[0]
        pixel_cols = [c for c in df.columns if c != label_col]
        labels = df[label_col].astype(int).to_numpy()
        # Keep only valid ASL MNIST labels (exclude J=9)
        mask = np.isin(labels, list(config.LABEL_TO_LETTER.keys()))
        self.labels = labels[mask]
        pixels = df.loc[mask, pixel_cols].to_numpy(dtype=np.float32)
        self.images = pixels.reshape(-1, 1, config.IMG_SIZE, config.IMG_SIZE) / 255.0
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.images[idx])
        if self.augment:
            # Live webcam crops vary in scale, angle, brightness, and contrast.
            # Small perturbations teach the CNN to generalize beyond clean MNIST.
            angle = random.uniform(-12.0, 12.0)
            theta = torch.tensor(
                [
                    [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0.0],
                    [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0.0],
                ],
                dtype=torch.float32,
            ).unsqueeze(0)
            grid = torch.nn.functional.affine_grid(theta, (1, 1, 28, 28), align_corners=False)
            x = torch.nn.functional.grid_sample(x.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False).squeeze(0)
            x = x * random.uniform(0.72, 1.28) + random.uniform(-0.08, 0.08)
            if random.random() < 0.25:
                x = x + torch.randn_like(x) * random.uniform(0.01, 0.025)
            x = torch.clamp(x, 0.0, 1.0)
        # Map original dataset label -> contiguous class index 0..23
        y = LABEL_TO_CLASS_INDEX[int(self.labels[idx])]
        return x, y


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for xb, yb in tqdm(loader, desc="train", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loss_acc(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return total_loss / total, correct / total


def main() -> None:
    set_seed(config.SEED)
    ensure_dataset()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_full = SignMNISTDataset(config.TRAIN_CSV, augment=True)
    val_full = SignMNISTDataset(config.TRAIN_CSV, augment=False)
    val_len = max(1, int(len(train_full) * config.VAL_SPLIT))
    train_len = len(train_full) - val_len
    train_ds, _ = random_split(
        train_full,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(config.SEED),
    )
    _, val_ds = random_split(
        val_full,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(config.SEED),
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = ASLCNN(num_classes=config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_loss_acc(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"Epoch {epoch:02d}/{config.EPOCHS}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training produced no weights")
    model.load_state_dict(best_state)

    torch.save(
        {
            "model_state": model.state_dict(),
            "num_classes": config.NUM_CLASSES,
            "img_size": config.IMG_SIZE,
            "class_index_to_label": CLASS_INDEX_TO_LABEL,
            "label_to_letter": {str(k): v for k, v in config.LABEL_TO_LETTER.items()},
            "best_val_acc": best_val_acc,
            "history": history,
        },
        config.MODEL_PATH,
    )

    label_map = {
        "class_index_to_label": {str(k): v for k, v in CLASS_INDEX_TO_LABEL.items()},
        "label_to_letter": {str(k): v for k, v in config.LABEL_TO_LETTER.items()},
        "supported_letters": config.SUPPORTED_LETTERS,
    }
    config.LABEL_MAP_PATH.write_text(json.dumps(label_map, indent=2))
    (config.RESULTS_DIR / "train_history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved model -> {config.MODEL_PATH}")
    print(f"Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
