"""ASL alphabet CNN and inference wrapper."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

# Contiguous class indices 0..23 mapped to original dataset labels
CLASS_INDEX_TO_LABEL = {
    i: label for i, label in enumerate(sorted(config.LABEL_TO_LETTER.keys()))
}
LABEL_TO_CLASS_INDEX = {label: i for i, label in CLASS_INDEX_TO_LABEL.items()}
CLASS_INDEX_TO_LETTER = {
    i: config.LABEL_TO_LETTER[label] for i, label in CLASS_INDEX_TO_LABEL.items()
}


class ASLCNN(nn.Module):
    """Compact CNN for 28x28 grayscale ASL letters — CPU-friendly."""

    def __init__(self, num_classes: int = 24):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(128, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class LetterClassifier:
    """Loads trained weights and predicts letter + confidence from a 28x28 array."""

    def __init__(self, model_path: Optional[Path] = None, device: Optional[str] = None):
        self.model_path = Path(model_path or config.MODEL_PATH)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = ASLCNN(num_classes=config.NUM_CLASSES).to(self.device)
        self._load()
        self.model.eval()

    def _load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Run: python -m training.train"
            )
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        label_map_path = config.LABEL_MAP_PATH
        if label_map_path.exists():
            data = json.loads(label_map_path.read_text())
            self.class_to_letter = {
                int(k): config.LABEL_TO_LETTER[int(v)]
                if str(v).isdigit()
                else str(v)
                for k, v in data["class_index_to_label"].items()
            }
            # class_index_to_label stores original numeric labels
            fixed = {}
            for k, v in data["class_index_to_label"].items():
                label_id = int(v)
                fixed[int(k)] = config.LABEL_TO_LETTER[label_id]
            self.class_to_letter = fixed
        else:
            self.class_to_letter = CLASS_INDEX_TO_LETTER

    @torch.no_grad()
    def predict(self, image_28: np.ndarray) -> tuple[str, float, np.ndarray]:
        """
        Predict letter from float32 image in [0, 1], shape (28, 28) or (1, 28, 28).

        Returns (letter, confidence, class_probabilities).
        """
        if image_28.ndim == 2:
            arr = image_28[None, None, ...]
        elif image_28.ndim == 3:
            arr = image_28[None, ...]
        else:
            arr = image_28
        tensor = torch.from_numpy(arr.astype(np.float32)).to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        conf = float(probs[idx])
        letter = self.class_to_letter.get(idx, config.UNKNOWN_LABEL)
        return letter, conf, probs
