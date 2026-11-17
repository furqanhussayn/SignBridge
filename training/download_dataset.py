"""Download ASL Sign Language MNIST and prepare CSV + sample images.

Primary source: Hugging Face saifmughal23/American-Sign-Language-MNIST (when CSV
packaging is available). Fallback: well-known Sign Language MNIST CSV mirrors
(same 28×28 grayscale, 24-class format; ~27k train / ~7k test).
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

CSV_MIRRORS = {
    "train": [
        "https://raw.githubusercontent.com/MatthewSharpOTU/CSCI4050U_FinalProject/main/dataset/sign_mnist_train.csv",
        "https://github.com/samurainote/CNN_for_Sign_Language_Images/raw/master/sign_mnist_train.csv",
    ],
    "test": [
        "https://raw.githubusercontent.com/MatthewSharpOTU/CSCI4050U_FinalProject/main/dataset/sign_mnist_test.csv",
        "https://github.com/samurainote/CNN_for_Sign_Language_Images/raw/master/sign_mnist_test.csv",
    ],
}


def _download_url(url: str, dest: Path) -> None:
    print(f"Downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def _try_huggingface() -> bool:
    """Attempt HF download; return True if train/test CSVs were obtained."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files

        files = list_repo_files(config.HF_DATASET_ID, repo_type="dataset")
        csvs = [f for f in files if f.lower().endswith(".csv")]
        if not csvs:
            print("HF dataset has no CSV files (image folders only); using CSV mirrors.")
            return False
        for f in csvs:
            try:
                hf_hub_download(
                    repo_id=config.HF_DATASET_ID,
                    filename=f,
                    repo_type="dataset",
                    local_dir=str(config.DATA_DIR),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  HF skip {f}: {exc}")
        return any(config.DATA_DIR.rglob("*train*.csv")) and any(
            config.DATA_DIR.rglob("*test*.csv")
        )
    except Exception as exc:  # noqa: BLE001
        print(f"HF download unavailable: {exc}")
        return False


def _download_mirrors() -> None:
    if not config.TRAIN_CSV.exists():
        last_err = None
        for url in CSV_MIRRORS["train"]:
            try:
                _download_url(url, config.TRAIN_CSV)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        else:
            raise RuntimeError(f"Could not download train CSV: {last_err}")
    if not config.TEST_CSV.exists():
        last_err = None
        for url in CSV_MIRRORS["test"]:
            try:
                _download_url(url, config.TEST_CSV)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        else:
            raise RuntimeError(f"Could not download test CSV: {last_err}")


def _normalize_paths() -> tuple[Path, Path]:
    if config.TRAIN_CSV.exists() and config.TEST_CSV.exists():
        return config.TRAIN_CSV, config.TEST_CSV
    train = next(config.DATA_DIR.rglob("*train*.csv"), None)
    test = next(config.DATA_DIR.rglob("*test*.csv"), None)
    if train is None or test is None:
        raise FileNotFoundError("Train/test CSVs missing after download")
    if train.resolve() != config.TRAIN_CSV.resolve():
        pd.read_csv(train).to_csv(config.TRAIN_CSV, index=False)
    if test.resolve() != config.TEST_CSV.resolve():
        pd.read_csv(test).to_csv(config.TEST_CSV, index=False)
    return config.TRAIN_CSV, config.TEST_CSV


def save_sample_images(csv_path: Path, n: int = 24) -> None:
    config.SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    label_col = "label" if "label" in df.columns else df.columns[0]
    pixel_cols = [c for c in df.columns if c != label_col]
    seen: set[int] = set()
    for _, row in df.iterrows():
        label = int(row[label_col])
        if label in seen or label not in config.LABEL_TO_LETTER:
            continue
        seen.add(label)
        pixels = row[pixel_cols].to_numpy(dtype=np.uint8).reshape(28, 28)
        letter = config.LABEL_TO_LETTER[label]
        Image.fromarray(pixels, mode="L").save(
            config.SAMPLE_IMAGES_DIR / f"{label:02d}_{letter}.png"
        )
        if len(seen) >= n:
            break
    print(f"Saved {len(seen)} sample images to {config.SAMPLE_IMAGES_DIR}")


def ensure_dataset() -> tuple[Path, Path]:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not (config.TRAIN_CSV.exists() and config.TEST_CSV.exists()):
        print(f"Target dataset reference: {config.HF_DATASET_ID}")
        if not _try_huggingface():
            _download_mirrors()
    train, test = _normalize_paths()
    # Validate schema
    df = pd.read_csv(train, nrows=2)
    assert "label" in df.columns or df.columns[0].lower() == "label"
    n_train = sum(1 for _ in open(train)) - 1
    n_test = sum(1 for _ in open(test)) - 1
    print(f"Train CSV: {train} ({n_train} rows)")
    print(f"Test CSV:  {test} ({n_test} rows)")
    # Provenance note
    (config.DATA_DIR / "DATASET_SOURCE.txt").write_text(
        "Sign Language MNIST (24 static ASL letters, 28x28 grayscale).\n"
        f"Referenced HF card: {config.HF_DATASET_ID}\n"
        "Packaged CSVs obtained via HF when available, else public CSV mirrors "
        "of the same Kaggle Sign Language MNIST content.\n"
        "Licenses commonly cited: MIT (packaging) / CC0 (original content) — "
        "verify upstream dataset card for your distribution.\n"
    )
    save_sample_images(train)
    return train, test


def main() -> None:
    ensure_dataset()
    print("Dataset ready.")


if __name__ == "__main__":
    main()
