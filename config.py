"""SignBridge configuration — paths, model, recognition, and UI settings."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
ASSETS_DIR = ROOT / "assets"
DICTIONARY_DIR = ASSETS_DIR / "dictionary"

# Dataset
HF_DATASET_ID = "saifmughal23/American-Sign-Language-MNIST"
TRAIN_CSV = DATA_DIR / "sign_mnist_train.csv"
TEST_CSV = DATA_DIR / "sign_mnist_test.csv"
SAMPLE_IMAGES_DIR = DATA_DIR / "samples"

# Model
MODEL_PATH = MODELS_DIR / "asl_cnn.pt"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"
IMG_SIZE = 28
NUM_CLASSES = 24
# ASL MNIST labels: 0=A ... 24=Y, skipping 9=J
LABEL_TO_LETTER = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
}
SUPPORTED_LETTERS = list(LABEL_TO_LETTER.values())

# Training
BATCH_SIZE = 128
EPOCHS = 12
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.1
SEED = 42
NUM_WORKERS = 2

# Webcam / recognition
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
HAND_PADDING = 0.35
CONFIDENCE_THRESHOLD = 0.82  # avoid silently adding weak live predictions
STABLE_FRAMES = 6
SMOOTHING_WINDOW = 12
LETTER_COOLDOWN_SEC = 0.70
CONFIRM_COOLDOWN_SEC = 1.2
MIN_HAND_MOVEMENT_RESET = 0.09  # larger motion needed to re-arm same letter
UNKNOWN_LABEL = "?"

# Hand-quality / anti-face gates
MIN_HAND_AREA_FRAC = 0.015
MAX_HAND_AREA_FRAC = 0.55
MIN_BBOX_ASPECT = 0.45
MAX_BBOX_ASPECT = 2.4
MIN_LANDMARK_SPAN = 0.04
MIN_FINGERTIP_EXTENT = 0.08
MIN_PALM_WIDTH = 0.04
MIN_TIP_SPREAD = 0.015
MIN_HANDEDNESS_SCORE = 0.70
MIN_HAND_QUALITY = 0.45
FACE_ZONE_Y_MAX = 0.42  # upper portion of frame
FACE_ZONE_MIN_AREA = 0.04
FACE_ZONE_MIN_TIP_EXTENT = 0.12
MIN_PREDICTION_MARGIN = 0.20  # top1 - top2 softmax gap
MANUAL_LETTER_CONFIRMATION = True

# Word suggestions
MAX_SUGGESTIONS = 5
MAX_EDIT_DISTANCE = 2
MIN_PREFIX_LEN = 1
DICTIONARY_PATH = DICTIONARY_DIR / "frequency_dictionary_en_82_765.txt"

# Sentence / speech
DEFAULT_VOICE_RATE = 175

# UI
APP_TITLE = "SignBridge"
APP_GEOMETRY = "1280x800"
ACCENT = "#3DDC97"
BG_DARK = "#0F1419"
PANEL_BG = "#1A2332"
TEXT_PRIMARY = "#E8EEF5"
TEXT_MUTED = "#8B9BB0"
