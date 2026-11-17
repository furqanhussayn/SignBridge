# SignBridge

**Predictive ASL fingerspelling → words → natural English sentence → offline speech.**

SignBridge is an assistive communication prototype. You fingerspell English letters in front of a webcam; the app recognizes letters with a trained CNN, suggests words as you spell (Gboard-style), lets you confirm with a **thumbs-up** or a click, expands content words into a simple sentence, and speaks it offline.

> This is **not** a full ASL translator. It targets static fingerspelled English (24 letters). It does not model ASL grammar, classifiers, or continuous signing.

---

## Features

- Live webcam hand detection (MediaPipe) and ASL alphabet CNN recognition
- Hand-crop preprocessing (not naïve full-frame resize)
- Rolling smoothing, confidence gates, debounce, and repeat-letter reset
- Offline predictive word suggestions (SymSpell + wordfreq)
- Thumbs-up confirmation via hand geometry (separate from the alphabet CNN)
- Clickable suggestion confirmation + Undo Letter / Undo Word
- Rule-based offline sentence expansion (e.g. `NEED WATER` → `I need water.`)
- Offline TTS (`pyttsx3`)
- Dark CustomTkinter desktop UI with model-input debug view

---

## Architecture

```
Webcam → MediaPipe Hands → hand crop/preprocess → CNN → smoother
                                                      ↓
                         WordBuilder ← suggestions (SymSpell/wordfreq)
                                                      ↓
                         thumbs-up / click confirm → confirmed words
                                                      ↓
                         SentenceEngine (templates) → editable text → TTS
```

| Module | Role |
|--------|------|
| `src/camera.py` | Threaded capture |
| `src/hand_detection.py` | MediaPipe landmarks + bbox |
| `src/preprocessing.py` | Crop → 28×28 MNIST-like input |
| `src/classifier.py` | PyTorch CNN + inference |
| `src/prediction_smoothing.py` | Stability / debounce / repeats |
| `src/confirmation_gesture.py` | Thumbs-up geometry |
| `src/word_suggestions.py` | Offline dictionary ranking |
| `src/word_builder.py` | Partial word + undo |
| `src/sentence_engine.py` | Intent templates |
| `src/speech.py` | Offline TTS |
| `src/ui.py` | CustomTkinter UI |
| `training/` | Download, train, evaluate |

---

## Dataset and license

- **Task format:** Sign Language MNIST — ~27,455 train / ~7,172 test, 28×28 grayscale, **24** static letters (J and Z excluded).
- **Referenced card:** [saifmughal23/American-Sign-Language-MNIST](https://huggingface.co/datasets/saifmughal23/American-Sign-Language-MNIST) (HF packaging often cited MIT; original content often cited CC0 — verify upstream).
- **Download script:** `python -m training.download_dataset`  
  Tries Hugging Face first; if the repo ships image folders only, it falls back to public Sign Language MNIST CSV mirrors of the same format.

---

## Model

- Lightweight PyTorch CNN (Conv→BN→ReLU→Pool ×2, Conv 128, AdaptiveAvgPool, MLP head)
- 24-way softmax, CPU-friendly
- Artifacts: `models/asl_cnn.pt`, `models/label_map.json`
- Plots/metrics: `results/`

### Actual evaluation (held-out test set)

| Metric | Value |
|--------|------:|
| Accuracy | **0.9353** |
| Precision (macro) | **0.9378** |
| Recall (macro) | **0.9358** |
| F1 (macro) | **0.9325** |
| Test samples | 7172 |
| Best validation accuracy (training) | 0.9996 |

These numbers were produced by `python -m training.evaluate` after training on this machine. **Webcam accuracy can be lower** because live frames differ from the MNIST-style crops (domain gap). Use the on-screen **Model input** preview when diagnosing live errors.

---

## Requirements

- Linux desktop with a webcam (optional for non-camera unit tests)
- Python **3.11** (recommended; project uses `uv` + `.venv`)
- Packages in `requirements.txt` (PyTorch CPU wheels recommended)

Optional for speech on Linux: `espeak-ng` system package if `pyttsx3` cannot find a driver.

---

## Installation

```bash
cd signbridge
# Python 3.11 via uv (if needed)
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate

# CPU PyTorch first, then the rest
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt

# Dictionary for suggestions
python scripts/prepare_dictionary.py
```

---

## Training

```bash
source .venv/bin/activate
python -m training.download_dataset
python -m training.train
python -m training.evaluate
```

Outputs:

- `models/asl_cnn.pt`
- `models/label_map.json`
- `results/metrics.json`
- `results/classification_report.txt`
- `results/confusion_matrix.png`
- `results/training_curves.png`

---

## Launch

```bash
source .venv/bin/activate
python app.py
```

### Controls

| Control | Action |
|---------|--------|
| Start / Stop Camera | Webcam lifecycle |
| Undo Letter / Clear Word | Edit current partial word |
| Undo Word / Clear Sentence | Edit confirmed words |
| Confirm Best | Commit top suggestion |
| Click suggestion | Commit that word |
| Thumbs-up | Confirm best suggestion (geometry detector) |
| Speak / Stop | Offline TTS |
| Diagnostics | Model metrics panel |

### Supported alphabet

`A B C D E F G H I K L M N O P Q R S T U V W X Y`  
(J and Z require motion — excluded.)

---

## Limitations

- Fingerspelled English only; not ASL grammar/vocabulary translation
- Static letters only (no J/Z, no dynamic signs)
- Rule-based sentences are templates, not an LLM and not linguistic ASL→English MT
- Live webcam performance depends on lighting, background, hand framing, and skin tone contrast
- Domain gap: strong test-set accuracy ≠ guaranteed webcam accuracy
- Thumbs-up must be visually distinct from alphabet handshapes

---

## Testing

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

Unit tests cover suggestions, smoothing/repeats, thumbs-up, sentences, model load, and preprocessing. Physical camera UX still needs a human-in-the-loop pass (this environment did open `/dev/video0` and read frames successfully during smoke testing).

---

## Troubleshooting

| Issue | Try |
|-------|-----|
| Camera won't open | Check `/dev/video*`, close other apps, try `CAMERA_INDEX` in `config.py` |
| Letters unstable | Improve lighting, plain background, fill the hand box; watch Model input |
| Always unknown | Retrain / confirm `models/asl_cnn.pt` exists; raise hand closer |
| No suggestions | Run `python scripts/prepare_dictionary.py` |
| No speech | Install `espeak-ng`; verify speakers |
| MediaPipe / JAX install heavy | Use Python 3.11 venv; pin `mediapipe>=0.10.9,<0.10.15` as in requirements |

---

## Future FYP direction

See `PROJECT_REPORT.md` for expansion ideas: dynamic ASL, gloss→English sequence models, Pakistani Sign Language, Urdu TTS, Raspberry Pi deployment, and user studies with Deaf / Hard-of-Hearing participants.

---

## Demonstration

See [`scripts/demo_script.md`](scripts/demo_script.md) for a short presentation flow.
