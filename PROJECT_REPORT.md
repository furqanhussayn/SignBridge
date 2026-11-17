# SignBridge — Project Report

## Abstract

SignBridge is a desktop assistive prototype that converts ASL fingerspelling into spoken English without cloud services. A webcam captures the user’s hand; MediaPipe detects landmarks and a cropped hand region is classified by a lightweight convolutional neural network trained on Sign Language MNIST (24 static letters). Rolling prediction smoothing commits letters into a partial word while an offline dictionary engine ranks completions. The user confirms a suggestion with a thumbs-up gesture (landmark geometry, separate from the CNN) or a UI click. Confirmed content words are expanded by a rule-based sentence engine and spoken with offline text-to-speech. On the held-out MNIST test set the CNN reached **93.53% accuracy** (macro F1 **0.9325**). The system is a fingerspelling assistant, not a complete ASL translator.

## Problem statement

Deaf and Hard-of-Hearing users, and hearing partners, often need a lightweight way to produce short English phrases when a full interpreter or commercial ASL system is unavailable. Spelling every letter of every word is slow. Existing demos often stop at isolated letter accuracy and ignore word prediction, confirmation ergonomics, and speech output. There is also a practical gap between clean training images and messy webcam frames.

## Objectives

1. Train a real, reproducible CNN on a public ASL alphabet dataset and report honest metrics.
2. Build a responsive webcam pipeline with hand cropping and stability controls.
3. Provide Gboard-like predictive word suggestions offline.
4. Confirm words via thumbs-up geometry and UI selection without confusing confirmation with letter classes.
5. Expand content words into simple English sentences and speak them offline.
6. Document limitations and a credible future FYP roadmap.

## Proposed methodology

1. **Data:** Sign Language MNIST-style 28×28 grayscale images, 24 classes (A–Y excluding J/Z).
2. **Model:** Compact CNN trained with Adam; best validation checkpoint retained.
3. **Live CV:** MediaPipe Hands → padded bbox → CLAHE / soft background suppression → 28×28 normalize → CNN.
4. **UX layer:** Smoothing + debounce; SymSpell/wordfreq suggestions; thumbs-up confirmer; template sentence engine; pyttsx3.
5. **Evaluation:** sklearn classification report + confusion matrix on the official test split; automated unit tests for non-camera modules; camera smoke open/read.

## Dataset description

| Split | Samples |
|-------|--------:|
| Train | 27,455 |
| Test | 7,172 |

- Resolution: 28×28 grayscale  
- Classes: 24 (labels skip J=9; Z absent)  
- Source reference: Hugging Face card `saifmughal23/American-Sign-Language-MNIST`; CSVs obtained via that ecosystem or public Sign Language MNIST mirrors when HF ships image folders only  
- Licensing commonly cited on packaging/cards: MIT (packaged) / CC0 (original content) — verify upstream for redistribution  

Sample inspection showed relatively bright hand regions (mean intensity roughly 150–170) with soft shading, informing preprocessing away from harsh binary-only silhouettes.

## Preprocessing

**Training:** pixels scaled to `[0, 1]`; labels remapped to contiguous indices `0…23`.

**Webcam (critical):**

1. Detect hand landmarks  
2. Crop with padding  
3. Grayscale + CLAHE + mild blur  
4. Soft Otsu-based background suppression while preserving shading  
5. Center on square canvas, resize to 28×28, normalize toward dataset brightness  

Full-frame resize without cropping was deliberately avoided. A live **Model input** preview exposes the tensor the CNN sees.

## CNN architecture

```
Input 1×28×28
Conv2d 32, BN, ReLU, MaxPool → 14×14
Conv2d 64, BN, ReLU, MaxPool → 7×7
Conv2d 128, BN, ReLU, AdaptiveAvgPool → 128
Dropout → Linear 96 → ReLU → Dropout → Linear 24
```

Training used CrossEntropyLoss, Adam (`lr=1e-3`), batch size 128, 12 epochs, 10% validation split from the training CSV, CPU PyTorch.

## Model evaluation (actual metrics)

Produced by `python -m training.evaluate` on this project’s trained weights:

| Metric | Value |
|--------|------:|
| Test accuracy | **0.9353** |
| Macro precision | **0.9378** |
| Macro recall | **0.9358** |
| Macro F1 | **0.9325** |
| N test | 7172 |
| Best val accuracy during training | 0.9996 |

Artifacts: `results/metrics.json`, `results/classification_report.txt`, `results/confusion_matrix.png`, `results/training_curves.png`.

**Note:** Validation accuracy oscillated across epochs; the trainer keeps the best checkpoint. Test accuracy (~93.5%) is the honest generalization figure to cite. Webcam results are a separate, usually harder domain and must not be equated with this score.

## Word prediction approach

- Offline lexicon built from `wordfreq` top English words (`scripts/prepare_dictionary.py`)  
- SymSpell for compact lookup / near matches  
- Ranking uses Zipf frequency, prefix match bonus, mild length penalty, and light context boost from recently confirmed words  
- Example: prefix `NEE` → NEED, NEEDS, NEEDED, NEEDING, NEEDLE  

## Sentence generation approach

Intent detection over confirmed tokens (need/want, help, feeling, greeting, response, question, statement) plus templates such as:

- `NEED WATER` → `I need water.`  
- `WANT FOOD` → `I want food.`  
- `HELP` → `Please help me.`  

This is **not** ASL grammar translation. It only adds light English function words for speechability.

## Text-to-speech

`pyttsx3` runs locally in a worker thread with Speak / Stop controls. No cloud TTS APIs.

## Limitations

- Static fingerspelling only; no continuous ASL lexicon  
- No J/Z  
- Template sentences are shallow  
- Domain gap between MNIST crops and webcam hands  
- Thumbs-up can fail under occlusion or unusual camera angles  
- Not clinically validated with target users in this prototype phase  

## Future FYP improvements

1. **Dynamic ASL** — include motion signs (J, Z) and common lexical signs with sequence models (CTC / transformers on landmark streams).  
2. **Gloss → English** — paired corpora and seq2seq / encoder–decoder translation rather than English fingerspelling templates.  
3. **Pakistani Sign Language (PSL)** — collect/licensed PSL data; adapt landmark pipelines.  
4. **Urdu output** — Urdu sentence templates + offline Urdu TTS.  
5. **Conversational UX** — dialogue state, repair strategies, multi-user turn taking.  
6. **Edge deployment** — Raspberry Pi / Coral with quantized CNN or landmark-only classifiers.  
7. **Human evaluation** — studies with Deaf / Hard-of-Hearing participants on speed, error recovery, and acceptability.  
8. **Better webcam adaptation** — few-shot calibration, synthetic backgrounds, or self-supervised crop normalization.

## Conclusion

SignBridge delivers an end-to-end, offline fingerspelling assistant with a trained CNN (**93.53%** test accuracy), predictive words, gesture confirmation, sentence expansion, and speech. It is a solid FYP-ready prototype when presented with clear scope boundaries and measured results rather than inflated claims about full sign-language translation.
