# SignBridge — Live Demonstration Script (≈4 minutes)

## Setup (30s)
1. Activate the environment and launch:
   ```bash
   cd signbridge
   source .venv/bin/activate
   python app.py
   ```
2. Click **Start Camera**. Confirm the preview mirrors you and “Hand: …” appears when your hand is visible.
3. Open **Diagnostics** briefly to show real test metrics, then close it.

## Live recognition (60s)
1. Hold a clear ASL letter (e.g. **A**, then **L**) in front of the camera with a plain background if possible.
2. Point out: current letter, confidence %, stability bar, and the small **Model input** crop (hand cropped — not the full frame).
3. Spell **N → E → E** slowly. Show letters appearing in **Partial**.
4. Mention: holding a pose does not spam the same letter; briefly drop the hand (or move) to repeat a letter (e.g. the two L’s in HELLO).

## Predictive suggestions (45s)
1. With partial `NEE`, highlight suggestions such as **NEED | NEEDLE | NEEDS**.
2. Click **NEED** (or give a clear thumbs-up) to confirm without spelling D.
3. Show **Words:** update and the auto sentence **“I need water.”** after confirming WATER the same way (or spell W-A-T-E-R / pick WATER).

## Sentence + speech (45s)
1. Edit the sentence box if desired.
2. Click **Speak**, then **Stop** if needed.
3. Use **Undo Word** / **Clear Sentence** to show recovery.

## Honest close (30s)
- Prototype for **fingerspelled English**, not full ASL grammar.
- J and Z are unsupported (motion signs).
- Webcam accuracy can be lower than the ~93.5% test-set score due to domain gap; the debug crop helps diagnose that.

## Backup if camera is weak
Demonstrate word confirmation + sentence + Speak using **Undo Letter** / typed flow is not available — instead pre-confirm via suggestion clicks after any partial letters, or show unit-test / Diagnostics metrics and a recorded clip if prepared.
