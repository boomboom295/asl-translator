# ASL Translator (fingerspelling → text + speech)

Signs the ASL alphabet into your webcam and turns it into written and spoken English.

```
webcam frame ─► MediaPipe hand tracker (21 points on your hand)
             ─► 42 numbers (positions relative to the wrist, scaled)
             ─► small neural network ─► letter A–Z + confidence
             ─► speller (hold to type, pause to end a word) ─► text on screen + `say` aloud
```

Using hand landmarks instead of raw pixels makes the model small (64 KB), fast on a laptop CPU,
and much less sensitive to lighting, skin tone and background.

## Setup (macOS, one time)

Use Python 3.10–3.12 (MediaPipe may not support the newest Python yet).

```bash
cd ~/PycharmProjects/asl-translator
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

The first time, macOS asks for camera permission for Terminal (or PyCharm) – allow it,
then run again. Use `--camera 1` if the wrong camera opens.

| Action | How |
|---|---|
| Type a letter | Hold the sign steady until the green bar fills (~0.7 s) |
| Double letter (the "LL" in HELLO) | Drop your hand for a moment, then sign it again |
| End a word (spoken aloud) | Hand out of view ~1.5 s, or press **Space** |
| Speak the whole sentence | **s** |
| Delete / clear | **Backspace** / **c** |
| Quit | **q** or **Esc** |

Tips: keep your whole hand in frame, palm toward the camera, about arm's length away.

## Files

| File | What it does |
|---|---|
| `app.py` | Live webcam translator |
| `features.py` | Turns landmarks into features; runs the trained network with numpy |
| `train.py` | Trains + evaluates the classifier, writes `models/asl_classifier.npz` |
| `extract_landmarks.py` | Runs MediaPipe over a folder-per-letter image dataset → landmark CSV |
| `models/hand_landmarker.task` | Google MediaPipe hand-tracking model |
| `models/asl_classifier.npz` | The trained letter classifier |
| `data/*.csv` | Hand-landmark training data (no images) |

## Website version (runs in the browser)

The `docs/` folder is a website version of the app. It does the same thing as `app.py`
(same hand tracking, same trained model, same hold-to-type rules), but runs entirely in the
visitor's browser. The video never leaves their computer, and speech uses the browser's
built-in voice, so it works on Windows, Mac, Chromebooks and phones.

**Host it free on GitHub Pages:** on GitHub open the repo -> **Settings** -> **Pages** ->
under *Build and deployment* choose **Deploy from a branch**, branch **main**, folder **/docs** -> **Save**.
After a minute the site is live at `https://<your-username>.github.io/asl-translator/`.

**Try it on your own computer first:**

```bash
cd docs
python -m http.server 8000
```

Then open http://localhost:8000 (the camera only works over `http://localhost` or `https://`).

**After retraining** (`python train.py`), run `python export_web_model.py` to copy the new model
into `docs/models/`, then commit and push.

| File | What it does |
|---|---|
| `docs/index.html`, `docs/style.css` | The page |
| `docs/app.js` | Camera, MediaPipe hand tracking (loaded from the jsDelivr CDN), drawing, speech |
| `docs/core.js` | Features, the neural network and the speller, ported from Python |
| `docs/models/` | Copies of the trained model, written by `export_web_model.py` |

## Words and moving letters (J, Z)

The website has two modes (switch with the buttons or the **W** key):

- **Letters** - fingerspelling. Handshape letters are typed by holding them; **J** and **Z** are
  recognised from the movement, using a second small model.
- **Words** - 54 common signs (hello, thank you, please, sorry, yes, no, help, family, ...).
  Sign one word, then lower your hands. If the guess is wrong, tap one of the other guesses.

How it works: the site tracks both hands and the face. A sign becomes 16 snapshots of where each
hand is relative to the face and what shape it makes (`seqfeat.py`, mirrored in `docs/sequence.js`),
and a neural network picks the sign. `docs/segment.js` decides when a sign starts and ends.

| File | What it does |
|---|---|
| `tools/extract_video_landmarks.js` | Runs in a browser tab: turns sign videos into hand/face landmark data |
| `data/asl_video_landmarks.json` | That data: 2,269 clips of 54 words + the alphabet |
| `seqfeat.py` | Features for moving signs (same maths as `docs/sequence.js`) |
| `synth_motion.py` | Makes extra synthetic J/Z clips (the videos have very few) |
| `train_signs.py` | Trains + tests the word model and the moving-letter model |
| `models/word_classifier.npz`, `models/motion_classifier.npz` | The trained models (`docs/models/*.json` for the website) |

Results on held-out video clips (recordings the model never saw): words about 73% right on the
first guess and about 89% with the right word in the top 3. The letter model was also retrained with
frames from the video clips (many more signers): on new signers it went from 45% to 73%, and in those test clips P went
from almost never to nearly always right. J/Z are trained mostly on synthetic movements, so they are the
least tested - real-world feedback helps.

Video source: [ASL dataset on Hugging Face](https://huggingface.co/datasets/akasheroor/American-Sign-Language-Dataset)
(MIT licence, collected from several sources). Only landmarks are stored here, not the videos.
About half of its clips use a video format browsers can't decode, so they were skipped.

## Datasets

1. **ASL keypoint dataset** – 36,401 hand-landmark samples for A–Z, from
   [AkramOM606/American-Sign-Language-Detection](https://github.com/AkramOM606/American-Sign-Language-Detection) (MIT license).
2. **UTEC ASL image dataset** – 2,256 photos of 24 letters (no J/Z) from several students, from
   [cristian20a/ASL_Dataset](https://github.com/cristian20a/ASL_Dataset). Its landmark file
   (`data/utec_landmarks.csv`) is not in this repo because the dataset can't be redistributed; to
   rebuild it, clone that repo and run `python extract_landmarks.py ASL_Dataset data/utec_landmarks.csv`.
   `train.py` works without it (keypoint dataset only). Free for research/educational use only; cite:
   C. Amaya and V. Murray, "Real-Time Sign Language Recognition," IEEE INTERCON 2020,
   doi:10.1109/INTERCON50315.2020.9220241.

## Results (from `python train.py`)

| Test | Accuracy |
|---|---|
| Random 20% of the keypoint dataset | 91.7% |
| Trained on keypoint dataset only → tested on UTEC (new people, new cameras) | 77.5% |
| Trained on both → tested on held-out 25% of UTEC | 95.0% |

The second number is the most honest guide to how it will work for someone it has never seen.
Weakest letters are the closed-fist ones that look alike from the front: **M, N, S, T, E** and **D**.
**J** and **Z** involve motion; here they're recognized from a single frame of the sign,
so they're less reliable.

## Making it better with your own hand

The single biggest improvement is adding ~50–100 photos per letter of *your* hand, in *your* room:

1. Put photos in folders `my_data/A/…`, `my_data/B/…` etc.
2. `python extract_landmarks.py my_data data/my_landmarks.csv`
3. Add that CSV to `load_utec()` in `train.py` (same format), then `python train.py`.

## Next steps

- Word-level signs (not just spelling) using sequences of landmarks over time – e.g. the WLASL dataset with an LSTM/Transformer.
- Autocorrect typed words against a dictionary.
- Browser version with MediaPipe JS so others can try it without installing Python.
