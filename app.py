"""Live ASL fingerspelling translator.

Webcam -> MediaPipe hand landmarks -> letter classifier -> written text + spoken English.

Run:  python app.py            (add --camera 1 to use another camera)
Keys: q / Esc quit | Space add space | Backspace delete | c clear | s speak the whole sentence
How spelling works:
  - Hold a letter steady for ~0.7 s to type it.
  - To type the same letter twice, drop your hand briefly (or change shape) and sign it again.
  - Take your hand out of view for ~1.5 s to end a word; the word is spoken aloud.
"""
import argparse, os, platform, shutil, subprocess, sys, time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

from features import LetterClassifier, landmarks_to_features

HERE = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL = os.path.join(HERE, "models", "hand_landmarker.task")
CLASSIFIER = os.path.join(HERE, "models", "asl_classifier.npz")

HOLD_SECONDS = 0.7      # how long a letter must be held to be typed
MIN_CONFIDENCE = 0.6    # ignore predictions below this probability
WORD_GAP_SECONDS = 1.5  # no hand for this long -> end of word

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
                    (17, 18), (18, 19), (19, 20)]


class Speaker:
    """Speaks text without freezing the video (uses macOS `say`, else pyttsx3 if installed)."""

    def __init__(self):
        self.proc = None
        self.engine = None
        if platform.system() == "Darwin" and shutil.which("say"):
            self.mode = "say"
        else:
            try:
                import pyttsx3
                self.engine = pyttsx3.init()
                self.mode = "pyttsx3"
            except Exception:
                self.mode = None

    def speak(self, text):
        text = text.strip()
        if not text or self.mode is None:
            return
        if self.mode == "say":
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
            self.proc = subprocess.Popen(["say", text])
        else:
            self.engine.say(text)
            self.engine.runAndWait()


class Speller:
    """Turns a noisy stream of per-frame predictions into typed letters and words."""

    def __init__(self, on_word):
        self.text = ""
        self.on_word = on_word
        self.candidate, self.since = None, 0.0
        self.last_typed = None
        self.last_hand_time = time.time()

    def update(self, letter, conf, now):
        if letter is None:  # no hand visible
            self.candidate, self.last_typed = None, None
            if self.text and not self.text.endswith(" ") and now - self.last_hand_time > WORD_GAP_SECONDS:
                self.end_word()
            return 0.0
        self.last_hand_time = now
        if conf < MIN_CONFIDENCE:
            self.candidate = None
            return 0.0
        if letter != self.candidate:
            self.candidate, self.since = letter, now
        held = now - self.since
        if letter != self.last_typed and held >= 0.3:
            self.last_typed = None  # a different shape was held, so repeats are allowed again
        if held >= HOLD_SECONDS and letter != self.last_typed:
            self.text += letter
            self.last_typed = letter
        return min(held / HOLD_SECONDS, 1.0)

    def end_word(self):
        words = self.text.split()
        if words:
            self.on_word(words[-1])
        self.text = self.text.rstrip() + " "

    def backspace(self):
        self.text = self.text[:-1]


def draw_hand(frame, pts):
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (255, 255, 255), 2)
    for p in pts:
        cv2.circle(frame, tuple(p), 4, (0, 140, 255), -1)


def draw_ui(frame, letter, conf, progress, text):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)
    label = f"{letter}  {conf * 100:.0f}%" if letter else "show your hand"
    cv2.putText(frame, label, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    if progress > 0:
        cv2.rectangle(frame, (w - 220, 25), (w - 220 + int(200 * progress), 45), (0, 200, 120), -1)
        cv2.rectangle(frame, (w - 220, 25), (w - 20, 45), (200, 200, 200), 1)
    cv2.rectangle(frame, (0, h - 70), (w, h), (30, 30, 30), -1)
    shown = text[-40:] + ("_" if int(time.time() * 2) % 2 else " ")
    cv2.putText(frame, shown, (15, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(CLASSIFIER):
        sys.exit("models/asl_classifier.npz not found - run `python train.py` first.")
    clf = LetterClassifier(CLASSIFIER)

    speaker = Speaker()
    speller = Speller(on_word=speaker.speak)
    if speaker.mode is None:
        print("Text-to-speech not available; showing text only.")

    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("Could not open the camera. On macOS, allow camera access for your Terminal/IDE "
                 "in System Settings > Privacy & Security > Camera.")

    recent = deque(maxlen=5)  # average probabilities over a few frames to reduce flicker
    start = time.time()
    with vision.HandLandmarker.create_from_options(opts) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror view, like a selfie camera
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = int((time.time() - start) * 1000)
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            letter, conf = None, 0.0
            if res.hand_landmarks:
                pts = np.array([[int(p.x * w), int(p.y * h)] for p in res.hand_landmarks[0]])
                draw_hand(frame, pts)
                recent.append(clf.predict_proba([landmarks_to_features(pts)])[0])
                avg = np.mean(recent, axis=0)
                i = int(np.argmax(avg))
                letter, conf = clf.classes_[i], float(avg[i])
            else:
                recent.clear()

            progress = speller.update(letter, conf, time.time())
            draw_ui(frame, letter, conf, progress, speller.text)
            cv2.imshow("ASL Translator", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                speller.end_word()
            elif key in (8, 127):
                speller.backspace()
            elif key == ord("c"):
                speller.text = ""
            elif key == ord("s"):
                speaker.speak(speller.text)

    cap.release()
    cv2.destroyAllWindows()
    if speller.text.strip():
        print("Final text:", speller.text.strip())


if __name__ == "__main__":
    main()
