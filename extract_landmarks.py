"""Run MediaPipe HandLandmarker over a folder-per-letter image dataset and save landmarks.

Usage: python extract_landmarks.py <dataset_dir> <out.csv>
Output columns: label, img_w, img_h, handedness, x0,y0,z0 ... x20,y20,z20 (MediaPipe normalized coords)
"""
import csv, os, sys
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")


def main(src, out):
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
    )
    found = missed = 0
    with vision.HandLandmarker.create_from_options(opts) as lm, open(out, "w", newline="") as f:
        w = csv.writer(f)
        for label in sorted(os.listdir(src)):
            d = os.path.join(src, label)
            if not os.path.isdir(d) or len(label) != 1:
                continue
            for name in sorted(os.listdir(d)):
                if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img = cv2.imread(os.path.join(d, name))
                if img is None:
                    continue
                s = 960 / max(img.shape[:2])
                if s < 1:
                    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                h, wd = img.shape[:2]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if not res.hand_landmarks:
                    missed += 1
                    continue
                found += 1
                pts = res.hand_landmarks[0]
                hand = res.handedness[0][0].category_name
                w.writerow([label, wd, h, hand] + [round(v, 5) for p in pts for v in (p.x, p.y, p.z)])
            print(label, "done", found, missed, flush=True)
    print(f"hands found in {found} images, missed {missed}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
