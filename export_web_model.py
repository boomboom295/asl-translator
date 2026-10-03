"""Copies the trained model into the website folder (docs/) so the browser version can use it.

Run after `python train.py` whenever you retrain:  python export_web_model.py
Writes docs/models/asl_classifier.json and copies models/hand_landmarker.task.
(The word and J/Z models are written straight to docs/models/ by train_signs.py.)
"""
import json, os, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "models")
DST = os.path.join(HERE, "docs", "models")


def main():
    os.makedirs(DST, exist_ok=True)
    d = np.load(os.path.join(SRC, "asl_classifier.npz"))
    n = int(d["n_layers"])
    model = {
        "classes": [str(c) for c in d["classes"]],
        "layers": [{"W": np.round(d[f"W{i}"], 6).tolist(), "b": np.round(d[f"b{i}"], 6).tolist()}
                   for i in range(n)],
    }
    with open(os.path.join(DST, "asl_classifier.json"), "w") as f:
        json.dump(model, f, separators=(",", ":"))
    shutil.copyfile(os.path.join(SRC, "hand_landmarker.task"), os.path.join(DST, "hand_landmarker.task"))
    print("Wrote docs/models/asl_classifier.json and docs/models/hand_landmarker.task")


if __name__ == "__main__":
    main()
