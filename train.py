"""Train the ASL alphabet classifier on hand-landmark features.

Data:
  data/akram_keypoints.csv  - 36k pre-extracted keypoint rows (label index, 42 features)
  data/utec_landmarks.csv   - landmarks extracted from the UTEC ASL image dataset (optional)

Usage:
  python train.py            # evaluate, then train final model -> models/asl_classifier.npz
"""
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from features import LETTERS, landmarks_to_features, mirror

HERE = os.path.dirname(os.path.abspath(__file__))


def load_akram():
    rows = []
    for line in open(os.path.join(HERE, "data", "akram_keypoints.csv"), encoding="utf-8-sig"):
        p = line.split(",")
        if len(p) == 43:  # skip the one malformed row in the source file
            rows.append([float(v) for v in p])
    raw = np.array(rows)
    y = np.array([LETTERS[int(i)] for i in raw[:, 0]])
    return raw[:, 1:].astype(np.float32), y


def load_utec():
    path = os.path.join(HERE, "data", "utec_landmarks.csv")
    if not os.path.exists(path):
        return None, None
    X, y = [], []
    for line in open(path):
        p = line.strip().split(",")
        w, h = float(p[1]), float(p[2])
        v = np.array(p[4:], dtype=np.float32).reshape(21, 3)
        X.append(landmarks_to_features(v[:, :2] * [w, h]))
        y.append(p[0])
    return np.array(X), np.array(y)


def with_mirrors(X, y):
    return np.vstack([X, mirror(X)]), np.concatenate([y, y])


def make_model():
    return MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=400,
                         early_stopping=True, random_state=0)


def main():
    Xa, ya = load_akram()
    Xu, yu = load_utec()
    print(f"Akram keypoints: {len(Xa)} samples | UTEC images: {0 if Xu is None else len(Xu)} samples")

    # 1) In-dataset check (optimistic: train/test frames come from the same recordings)
    Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.2, stratify=ya, random_state=0)
    m = make_model().fit(*with_mirrors(Xtr, ytr))
    print(f"\n[1] Akram random split accuracy: {accuracy_score(yte, m.predict(Xte)):.3f}")

    if Xu is not None:
        # 2) Cross-dataset check: train on Akram only, test on different people/cameras (UTEC)
        m = make_model().fit(*with_mirrors(Xa, ya))
        pu = m.predict(Xu)
        print(f"[2] Train Akram -> test UTEC (unseen people) accuracy: {accuracy_score(yu, pu):.3f}")
        # 3) Both sources, held-out 25% of UTEC
        Xu_tr, Xu_te, yu_tr, yu_te = train_test_split(Xu, yu, test_size=0.25, stratify=yu, random_state=0)
        m = make_model().fit(*with_mirrors(np.vstack([Xa, Xu_tr]), np.concatenate([ya, yu_tr])))
        pu = m.predict(Xu_te)
        print(f"[3] Train Akram+75% UTEC -> test 25% UTEC accuracy: {accuracy_score(yu_te, pu):.3f}\n")
        print(classification_report(yu_te, pu, zero_division=0))

    # Final model on everything
    X = Xa if Xu is None else np.vstack([Xa, Xu])
    y = ya if Xu is None else np.concatenate([ya, yu])
    final = make_model().fit(*with_mirrors(X, y))
    out = os.path.join(HERE, "models", "asl_classifier.npz")
    arrays = {f"W{i}": w for i, w in enumerate(final.coefs_)}
    arrays.update({f"b{i}": b for i, b in enumerate(final.intercepts_)})
    np.savez(out, n_layers=len(final.coefs_), classes=final.classes_, **arrays)
    print(f"Saved final model ({len(X)} samples x2 mirrored) -> {out}")


if __name__ == "__main__":
    main()
