"""Train the ASL alphabet classifier on hand-landmark features.

Data:
  data/akram_keypoints.csv  - 36k pre-extracted keypoint rows (label index, 42 features)
  data/utec_landmarks.csv   - landmarks extracted from the UTEC ASL image dataset (optional)
  data/asl_video_landmarks.json - letter clips from the video dataset (optional; many different signers)

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


VIDEO_WEIGHT = 5   # video frames are repeated: they are fewer but come from many more people


def load_video_letters(only=None):
    """Handshape frames from the letter clips of the video dataset (middle of each clip)."""
    if not os.path.exists(os.path.join(HERE, "data", "asl_video_landmarks.json")):
        return None, None, None
    import train_signs, seqfeat
    vids = [v for v in train_signs.load_videos() if len(v["word"]) == 1]
    if only is not None:
        vids = [vids[i] for i in only]
    X, y, g = [], [], []
    for gi, v in enumerate(vids):
        f = seqfeat.trim(v["frames"])
        n = len(f)
        for fr in f[int(n * 0.3): int(n * 0.8) + 1]:
            if fr[4] > 0.5 and fr[48] < 0.5:
                pts = np.stack([fr[6:27] * v["aspect"], fr[27:48]], 1)
                X.append(landmarks_to_features(pts)); y.append(v["word"].upper()); g.append(gi)
    return np.array(X, np.float32), np.array(y), vids


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

    Xv, yv, _ = load_video_letters()
    if Xv is not None:
        # 4) New signers: hold out whole video clips, test on their frames
        import train_signs
        vids = [v for v in train_signs.load_videos() if len(v["word"]) == 1]
        tr, te = train_signs.split(vids)
        Xvt, yvt, _ = load_video_letters(tr)
        Xve, yve, _ = load_video_letters(te)
        base = np.vstack([Xa] + ([] if Xu is None else [Xu]))
        yb = np.concatenate([ya] + ([] if Xu is None else [yu]))
        m = make_model().fit(*with_mirrors(base, yb))
        print(f"[4] Video signers, model without video frames: {accuracy_score(yve, m.predict(Xve)):.3f}")
        m = make_model().fit(*with_mirrors(np.vstack([base] + [Xvt] * VIDEO_WEIGHT),
                                           np.concatenate([yb] + [yvt] * VIDEO_WEIGHT)))
        pv = m.predict(Xve)
        print(f"[5] Video signers, model with other clips' frames: {accuracy_score(yve, pv):.3f}  (P: {np.mean(pv[yve == 'P'] == 'P'):.2f})")

    # Final model on everything
    X = Xa if Xu is None else np.vstack([Xa, Xu])
    y = ya if Xu is None else np.concatenate([ya, yu])
    if Xv is not None:
        X = np.vstack([X] + [Xv] * VIDEO_WEIGHT)
        y = np.concatenate([y] + [yv] * VIDEO_WEIGHT)
    final = make_model().fit(*with_mirrors(X, y))
    out = os.path.join(HERE, "models", "asl_classifier.npz")
    arrays = {f"W{i}": w for i, w in enumerate(final.coefs_)}
    arrays.update({f"b{i}": b for i, b in enumerate(final.intercepts_)})
    np.savez(out, n_layers=len(final.coefs_), classes=final.classes_, **arrays)
    print(f"Saved final model ({len(X)} samples x2 mirrored) -> {out}")


if __name__ == "__main__":
    main()
