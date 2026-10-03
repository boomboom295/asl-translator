"""Train the two models for signs that move:

  words  - ASL word signs (from the video dataset)
  motion - the alphabet as movements, used in Letters mode to catch J and Z
           (video clips of all letters + synthetic J/Z clips, see synth_motion.py)

Input:  data/asl_video_landmarks.json  (made in the browser by tools/extract_video_landmarks.js)
Usage:  python train_signs.py            evaluate on held-out videos, then train + save both models
        python train_signs.py --eval-only
Writes: models/{word,motion}_classifier.npz and docs/models/{word,motion}_classifier.json
"""
import base64, json, os, re, sys, collections
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import seqfeat as sf
import synth_motion
from train import load_akram

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "asl_video_landmarks.json")
AUG_PER_VIDEO = 24
SEED = 0

LABELS = {"thankyou": "thank you"}   # how a word is written/spoken


def load_videos(path=DATA):
    d = json.load(open(path))
    vids = []
    for v in d["videos"]:
        a = np.frombuffer(base64.b64decode(v["data"]), dtype="<i2").astype(np.float32) / d["scale"]
        frames = a.reshape(v["n"], d["perFrame"])
        if not (frames[:, 4] > 0.5).any() and not (frames[:, 48] > 0.5).any():
            continue
        vids.append({"word": v["word"], "file": v["file"], "aspect": v["w"] / v["h"], "frames": frames})
    return vids


def source_key(fname):
    """Group files that look like copies/edits of the same recording, so the test split is fair."""
    s = fname.lower()
    s = re.sub(r"\.mp4$", "", s)
    s = re.sub(r"(_\d+)+$", "", s)          # eat_20241119_172641_3 -> eat_20241119_172641
    s = re.sub(r"_video$", "", s)
    return s


def build(vids, rng, n_aug):
    X, y, g = [], [], []
    for gi, v in enumerate(vids):
        base = sf.sequence_features(v["frames"], v["aspect"])
        if base is None:
            continue
        X.append(base); y.append(v["word"]); g.append(gi)
        for _ in range(n_aug if rng is not None else 0):
            f = sf.augment(v["frames"], rng, v["aspect"])
            x = sf.sequence_features(f, v["aspect"])
            if x is not None:
                X.append(x); y.append(v["word"]); g.append(gi)
    return np.array(X, np.float32), np.array(y), np.array(g)


def make_model(hidden=(512, 256)):
    return MLPClassifier(hidden_layer_sizes=hidden, alpha=1e-2, batch_size=256,
                         learning_rate_init=1e-3, max_iter=200, early_stopping=True,
                         n_iter_no_change=12, random_state=SEED)


def split(vids, frac=0.2, seed=SEED):
    """Hold out whole source groups per word."""
    rng = np.random.default_rng(seed)
    by_word = collections.defaultdict(lambda: collections.defaultdict(list))
    for i, v in enumerate(vids):
        by_word[v["word"]][source_key(v["file"])].append(i)
    tr, te = [], []
    for w, groups in by_word.items():
        keys = sorted(groups)
        rng.shuffle(keys)
        k = max(1, int(round(len(keys) * frac))) if len(keys) >= 3 else 0
        for j, key in enumerate(keys):
            (te if j < k else tr).extend(groups[key])
    return tr, te


def export(model, mean, std, path_npz, path_json):
    np.savez(path_npz, n_layers=len(model.coefs_), classes=model.classes_, mean=mean, std=std, T=sf.T,
             **{f"W{i}": w for i, w in enumerate(model.coefs_)},
             **{f"b{i}": b for i, b in enumerate(model.intercepts_)})
    js = {
        "T": sf.T,
        "classes": [str(c) for c in model.classes_],
        "labels": {str(c): LABELS.get(str(c), str(c)) for c in model.classes_},
        "mean": np.round(mean, 5).tolist(), "std": np.round(std, 5).tolist(),
        # weights as base64 little-endian float32 (row-major), much smaller than JSON numbers
        "layers": [{"rows": W.shape[0], "cols": W.shape[1],
                    "W": base64.b64encode(W.astype("<f4").tobytes()).decode(),
                    "b": base64.b64encode(b.astype("<f4").tobytes()).decode()}
                   for W, b in zip(model.coefs_, model.intercepts_)],
    }
    os.makedirs(os.path.dirname(path_json), exist_ok=True)
    with open(path_json, "w") as f:
        json.dump(js, f, separators=(",", ":"))


def fit(train_sets, rng):
    """train_sets: list of (videos, augmentations per video)."""
    parts = [build(v, rng, n) for v, n in train_sets if v]
    X = np.vstack([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    mean, std = X.mean(0), X.std(0) + 1e-4
    return X, y, mean, std


def per_class(y, pred):
    acc = collections.defaultdict(list)
    for t, q in zip(y, pred):
        acc[t].append(t == q)
    return sorted(acc.items(), key=lambda kv: np.mean(kv[1]))


def main():
    eval_only = "--eval-only" in sys.argv
    vids = load_videos()
    words = [v for v in vids if len(v["word"]) > 1]
    letters = [v for v in vids if len(v["word"]) == 1]
    print(f"{len(words)} word videos ({len({v['word'] for v in words})} words), {len(letters)} letter videos")
    Xa, ya = load_akram()
    shapes = synth_motion.akram_shapes(Xa, ya)
    synth = synth_motion.synth_set(shapes, np.random.default_rng(SEED + 7))
    print(f"+ {len(synth)} synthetic moving-letter clips (J, Z and ordinary hand movement)")

    # ---------- words ----------
    tr, te = split(words)
    X, y, mean, std = fit([([words[i] for i in tr], AUG_PER_VIDEO)], np.random.default_rng(SEED))
    m = make_model().fit((X - mean) / std, y)
    Xte, yte, _ = build([words[i] for i in te], None, 0)
    p = m.predict_proba((Xte - mean) / std)
    pred = m.classes_[p.argmax(1)]
    top3 = np.mean([yte[i] in m.classes_[np.argsort(-p[i])[:3]] for i in range(len(yte))])
    print(f"\n[words] held-out videos: {len(yte)} (recordings not used for training)")
    print(f"  top-1 accuracy: {accuracy_score(yte, pred):.3f}   right word in top 3: {top3:.3f}")
    print("  hardest:", ", ".join(f"{k} {np.mean(v):.0%}" for k, v in per_class(yte, pred)[:8]))

    # ---------- motion letters ----------
    tr, te = split(letters)
    X, y, mean2, std2 = fit([([letters[i] for i in tr], AUG_PER_VIDEO), (synth, 3)], np.random.default_rng(SEED))
    m2 = make_model((256, 128)).fit((X - mean2) / std2, y)
    Xte, yte, _ = build([letters[i] for i in te], None, 0)
    pred = m2.predict((Xte - mean2) / std2)
    jz = np.isin(yte, ["j", "z"])
    print(f"\n[motion letters] held-out letter videos: {len(yte)} (only {jz.sum()} are J/Z)")
    for L in "jz":
        s_ = yte == L
        if s_.any():
            print(f"  real {L.upper()} clips recognised: {np.mean(pred[s_] == L):.2f} of {s_.sum()}  {collections.Counter(pred[s_]).most_common(3)}")
    print(f"  other letters wrongly read as J/Z: {np.mean(np.isin(pred[~jz], ['j', 'z'])):.2f} of {(~jz).sum()}")
    sx = synth_motion.synth_set(shapes, np.random.default_rng(99), 100, 100, 300)
    Xs, ys_, _ = build(sx, None, 0)
    ps = m2.predict((Xs - mean2) / std2)
    for L in "jz":
        print(f"  fresh synthetic {L.upper()} recognised: {np.mean(ps[ys_ == L] == L):.2f}")
    s_ = ~np.isin(ys_, ["j", "z"])
    print(f"  fresh synthetic ordinary movement read as J/Z: {np.mean(np.isin(ps[s_], ['j', 'z'])):.2f}")

    if eval_only:
        return
    print("\nTraining final models on all data…")
    X, y, mean, std = fit([(words, AUG_PER_VIDEO)], np.random.default_rng(SEED + 1))
    final = make_model().fit((X - mean) / std, y)
    export(final, mean, std, os.path.join(HERE, "models", "word_classifier.npz"),
           os.path.join(HERE, "docs", "models", "word_classifier.json"))
    X, y, mean2, std2 = fit([(letters, AUG_PER_VIDEO), (synth, 3)], np.random.default_rng(SEED + 1))
    final2 = make_model((256, 128)).fit((X - mean2) / std2, y)
    export(final2, mean2, std2, os.path.join(HERE, "models", "motion_classifier.npz"),
           os.path.join(HERE, "docs", "models", "motion_classifier.json"))
    print("Saved models/word_classifier.npz, models/motion_classifier.npz and their docs/models/*.json copies")


if __name__ == "__main__":
    main()
