"""Features for signs that move (words, and the letters J and Z).

Each video frame gives a raw 92-number vector (same layout the website produces):
  [0:4]   face box center x, center y, width, height   (0..1 of the frame; zeros = no face)
  then 2 hand slots of 44 numbers:
          present, handedness label (unused), 21 x values, 21 y values  (0..1 of the frame)

A sign becomes a fixed-size vector:
  1. trim to the frames where a hand is visible
  2. resample to T frames
  3. per frame and per hand: where the hand is relative to the face (scaled by face width)
     and its shape (points relative to the wrist, scaled like the letter model)
The website's sequence.js does exactly the same steps.
"""
import numpy as np

T = 16                 # frames per sign after resampling
PER_FRAME = 92
HAND_FEATS = 47        # present + wrist(2) + middle knuckle(2) + shape(42)
FRAME_FEATS = 2 * HAND_FEATS
DEFAULT_FACE = (0.5, 0.3, 0.18)   # used when no face was ever found (x, y, width in frame-height units)


def hand_present(f):
    return f[:, 4] > 0.5, f[:, 48] > 0.5


def trim(frames):
    """Keep the span from the first to the last frame with a hand."""
    a, b = hand_present(frames)
    idx = np.where(a | b)[0]
    if len(idx) == 0:
        return frames[:0]
    return frames[idx[0]: idx[-1] + 1]


def fill_faces(frames, aspect):
    """Per-frame face (x, y, width) in frame-height units; gaps filled from the nearest detected face."""
    n = len(frames)
    has = frames[:, 2] > 0
    out = np.zeros((n, 3), np.float32)
    if not has.any():
        out[:] = DEFAULT_FACE
        out[:, 0] *= aspect
        return out
    idx = np.where(has)[0]
    near = idx[np.abs(np.arange(n)[:, None] - idx[None, :]).argmin(1)]
    out[:, 0] = frames[near, 0] * aspect
    out[:, 1] = frames[near, 1]
    out[:, 2] = np.maximum(frames[near, 2] * aspect, 1e-3)
    return out


def resample_idx(n, t=T):
    """Evenly spaced frame indices (exact integer rounding, identical in sequence.js)."""
    if n <= 1:
        return np.zeros(t, int)
    k = np.arange(t)
    return (2 * k * (n - 1) + (t - 1)) // (2 * (t - 1))


def hand_block(xs, ys, face):
    """xs, ys: (21,) in frame-height units. face: (x, y, w). Returns 47 features."""
    fx, fy, fw = face
    v = np.zeros(HAND_FEATS, np.float32)
    v[0] = 1
    v[1], v[2] = (xs[0] - fx) / fw, (ys[0] - fy) / fw
    v[3], v[4] = (xs[9] - fx) / fw, (ys[9] - fy) / fw
    rel = np.stack([xs - xs[0], ys - ys[0]], 1).reshape(-1)
    m = np.abs(rel).max()
    v[5:] = rel / m if m > 0 else rel
    return v


def frame_features(f, face, aspect):
    """One raw frame -> 94 features. Hands go into slots by image position (left of image first)."""
    hands = []
    for o in (4, 48):
        if f[o] > 0.5:
            xs = f[o + 2: o + 23] * aspect
            ys = f[o + 23: o + 44]
            hands.append((xs, ys))
    out = np.zeros(FRAME_FEATS, np.float32)
    if len(hands) == 2:
        hands.sort(key=lambda h: h[0][0])
        slots = [hands[0], hands[1]]
    elif len(hands) == 1:
        slots = [hands[0], None] if hands[0][0][0] <= face[0] else [None, hands[0]]
    else:
        slots = [None, None]
    for k, h in enumerate(slots):
        if h is not None:
            out[k * HAND_FEATS:(k + 1) * HAND_FEATS] = hand_block(h[0], h[1], face)
    return out


def sequence_features(frames, aspect):
    """frames: (n, 92) raw. aspect: frame width / height. Returns (T * 94,) or None if no hands."""
    frames = trim(np.asarray(frames, np.float32))
    if len(frames) == 0:
        return None
    faces = fill_faces(frames, aspect)
    idx = resample_idx(len(frames))
    return np.concatenate([frame_features(frames[i], faces[i], aspect) for i in idx])


# ---------- augmentation on raw frames (before features) ----------

def _coords(frames):
    """Views of all x and y columns (face center + 42 per hand)."""
    xcols = [0] + [o + 2 + i for o in (4, 48) for i in range(21)]
    ycols = [1] + [o + 23 + i for o in (4, 48) for i in range(21)]
    return np.array(xcols), np.array(ycols)


XC, YC = _coords(None)


def mirror_frames(frames):
    f = frames.copy()
    for o in (4, 48):
        on = f[:, o] > 0.5
        f[on, o + 2: o + 23] = 1 - f[on, o + 2: o + 23]
    face = f[:, 2] > 0
    f[face, 0] = 1 - f[face, 0]
    return f


def augment(frames, rng, aspect):
    f = np.asarray(frames, np.float32).copy()
    if rng.random() < 0.5:
        f = mirror_frames(f)
    # zoom / shift / rotate everything (as if the camera were a bit closer, farther or tilted)
    s = rng.uniform(0.85, 1.15)
    dx, dy = rng.uniform(-0.08, 0.08, 2)
    ang = np.deg2rad(rng.uniform(-10, 10))
    c, si = np.cos(ang), np.sin(ang)
    for o in (4, 48):
        on = f[:, o] > 0.5
        if not on.any():
            continue
        x = (f[on, o + 2: o + 23] - 0.5) * aspect
        y = f[on, o + 23: o + 44] - 0.5
        xr, yr = c * x - si * y, si * x + c * y
        f[on, o + 2: o + 23] = (xr * s) / aspect + 0.5 + dx
        f[on, o + 23: o + 44] = yr * s + 0.5 + dy
    face = f[:, 2] > 0
    if face.any():
        x = (f[face, 0] - 0.5) * aspect
        y = f[face, 1] - 0.5
        f[face, 0] = (c * x - si * y) * s / aspect + 0.5 + dx
        f[face, 1] = (si * x + c * y) * s + 0.5 + dy
        f[face, 2:4] *= s
    # small jitter on hand points (tracking noise)
    for o in (4, 48):
        on = f[:, o] > 0.5
        f[on, o + 2: o + 44] += rng.normal(0, 0.003, (on.sum(), 42)).astype(np.float32)
    # time: drop some frames, and trim a little off the start/end of the hand-visible span
    t = trim(f)
    if len(t) > 6:
        a = rng.integers(0, max(1, len(t) // 8))
        b = len(t) - rng.integers(0, max(1, len(t) // 8))
        t = t[a:b]
        keep = rng.random(len(t)) > 0.15
        keep[0] = keep[-1] = True
        t = t[keep]
    # occasionally lose a hand for a frame (tracking dropouts)
    if len(t) > 4 and rng.random() < 0.3:
        k = rng.integers(1, len(t) - 1)
        t[k, 4] = 0
    return t
