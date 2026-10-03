"""Synthetic training examples for the letters that move (J and Z).

The video dataset has only a handful of J and Z clips, so we also make some:
take real handshapes (I for J, D for Z) from the alphabet data, place the hand next to a face
like in the videos, and move it along a J or Z path. We also make "moving but not J/Z"
examples (any other letter sliding around), so ordinary hand movement isn't mistaken for J or Z.
Everything is generated in the same raw 92-number frame layout the website produces.
"""
import numpy as np

FPS = 15


def akram_shapes(Xa, ya):
    """Akram rows (42 features, mirrored webcam view) -> dict letter -> (n, 21, 2) shapes
    in 'hand size' units (wrist at 0, wrist->middle knuckle = 1), un-mirrored."""
    out = {}
    for L in np.unique(ya):
        P = Xa[ya == L].reshape(-1, 21, 2).copy()
        P[:, :, 0] *= -1                                      # undo the webcam mirror
        size = np.linalg.norm(P[:, 9], axis=1, keepdims=True)[:, :, None]
        ok = size[:, 0, 0] > 1e-3
        out[str(L).lower()] = P[ok] / size[ok]
    return out


def _path(kind, rng):
    """Wrist offset path (k, 2) in hand-size units and hand rotation (k,) in radians."""
    k = int(rng.integers(10, 19))
    t = np.linspace(0, 1, k)
    if kind == "j":
        # down, then hook toward the signer's left (image right), twisting the wrist
        depth, hook = rng.uniform(0.9, 2.2), rng.uniform(0.6, 1.6)
        y = depth * np.sin(np.minimum(t / 0.7, 1) * np.pi / 2)
        x = hook * np.clip((t - 0.5) / 0.5, 0, 1) ** 1.5
        rot = np.deg2rad(rng.uniform(40, 90)) * t
        return np.stack([x, y], 1), rot
    if kind == "z":
        w, h = rng.uniform(1.4, 3.0), rng.uniform(1.1, 2.6)
        corners = np.array([[0, 0], [-w, 0], [0, h], [-w, h]])   # signer's right = image left
        seg = np.linspace(0, 3, k)
        i = np.minimum(seg.astype(int), 2)
        f = (seg - i)[:, None]
        return corners[i] * (1 - f) + corners[i + 1] * f, np.zeros(k)
    # ordinary movement: a straight or gently curved slide in a random direction
    ang = rng.uniform(0, 2 * np.pi)
    length = rng.uniform(1.2, 3.0)
    bend = rng.uniform(-0.5, 0.5)
    d = np.array([np.cos(ang), np.sin(ang)])
    n = np.array([-d[1], d[0]])
    p = (t[:, None] * length) * d + (np.sin(t * np.pi) * bend * length)[:, None] * n
    return p, np.deg2rad(rng.uniform(-20, 20)) * t


def make_sequence(shape, kind, rng, shape_end=None):
    """One synthetic clip as raw frames (n, 92)."""
    aspect = rng.choice([4 / 3, 16 / 9])
    fx, fy = rng.uniform(0.4, 0.6) * aspect, rng.uniform(0.22, 0.4)
    fw = rng.uniform(0.18, 0.32)                                # face width, frame-height units
    hs = fw * rng.uniform(0.38, 0.6)                            # hand size
    wx0 = fx + fw * rng.uniform(-1.7, -0.4)
    wy0 = fy + fw * rng.uniform(0.3, 1.6)
    path, rot = _path(kind, rng)
    pre, post = int(rng.integers(3, 9)), int(rng.integers(2, 7))
    path = np.vstack([np.repeat(path[:1], pre, 0), path, np.repeat(path[-1:], post, 0)])
    rot = np.concatenate([np.repeat(rot[:1], pre), rot, np.repeat(rot[-1:], post)])
    if rng.random() < 0.35:
        # like the videos: the hand first rises into place from below
        k = int(rng.integers(3, 7))
        rise = np.linspace(rng.uniform(2.5, 5.0), 0, k + 1)[:-1]
        path = np.vstack([path[:1] + np.stack([np.zeros(k), rise], 1), path])
        rot = np.concatenate([np.repeat(rot[:1], k), rot])
        pre += k
    base_rot = np.deg2rad(rng.uniform(-15, 15))
    n = len(path)
    frames = np.zeros((n, 92), np.float32)
    for i in range(n):
        s = shape
        if shape_end is not None:
            a = np.clip((i - pre) / max(1, n - pre - post), 0, 1)
            s = shape * (1 - a) + shape_end * a
        c, si = np.cos(rot[i] + base_rot), np.sin(rot[i] + base_rot)
        pts = s @ np.array([[c, si], [-si, c]]) * hs
        pts = pts + [wx0 + path[i, 0] * hs, wy0 + path[i, 1] * hs]
        pts += rng.normal(0, 0.002, pts.shape)
        frames[i, 0:4] = [fx / aspect, fy, fw / aspect, fw * 1.1]
        frames[i, 4] = 1
        frames[i, 6:27] = pts[:, 0] / aspect
        frames[i, 27:48] = pts[:, 1]
    return frames, aspect


def synth_set(shapes, rng, n_j=300, n_z=300, n_other=900):
    """Returns list of dicts like train_signs videos: word, file, aspect, frames."""
    out = []
    pick = lambda L: shapes[L][rng.integers(len(shapes[L]))]
    for k in range(n_j):
        f, a = make_sequence(pick("i"), "j", rng)
        out.append({"word": "j", "file": f"synth_j_{k}", "aspect": a, "frames": f})
    for k in range(n_z):
        f, a = make_sequence(pick("d"), "z", rng)
        out.append({"word": "z", "file": f"synth_z_{k}", "aspect": a, "frames": f})
    others = [L for L in shapes if L not in ("j", "z")]
    for k in range(n_other):
        L = others[rng.integers(len(others))]
        end = None
        if rng.random() < 0.4:   # sliding from another letter into this one
            end = pick(L)
            start = pick(others[rng.integers(len(others))])
        else:
            start = pick(L)
        f, a = make_sequence(start, "other", rng, shape_end=end)
        out.append({"word": L, "file": f"synth_move_{k}", "aspect": a, "frames": f})
    return out
