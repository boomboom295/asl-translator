"""Shared feature code: turns 21 MediaPipe hand landmarks into a 42-number vector.

Steps (same as used by the training data):
  1. convert normalized landmarks to pixel coordinates
  2. make them relative to the wrist (landmark 0)
  3. scale so the largest absolute value is 1
This makes the features independent of where the hand is in the frame and how big it looks.
"""
import numpy as np

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def landmarks_to_features(xy_pixels):
    """xy_pixels: array of shape (21, 2) in pixel units. Returns shape (42,)."""
    pts = np.asarray(xy_pixels, dtype=np.float32).reshape(21, 2)
    pts = pts - pts[0]
    flat = pts.flatten()
    m = np.abs(flat).max()
    return flat / m if m > 0 else flat


def mirror(features):
    """Flip horizontally (left hand <-> right hand). Works on (..., 42) arrays."""
    f = np.array(features, dtype=np.float32, copy=True)
    f[..., 0::2] *= -1
    return f


class LetterClassifier:
    """Runs the trained network with plain numpy (no scikit-learn needed at run time)."""

    def __init__(self, path):
        d = np.load(path)
        n = int(d["n_layers"])
        self.weights = [d[f"W{i}"] for i in range(n)]
        self.biases = [d[f"b{i}"] for i in range(n)]
        self.classes_ = d["classes"]

    def predict_proba(self, X):
        a = np.atleast_2d(np.asarray(X, dtype=np.float32))
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            a = a @ W + b
            if i < len(self.weights) - 1:
                a = np.maximum(a, 0)  # ReLU
        a = np.exp(a - a.max(axis=1, keepdims=True))
        return a / a.sum(axis=1, keepdims=True)
