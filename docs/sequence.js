// Moving-sign features and classifier (words, and the letters J and Z).
// Mirrors seqfeat.py exactly, so the website sees the same numbers the model was trained on.
//
// A raw frame is 92 numbers: face box (cx, cy, w, h; 0..1 of the frame, zeros = no face),
// then two hand slots of [present, handedness, 21 x, 21 y] (0..1 of the frame).

export const PER_FRAME = 92;
const HAND_FEATS = 47;
const FRAME_FEATS = 2 * HAND_FEATS;
const DEFAULT_FACE = [0.5, 0.3, 0.18];

/** Build a raw frame from MediaPipe results (unmirrored landmarks). */
export function rawFrame(handResult, faceResult, w, h) {
  const v = new Float32Array(PER_FRAME);
  const f = faceResult?.detections?.[0];
  if (f) {
    const b = f.boundingBox;
    v[0] = (b.originX + b.width / 2) / w; v[1] = (b.originY + b.height / 2) / h;
    v[2] = b.width / w; v[3] = b.height / h;
  }
  (handResult?.landmarks || []).slice(0, 2).forEach((lm, k) => {
    const o = 4 + k * 44;
    v[o] = 1;
    v[o + 1] = handResult.handedness?.[k]?.[0]?.categoryName === "Right" ? 1 : 0;
    for (let i = 0; i < 21; i++) { v[o + 2 + i] = lm[i].x; v[o + 23 + i] = lm[i].y; }
  });
  return v;
}

const anyHand = (f) => f[4] > 0.5 || f[48] > 0.5;

export function trim(frames) {
  let a = -1, b = -1;
  for (let i = 0; i < frames.length; i++) if (anyHand(frames[i])) { if (a < 0) a = i; b = i; }
  return a < 0 ? [] : frames.slice(a, b + 1);
}

function fillFaces(frames, aspect) {
  const n = frames.length;
  const idx = [];
  for (let i = 0; i < n; i++) if (frames[i][2] > 0) idx.push(i);
  return frames.map((_, i) => {
    if (!idx.length) return [DEFAULT_FACE[0] * aspect, DEFAULT_FACE[1], DEFAULT_FACE[2]];
    let best = idx[0];
    for (const j of idx) if (Math.abs(j - i) < Math.abs(best - i)) best = j;
    const f = frames[best];
    return [f[0] * aspect, f[1], Math.max(f[2] * aspect, 1e-3)];
  });
}

function resampleIdx(n, t) {
  if (n <= 1) return new Array(t).fill(0);
  // evenly spaced indices with exact integer rounding (same formula as seqfeat.py)
  return Array.from({ length: t }, (_, k) => Math.floor((2 * k * (n - 1) + (t - 1)) / (2 * (t - 1))));
}

function handBlock(xs, ys, face, out, off) {
  const [fx, fy, fw] = face;
  out[off] = 1;
  out[off + 1] = (xs[0] - fx) / fw; out[off + 2] = (ys[0] - fy) / fw;
  out[off + 3] = (xs[9] - fx) / fw; out[off + 4] = (ys[9] - fy) / fw;
  let m = 0;
  for (let i = 0; i < 21; i++) {
    const dx = xs[i] - xs[0], dy = ys[i] - ys[0];
    out[off + 5 + 2 * i] = dx; out[off + 6 + 2 * i] = dy;
    m = Math.max(m, Math.abs(dx), Math.abs(dy));
  }
  if (m > 0) for (let i = 0; i < 42; i++) out[off + 5 + i] /= m;
}

function frameFeatures(f, face, aspect, out, off) {
  const hands = [];
  for (const o of [4, 48]) {
    if (f[o] > 0.5) {
      const xs = new Float32Array(21), ys = new Float32Array(21);
      for (let i = 0; i < 21; i++) { xs[i] = f[o + 2 + i] * aspect; ys[i] = f[o + 23 + i]; }
      hands.push([xs, ys]);
    }
  }
  let slots;
  if (hands.length === 2) { hands.sort((a, b) => a[0][0] - b[0][0]); slots = hands; }
  else if (hands.length === 1) slots = hands[0][0][0] <= face[0] ? [hands[0], null] : [null, hands[0]];
  else slots = [null, null];
  slots.forEach((hnd, k) => { if (hnd) handBlock(hnd[0], hnd[1], face, out, off + k * HAND_FEATS); });
}

/** frames: array of raw frames; aspect = video width / height. Returns Float32Array or null. */
export function sequenceFeatures(frames, aspect, T) {
  const tr = trim(frames);
  if (!tr.length) return null;
  const faces = fillFaces(tr, aspect);
  const idx = resampleIdx(tr.length, T);
  const out = new Float32Array(T * FRAME_FEATS);
  idx.forEach((i, k) => frameFeatures(tr[i], faces[i], aspect, out, k * FRAME_FEATS));
  return out;
}

/** base64 little-endian float32 -> Float32Array */
function decodeF32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

/** A moving-sign network (exported by train_signs.py). */
export class SignClassifier {
  constructor(model) {
    this.T = model.T;
    this.classes = model.classes;
    this.labels = model.labels || {};
    this.mean = Float32Array.from(model.mean);
    this.std = Float32Array.from(model.std);
    this.layers = model.layers.map((L) => typeof L.W === "string"
      ? { rows: L.rows, cols: L.cols, W: decodeF32(L.W), b: decodeF32(L.b) }
      : { rows: L.W.length, cols: L.W[0].length, W: Float32Array.from(L.W.flat()), b: Float32Array.from(L.b) });
    this.isLetter = this.classes.map((c) => c.length === 1);
  }

  predictProba(features) {
    let a = new Float32Array(features.length);
    for (let i = 0; i < a.length; i++) a[i] = (features[i] - this.mean[i]) / this.std[i];
    this.layers.forEach((L, li) => {
      const z = Float32Array.from(L.b);
      for (let i = 0; i < L.rows; i++) {
        const ai = a[i];
        if (ai === 0) continue;
        const row = i * L.cols;
        for (let j = 0; j < L.cols; j++) z[j] += ai * L.W[row + j];
      }
      if (li < this.layers.length - 1) for (let j = 0; j < z.length; j++) z[j] = Math.max(z[j], 0);
      a = z;
    });
    let max = -Infinity;
    for (const v of a) max = Math.max(max, v);
    let sum = 0;
    for (let j = 0; j < a.length; j++) { a[j] = Math.exp(a[j] - max); sum += a[j]; }
    for (let j = 0; j < a.length; j++) a[j] /= sum;
    return a;
  }

  /** Top-k among words only (kind = "word") or letters only (kind = "letter"), renormalized. */
  rank(probs, kind, k = 3) {
    const want = kind === "letter";
    let total = 0;
    const items = [];
    probs.forEach((p, i) => { if (this.isLetter[i] === want) { items.push([p, i]); total += p; } });
    items.sort((a, b) => b[0] - a[0]);
    return items.slice(0, k).map(([p, i]) => ({
      id: this.classes[i], label: this.labels[this.classes[i]] || this.classes[i], p: p / total,
    }));
  }
}
