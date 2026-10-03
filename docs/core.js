// Shared logic for the browser version: features, classifier and speller.
// It mirrors features.py and the Speller class in app.py, so the website
// and the Python app give the same results.

export const HOLD_SECONDS = 0.7;      // how long a letter must be held to be typed
export const MIN_CONFIDENCE = 0.6;    // ignore predictions below this probability
export const WORD_GAP_SECONDS = 1.5;  // no hand for this long -> end of word
export const SMOOTHING_FRAMES = 5;    // average probabilities over this many frames

export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10],
  [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17],
  [17, 18], [18, 19], [19, 20],
];

/** points: 21 [x, y] pairs in pixel units. Returns a Float32Array of 42 features. */
export function landmarksToFeatures(points) {
  const out = new Float32Array(42);
  const [x0, y0] = points[0];
  let m = 0;
  for (let i = 0; i < 21; i++) {
    const dx = points[i][0] - x0;
    const dy = points[i][1] - y0;
    out[2 * i] = dx;
    out[2 * i + 1] = dy;
    m = Math.max(m, Math.abs(dx), Math.abs(dy));
  }
  if (m > 0) for (let i = 0; i < 42; i++) out[i] /= m;
  return out;
}

/** The trained network (exported by export_web_model.py), run in plain JavaScript. */
export class Classifier {
  constructor(model) {
    this.classes = model.classes;
    this.layers = model.layers.map(({ W, b }) => ({
      rows: W.length,
      cols: W[0].length,
      W: Float32Array.from(W.flat()),
      b: Float32Array.from(b),
    }));
  }

  predictProba(features) {
    let a = Float32Array.from(features);
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
}

/** Averages the last few probability vectors to reduce flicker. */
export class Smoother {
  constructor(n = SMOOTHING_FRAMES) { this.n = n; this.items = []; }
  push(p) {
    this.items.push(p);
    if (this.items.length > this.n) this.items.shift();
    const avg = new Float32Array(p.length);
    for (const it of this.items) for (let j = 0; j < it.length; j++) avg[j] += it[j] / this.items.length;
    return avg;
  }
  clear() { this.items = []; }
}

/** Turns a noisy stream of per-frame predictions into typed letters and words. Times are in seconds. */
export class Speller {
  constructor(onWord = () => {}, now = 0) {
    this.text = "";
    this.onWord = onWord;
    this.candidate = null;
    this.since = 0;
    this.lastTyped = null;
    this.lastHandTime = now;
  }

  /** letter: string or null (no hand). Returns hold progress 0..1. */
  update(letter, conf, now) {
    if (letter === null) {
      this.candidate = null;
      this.lastTyped = null;
      if (this.text && !this.text.endsWith(" ") && now - this.lastHandTime > WORD_GAP_SECONDS) this.endWord();
      return 0;
    }
    this.lastHandTime = now;
    if (conf < MIN_CONFIDENCE) { this.candidate = null; return 0; }
    if (letter !== this.candidate) { this.candidate = letter; this.since = now; }
    const held = now - this.since;
    if (letter !== this.lastTyped && held >= 0.3) this.lastTyped = null; // a new shape allows repeats again
    if (held >= HOLD_SECONDS && letter !== this.lastTyped) {
      this.text += letter;
      this.lastTyped = letter;
    }
    return Math.min(held / HOLD_SECONDS, 1);
  }

  endWord() {
    const words = this.text.trim().split(/\s+/).filter(Boolean);
    if (words.length) this.onWord(words[words.length - 1]);
    this.text = this.text.trimEnd() + " ";
  }

  backspace() { this.text = this.text.slice(0, -1); }
  clear() { this.text = ""; }
}
