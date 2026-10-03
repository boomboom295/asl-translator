// ASL Translator, browser version.
// Webcam -> MediaPipe hands + face -> letter model (handshapes) and sign model (movements)
// -> text + speech. Everything runs locally in the browser; no video leaves the device.
import { Classifier, Smoother, Speller, HAND_CONNECTIONS, landmarksToFeatures, MIN_CONFIDENCE } from "./core.js?v=3";
import { SignClassifier, rawFrame, sequenceFeatures } from "./sequence.js?v=3";
import { MotionDetector, SignSegmenter } from "./segment.js?v=3";

const MP_VERSION = "0.10.14";
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const FACE_MODEL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
const RING_LENGTH = 175.93;     // circumference of the progress ring (2 * pi * 28)
const MOTION_LETTERS = new Set(["j", "z"]);
const MOTION_LETTER_MIN = 0.45; // how sure the sign model must be to type J or Z
const WORD_MIN = 0.3;           // below this, a word is only offered as a suggestion

const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const ui = {
  startBtn: $("startBtn"), startPanel: $("startPanel"), loadMsg: $("loadMsg"),
  hud: $("hud"), letter: $("currentLetter"), ring: $("ringFill"),
  status: $("statusLine"), conf: $("confLine"), top3: $("top3"), guessNote: $("guessNote"),
  transcript: $("transcript"), autoSpeak: $("autoSpeak"), vocab: $("vocab"), vocabCount: $("vocabCount"),
  modeLetters: $("modeLetters"), modeWords: $("modeWords"), modeHint: $("modeHint"), tipsLetters: $("tipsLetters"), tipsWords: $("tipsWords"),
};

let hands = null, face = null, letterModel = null, wordModel = null, motionModel = null;
let mode = "letters";
const smoother = new Smoother();
const speller = new Speller((word) => { if (ui.autoSpeak.checked) speak(word); }, now());
const motion = new MotionDetector();
const segmenter = new SignSegmenter();
let lastVideoTime = -1;
let lastRendered = null;
let wordGuesses = null;     // last word result: {items, added}
let flashUntil = 0, flashText = "";

function now() { return performance.now() / 1000; }

/* ---------- speech ---------- */
function speak(text) {
  text = text.trim();
  if (!text || !("speechSynthesis" in window)) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text.toLowerCase());
  u.lang = "en-US";
  speechSynthesis.speak(u);
}

/* ---------- loading ---------- */
function setLoad(msg, isError = false) {
  ui.loadMsg.textContent = msg;
  ui.loadMsg.classList.toggle("error", isError);
}

async function loadJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Could not load ${path}`);
  return res.json();
}

async function loadModels() {
  setLoad("Loading the sign models…");
  const [letters, words, motions] = await Promise.all([
    loadJSON("models/asl_classifier.json?v=3"), loadJSON("models/word_classifier.json?v=3"), loadJSON("models/motion_classifier.json?v=3"),
  ]);
  letterModel = new Classifier(letters);
  wordModel = new SignClassifier(words);
  motionModel = new SignClassifier(motions);
  renderVocab();

  setLoad("Loading hand tracking…");
  let mp;
  try { mp = await import(`${MP_BASE}/vision_bundle.mjs`); } catch { mp = await import(MP_BASE); }
  const fileset = await mp.FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
  const handOpts = (delegate) => ({
    baseOptions: { modelAssetPath: "models/hand_landmarker.task", delegate },
    runningMode: "VIDEO", numHands: 2,
    minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  try { hands = await mp.HandLandmarker.createFromOptions(fileset, handOpts("GPU")); }
  catch { hands = await mp.HandLandmarker.createFromOptions(fileset, handOpts("CPU")); }
  try {
    face = await mp.FaceDetector.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: FACE_MODEL }, runningMode: "VIDEO", minDetectionConfidence: 0.4,
    });
  } catch (e) {
    console.warn("Face detector unavailable; word positions will be less accurate.", e);
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("This browser can't use the camera here. Open the site over https in Chrome, Edge, Firefox or Safari.");
  }
  setLoad("Waiting for camera permission…");
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 960 } }, audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

async function start() {
  ui.startBtn.disabled = true;
  try {
    await loadModels();
    await startCamera();
    ui.startPanel.hidden = true;
    ui.hud.hidden = false;
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    let msg = err?.message || String(err);
    if (err?.name === "NotAllowedError") msg = "Camera access was blocked. Allow the camera in your browser's address bar, then try again.";
    else if (err?.name === "NotFoundError") msg = "No camera was found. Plug one in and try again.";
    else if (!hands && wordModel) msg = "Couldn't load hand tracking. Check your internet connection and try again.";
    setLoad(msg, true);
    ui.startBtn.disabled = false;
    ui.startBtn.textContent = "Try again";
  }
}

/* ---------- main loop ---------- */
function loop() {
  requestAnimationFrame(loop);
  if (video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  const w = video.videoWidth, h = video.videoHeight, aspect = w / h;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  ctx.clearRect(0, 0, w, h);

  const ts = performance.now();
  const hr = hands.detectForVideo(video, ts);
  const fr = face ? face.detectForVideo(video, ts) : null;
  const raw = rawFrame(hr, fr, w, h);
  const t = now();
  (hr.landmarks || []).forEach((lm) => drawHand(lm, w, h));

  if (mode === "letters") lettersStep(hr, raw, t, w, h, aspect);
  else wordsStep(raw, t, aspect);
  renderTranscript();
}

let twoHandsSince = null;
function updateModeHint(nHands, t) {
  if (mode !== "letters") { ui.modeHint.hidden = true; twoHandsSince = null; return; }
  if (nHands >= 2) { if (twoHandsSince === null) twoHandsSince = t; }
  else if (twoHandsSince !== null && t - twoHandsSince < 0.8) twoHandsSince = null;
  if (twoHandsSince !== null && t - twoHandsSince > 0.8) ui.modeHint.hidden = false;
}

function lettersStep(hr, raw, t, w, h, aspect) {
  updateModeHint((hr.landmarks || []).length, t);
  // 1) movement letters (J, Z)
  const moved = motion.push(t, raw, aspect);
  if (moved) {
    const x = sequenceFeatures(moved.frames, aspect, motionModel.T);
    if (x) {
      const best = motionModel.rank(motionModel.predictProba(x), "letter", 3)[0];
      if (MOTION_LETTERS.has(best.id) && best.p >= MOTION_LETTER_MIN) {
        speller.typeMotionLetter(best.id.toUpperCase(), t, moved.start);
        flash(`Typed ${best.id.toUpperCase()}`);
      }
    }
  }

  // 2) handshape letters (everything else), using the first hand the tracker found
  let letter = null, conf = 0, probs = null;
  if (hr.landmarks && hr.landmarks.length) {
    // The Python app mirrors the frame before tracking, so mirror x here to match its training setup.
    const pts = hr.landmarks[0].map((p) => [(1 - p.x) * w, p.y * h]);
    probs = smoother.push(letterModel.predictProba(landmarksToFeatures(pts)));
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    letter = letterModel.classes[best];
    conf = probs[best];
  } else {
    smoother.clear();
  }
  // while the hand is moving (drawing J or Z) don't type the handshape
  const progress = motion.isMoving ? speller.update(letter, 0, t) : speller.update(letter, conf, t);
  renderLetterHud(letter, conf, progress, motion.isMoving);
  renderLetterTop3(probs);
}

function wordsStep(raw, t, aspect) {
  const sign = segmenter.push(t, raw, aspect);
  if (sign) {
    const x = sequenceFeatures(sign.frames, aspect, wordModel.T);
    if (x) {
      const items = wordModel.rank(wordModel.predictProba(x), "word", 3);
      const added = items[0].p >= WORD_MIN;
      if (added) {
        speller.addWord(items[0].label);
        if (ui.autoSpeak.checked) speak(items[0].label);
      }
      wordGuesses = { items, added, chosen: added ? 0 : -1 };
      renderWordGuesses();
    }
  }
  renderWordHud(t);
}

function flash(text) { flashText = text; flashUntil = now() + 1.2; }

function drawHand(lm, w, h) {
  ctx.lineWidth = Math.max(2, w / 320);
  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.beginPath();
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.moveTo(lm[a].x * w, lm[a].y * h);
    ctx.lineTo(lm[b].x * w, lm[b].y * h);
  }
  ctx.stroke();
  ctx.fillStyle = "#ff8c1a";
  const r = Math.max(3, w / 220);
  for (const p of lm) { ctx.beginPath(); ctx.arc(p.x * w, p.y * h, r, 0, Math.PI * 2); ctx.fill(); }
}

/* ---------- rendering ---------- */
function setRing(progress) { ui.ring.style.strokeDashoffset = String(RING_LENGTH * (1 - progress)); }

function renderLetterHud(letter, conf, progress, moving) {
  ui.letter.textContent = letter ?? "–";
  setRing(progress);
  if (now() < flashUntil) { ui.status.textContent = flashText; ui.conf.textContent = ""; return; }
  if (moving) { ui.status.textContent = "Watching the movement…"; ui.conf.textContent = "J and Z are drawn in the air"; }
  else if (!letter) { ui.status.textContent = "Show your hand"; ui.conf.textContent = ""; }
  else if (conf < MIN_CONFIDENCE) { ui.status.textContent = "Not sure yet…"; ui.conf.textContent = `${Math.round(conf * 100)}% ${letter}`; }
  else { ui.status.textContent = progress >= 1 ? `Typed ${letter}` : `Hold for ${letter}`; ui.conf.textContent = `${Math.round(conf * 100)}% sure`; }
}

function renderWordHud(t) {
  setRing(0);
  ui.letter.textContent = segmenter.isActive ? "●" : "–";
  if (segmenter.isActive) { ui.status.textContent = "Watching your sign…"; ui.conf.textContent = "Lower your hands when done"; }
  else if (wordGuesses) {
    const { items, chosen, picked } = wordGuesses;
    const g = items[Math.max(chosen, 0)];
    ui.status.textContent = chosen < 0 ? "Not sure — pick below" : picked ? `You picked “${g.label}”` : `Got “${g.label}”`;
    ui.conf.textContent = picked ? "" : `${Math.round(items[0].p * 100)}% sure`;
  } else { ui.status.textContent = "Sign a word"; ui.conf.textContent = "Raise your hands to start"; }
}

let lastTop3Key = "";
function renderLetterTop3(probs) {
  if (!probs) { if (lastTop3Key) { ui.top3.innerHTML = ""; lastTop3Key = ""; } return; }
  const ranked = [...probs].map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  ui.top3.innerHTML = ranked.map(([p, i]) =>
    `<div class="pred"><b>${letterModel.classes[i]}</b><div class="bar"><i style="width:${(p * 100).toFixed(1)}%"></i></div><span>${Math.round(p * 100)}%</span></div>`
  ).join("");
  lastTop3Key = "x";
}

function renderWordGuesses() {
  if (!wordGuesses) { ui.top3.innerHTML = ""; ui.guessNote.hidden = true; return; }
  const { items, added, chosen } = wordGuesses;
  ui.top3.innerHTML = items.map((g, k) =>
    `<button class="pred guess${k === wordGuesses.chosen ? " chosen" : ""}" data-k="${k}"><b>${g.label}</b><div class="bar"><i style="width:${(g.p * 100).toFixed(1)}%"></i></div><span>${Math.round(g.p * 100)}%</span></button>`
  ).join("");
  ui.guessNote.hidden = false;
  ui.guessNote.textContent = chosen >= 0 ? "Wrong word? Tap the right one to swap it." : "Not sure enough to type it. Tap the word you signed.";
}

ui.top3.addEventListener("click", (e) => {
  const b = e.target.closest("button.guess");
  if (!b || !wordGuesses) return;
  const k = Number(b.dataset.k);
  const g = wordGuesses.items[k];
  if (k === wordGuesses.chosen) return;
  if (wordGuesses.chosen >= 0) speller.replaceLastWord(g.label); else speller.addWord(g.label);
  wordGuesses = { ...wordGuesses, chosen: k, picked: true };
  if (ui.autoSpeak.checked) speak(g.label);
  renderWordGuesses();
  renderTranscript();
});

function renderTranscript() {
  if (speller.text === lastRendered) return;
  lastRendered = speller.text;
  if (!speller.text.trim()) {
    ui.transcript.innerHTML = '<span class="placeholder">What you sign appears here.</span>';
    return;
  }
  ui.transcript.textContent = speller.text;
  const caret = document.createElement("span");
  caret.className = "caret";
  ui.transcript.appendChild(caret);
  ui.transcript.scrollTop = ui.transcript.scrollHeight;
}

function renderVocab() {
  const words = wordModel.classes.map((c) => wordModel.labels[c] || c).sort();
  ui.vocabCount.textContent = String(words.length);
  ui.vocab.innerHTML = words.map((w) => `<li>${w}</li>`).join("");
}

function setMode(m) {
  mode = m;
  ui.modeLetters.setAttribute("aria-pressed", String(m === "letters"));
  ui.modeWords.setAttribute("aria-pressed", String(m === "words"));
  ui.tipsLetters.hidden = m !== "letters";
  ui.tipsWords.hidden = m !== "words";
  ui.modeHint.hidden = true; twoHandsSince = null;
  try { localStorage.setItem("asl-mode", m); } catch {}
  segmenter.reset();
  smoother.clear();
  wordGuesses = null;
  ui.top3.innerHTML = ""; lastTop3Key = "";
  ui.guessNote.hidden = true;
  if (speller.text && !speller.text.endsWith(" ")) speller.endWord();
}

/* ---------- controls ---------- */
const actions = {
  space: () => speller.endWord(),
  back: () => speller.backspace(),
  clear: () => { speller.clear(); wordGuesses = null; renderWordGuesses(); },
  speak: () => speak(speller.text),
  copy: async () => {
    try { await navigator.clipboard.writeText(speller.text.trim()); flashBtn($("copyBtn"), "Copied"); }
    catch { flashBtn($("copyBtn"), "Couldn't copy"); }
  },
};

function flashBtn(btn, label) {
  const old = btn.textContent;
  btn.textContent = label;
  setTimeout(() => { btn.textContent = old; }, 1200);
}

function act(name) { actions[name](); renderTranscript(); }

ui.startBtn.addEventListener("click", start);
try { if (localStorage.getItem("asl-mode") === "words") setMode("words"); } catch {}
ui.modeLetters.addEventListener("click", () => setMode("letters"));
ui.modeWords.addEventListener("click", () => setMode("words"));
$("hintSwitch").addEventListener("click", () => setMode("words"));
$("spaceBtn").addEventListener("click", () => act("space"));
$("backBtn").addEventListener("click", () => act("back"));
$("clearBtn").addEventListener("click", () => act("clear"));
$("speakBtn").addEventListener("click", () => act("speak"));
$("copyBtn").addEventListener("click", () => act("copy"));

document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, textarea, [contenteditable]") || e.ctrlKey || e.metaKey || e.altKey) return;
  const map = { " ": "space", Backspace: "back", c: "clear", C: "clear", s: "speak", S: "speak" };
  if (e.key === "w" || e.key === "W") { setMode(mode === "letters" ? "words" : "letters"); return; }
  const name = map[e.key];
  if (!name) return;
  if (e.target.closest("button") && e.key === " ") return; // let Space press focused buttons
  e.preventDefault();
  act(name);
});
