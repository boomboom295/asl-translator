// ASL Translator, browser version.
// Webcam -> MediaPipe hand landmarks -> letter classifier -> text + speech.
// Everything runs locally in the browser; no video leaves the device.
import { Classifier, Smoother, Speller, HAND_CONNECTIONS, landmarksToFeatures } from "./core.js";

const MP_VERSION = "0.10.14";
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const RING_LENGTH = 175.93; // circumference of the progress ring (2 * pi * 28)

const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const ui = {
  startBtn: $("startBtn"), startPanel: $("startPanel"), loadMsg: $("loadMsg"),
  hud: $("hud"), letter: $("currentLetter"), ring: $("ringFill"),
  status: $("statusLine"), conf: $("confLine"), top3: $("top3"),
  transcript: $("transcript"), autoSpeak: $("autoSpeak"),
};

let landmarker = null;
let classifier = null;
const smoother = new Smoother();
const speller = new Speller((word) => { if (ui.autoSpeak.checked) speak(word); }, now());
let lastVideoTime = -1;
let lastRendered = null;

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

async function loadModels() {
  setLoad("Loading the letter model…");
  const res = await fetch("models/asl_classifier.json");
  if (!res.ok) throw new Error("Could not load models/asl_classifier.json");
  classifier = new Classifier(await res.json());

  setLoad("Loading hand tracking…");
  let mp;
  try {
    mp = await import(`${MP_BASE}/vision_bundle.mjs`);
  } catch {
    mp = await import(MP_BASE);
  }
  const fileset = await mp.FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: "models/hand_landmarker.task", delegate },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    landmarker = await mp.HandLandmarker.createFromOptions(fileset, options("GPU"));
  } catch {
    landmarker = await mp.HandLandmarker.createFromOptions(fileset, options("CPU"));
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("This browser can't use the camera here. Open the site over https in Chrome, Edge, Firefox or Safari.");
  }
  setLoad("Waiting for camera permission…");
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 960 } },
    audio: false,
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
    else if (!landmarker && classifier) msg = "Couldn't load hand tracking. Check your internet connection and try again.";
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

  const w = video.videoWidth, h = video.videoHeight;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  ctx.clearRect(0, 0, w, h);

  const result = landmarker.detectForVideo(video, performance.now());
  let letter = null, conf = 0, probs = null;

  if (result.landmarks && result.landmarks.length) {
    const lm = result.landmarks[0];
    drawHand(lm, w, h);
    // The Python app mirrors the frame before tracking, so mirror x here to match its training setup.
    const pts = lm.map((p) => [(1 - p.x) * w, p.y * h]);
    probs = smoother.push(classifier.predictProba(landmarksToFeatures(pts)));
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    letter = classifier.classes[best];
    conf = probs[best];
  } else {
    smoother.clear();
  }

  const progress = speller.update(letter, conf, now());
  renderHud(letter, conf, progress);
  renderTop3(probs);
  renderTranscript();
}

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
  for (const p of lm) {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, r, 0, Math.PI * 2);
    ctx.fill();
  }
}

/* ---------- rendering ---------- */
function renderHud(letter, conf, progress) {
  ui.letter.textContent = letter ?? "–";
  ui.ring.style.strokeDashoffset = String(RING_LENGTH * (1 - progress));
  if (!letter) {
    ui.status.textContent = "Show your hand";
    ui.conf.textContent = "";
  } else if (conf < 0.6) {
    ui.status.textContent = "Not sure yet…";
    ui.conf.textContent = `${Math.round(conf * 100)}% ${letter}`;
  } else {
    ui.status.textContent = progress >= 1 ? `Typed ${letter}` : `Hold for ${letter}`;
    ui.conf.textContent = `${Math.round(conf * 100)}% sure`;
  }
}

function renderTop3(probs) {
  if (!probs) { ui.top3.innerHTML = ""; return; }
  const ranked = [...probs].map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  ui.top3.innerHTML = ranked.map(([p, i]) =>
    `<div class="pred"><b>${classifier.classes[i]}</b><div class="bar"><i style="width:${(p * 100).toFixed(1)}%"></i></div><span>${Math.round(p * 100)}%</span></div>`
  ).join("");
}

function renderTranscript() {
  if (speller.text === lastRendered) return;
  lastRendered = speller.text;
  if (!speller.text) {
    ui.transcript.innerHTML = '<span class="placeholder">Letters you sign appear here.</span>';
    return;
  }
  ui.transcript.textContent = speller.text;
  const caret = document.createElement("span");
  caret.className = "caret";
  ui.transcript.appendChild(caret);
  ui.transcript.scrollTop = ui.transcript.scrollHeight;
}

/* ---------- controls ---------- */
const actions = {
  space: () => speller.endWord(),
  back: () => speller.backspace(),
  clear: () => speller.clear(),
  speak: () => speak(speller.text),
  copy: async () => {
    try {
      await navigator.clipboard.writeText(speller.text.trim());
      flash($("copyBtn"), "Copied");
    } catch { flash($("copyBtn"), "Couldn't copy"); }
  },
};

function flash(btn, label) {
  const old = btn.textContent;
  btn.textContent = label;
  setTimeout(() => { btn.textContent = old; }, 1200);
}

function act(name) { actions[name](); renderTranscript(); }

ui.startBtn.addEventListener("click", start);
$("spaceBtn").addEventListener("click", () => act("space"));
$("backBtn").addEventListener("click", () => act("back"));
$("clearBtn").addEventListener("click", () => act("clear"));
$("speakBtn").addEventListener("click", () => act("speak"));
$("copyBtn").addEventListener("click", () => act("copy"));

document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, textarea, [contenteditable]") || e.ctrlKey || e.metaKey || e.altKey) return;
  const map = { " ": "space", Backspace: "back", c: "clear", C: "clear", s: "speak", S: "speak" };
  const name = map[e.key];
  if (!name) return;
  if (e.target.closest("button") && e.key === " ") return; // let Space press focused buttons
  e.preventDefault();
  act(name);
});
