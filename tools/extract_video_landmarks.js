// Landmark extractor that runs in a browser tab (same MediaPipe JS as the website,
// so training features match what the site sees at run time).
// Usage: open the live site, paste this into the console, then:
//   await ASLX.init(); await ASLX.run(jobs)   // jobs: [{word, url, file}]
//   ASLX.save("asl_video_landmarks.json")
(() => {
  const MP = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
  const FACE_MODEL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
  const FPS = 15, WORKERS = 3, MAX_SECONDS = 6;
  const S = { hands: null, face: null, results: [], errors: [], done: 0, total: 0, started: 0, running: false };

  // Per frame: face cx, cy, w, h (0..1, 0 = none), then 2 hand slots of
  // [present, isRightLabel, 21 x, 21 y] -> 4 + 2*44 = 92 values, stored as int16 * 1e4.
  const PER_FRAME = 92;

  async function init() {
    const mp = await import(`${MP}/vision_bundle.mjs`);
    const fs = await mp.FilesetResolver.forVisionTasks(`${MP}/wasm`);
    const mk = (d) => mp.HandLandmarker.createFromOptions(fs, {
      baseOptions: { modelAssetPath: "models/hand_landmarker.task", delegate: d },
      runningMode: "IMAGE", numHands: 2, minHandDetectionConfidence: 0.4, minHandPresenceConfidence: 0.4,
    });
    try { S.hands = await mk("GPU"); } catch { S.hands = await mk("CPU"); }
    S.face = await mp.FaceDetector.createFromOptions(fs, {
      baseOptions: { modelAssetPath: FACE_MODEL }, runningMode: "IMAGE", minDetectionConfidence: 0.4,
    });
    return "ready";
  }

  function frameVector(hr, fr, w, h) {
    const v = new Float32Array(PER_FRAME);
    const f = fr.detections && fr.detections[0];
    if (f) {
      const b = f.boundingBox;
      v[0] = (b.originX + b.width / 2) / w; v[1] = (b.originY + b.height / 2) / h;
      v[2] = b.width / w; v[3] = b.height / h;
    }
    (hr.landmarks || []).slice(0, 2).forEach((lm, k) => {
      const o = 4 + k * 44;
      v[o] = 1;
      const lab = hr.handedness?.[k]?.[0]?.categoryName;
      v[o + 1] = lab === "Right" ? 1 : 0;
      for (let i = 0; i < 21; i++) { v[o + 2 + i] = lm[i].x; v[o + 23 + i] = lm[i].y; }
    });
    return v;
  }

  function toB64(f32) {
    const q = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i++) q[i] = Math.max(-32767, Math.min(32767, Math.round(f32[i] * 1e4)));
    const bytes = new Uint8Array(q.buffer);
    let s = "";
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(s);
  }

  const seek = (video, t) => new Promise((res, rej) => {
    const ok = () => { video.removeEventListener("seeked", ok); res(); };
    video.addEventListener("seeked", ok);
    video.currentTime = t;
    setTimeout(() => rej(new Error("seek timeout")), 8000);
  });

  async function processOne(job, video) {
    const blob = await (await fetch(job.url)).blob();
    const url = URL.createObjectURL(blob);
    try {
      video.src = url;
      await new Promise((res, rej) => { video.onloadeddata = res; video.onerror = () => rej(new Error("decode")); });
      const w = video.videoWidth, h = video.videoHeight, dur = Math.min(video.duration || 0, MAX_SECONDS);
      const n = Math.max(1, Math.floor(dur * FPS));
      const all = new Float32Array(n * PER_FRAME);
      for (let i = 0; i < n; i++) {
        await seek(video, Math.min(i / FPS, Math.max(0, video.duration - 0.01)));
        const hr = S.hands.detect(video);
        const fr = S.face.detect(video);
        all.set(frameVector(hr, fr, w, h), i * PER_FRAME);
      }
      S.results.push({ word: job.word, file: job.file, w, h, fps: FPS, n, data: toB64(all) });
    } finally { URL.revokeObjectURL(url); }
  }

  async function run(jobs) {
    S.total += jobs.length; S.running = true; S.started = S.started || performance.now();
    const queue = jobs.slice();
    const worker = async () => {
      const video = document.createElement("video");
      video.muted = true; video.playsInline = true; video.preload = "auto";
      while (queue.length) {
        const job = queue.shift();
        try { await processOne(job, video); } catch (e) { S.errors.push({ file: job.file, err: String(e) }); }
        S.done++;
      }
    };
    await Promise.all(Array.from({ length: WORKERS }, worker));
    S.running = false;
    return status();
  }

  function status() {
    const secs = (performance.now() - S.started) / 1000;
    return { done: S.done, total: S.total, ok: S.results.length, errors: S.errors.length, running: S.running,
             secs: Math.round(secs), etaMin: S.done ? Math.round((S.total - S.done) * secs / S.done / 60) : null };
  }

  function save(name) {
    const blob = new Blob([JSON.stringify({ format: "aslx-v1", perFrame: PER_FRAME, scale: 1e4, videos: S.results })],
                          { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click();
    return blob.size;
  }

  window.ASLX = { init, run, status, save, state: S };
})();
