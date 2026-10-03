// Finding where a sign starts and ends in the live camera stream.

/** Speed of a hand in "hand sizes per second", from two raw frames (see sequence.js layout). */
function handSpeed(prev, cur, dt, aspect) {
  let best = 0;
  for (const o of [4, 48]) {
    if (!(prev[o] > 0.5 && cur[o] > 0.5)) continue;
    // hand size: wrist (0) to middle knuckle (9)
    const sx = (cur[o + 2 + 9] - cur[o + 2]) * aspect, sy = cur[o + 23 + 9] - cur[o + 23];
    const size = Math.max(Math.hypot(sx, sy), 1e-3);
    // fastest of wrist, index tip (8) and pinky tip (20)
    for (const i of [0, 8, 20]) {
      const dx = (cur[o + 2 + i] - prev[o + 2 + i]) * aspect, dy = cur[o + 23 + i] - prev[o + 23 + i];
      best = Math.max(best, Math.hypot(dx, dy) / size / dt);
    }
  }
  return best;
}

const present = (f) => f[4] > 0.5 || f[48] > 0.5;
const SPEED_WINDOW = 0.15; // seconds; measuring over a short window ignores tracking jitter

/** Frame from about SPEED_WINDOW seconds before the newest one in a {t, f} history. */
function earlier(history, t) {
  for (let i = history.length - 2; i >= 0; i--) if (t - history[i].t >= SPEED_WINDOW) return history[i];
  return history.length > 1 ? history[0] : null;
}

/**
 * Letters mode: notices a quick movement of the hand (J and Z are drawn in the air).
 * Call push() every frame; it returns {frames, start, end} when a movement has finished.
 */
export class MotionDetector {
  constructor({ startSpeed = 2.2, stopSpeed = 0.9, stopHold = 0.25, minDur = 0.25, maxDur = 2.5, minPath = 1.2 } = {}) {
    Object.assign(this, { startSpeed, stopSpeed, stopHold, minDur, maxDur, minPath });
    this.history = [];      // recent {t, f} for a little lead-in before the movement
    this.moving = false;
    this.seg = null;
  }

  get isMoving() { return this.moving; }

  push(t, f, aspect) {
    const last = this.history[this.history.length - 1];
    this.history.push({ t, f });
    while (this.history.length && t - this.history[0].t > 3) this.history.shift();
    const prev = earlier(this.history, t);
    if (!prev || !present(f) || !present(prev.f)) {
      const done = this.moving ? this.finish(t) : null;
      this.moving = false; this.seg = null;
      return done;
    }
    const v = handSpeed(prev.f, f, Math.max(t - prev.t, 1e-3), aspect);
    const dt = Math.max(t - last.t, 1e-3);
    if (!this.moving) {
      if (v > this.startSpeed) {
        this.moving = true;
        this.seg = { start: t, lastFast: t, path: v * dt };
      }
      return null;
    }
    this.seg.path += v * dt;
    if (v > this.stopSpeed) this.seg.lastFast = t;
    if (t - this.seg.lastFast >= this.stopHold || t - this.seg.start > this.maxDur) {
      const done = this.finish(t);
      this.moving = false; this.seg = null;
      return done;
    }
    return null;
  }

  finish(t) {
    const s = this.seg;
    if (!s) return null;
    const dur = s.lastFast - s.start;
    if (dur < this.minDur || dur > this.maxDur || s.path < this.minPath) return null;
    const from = s.start - 0.2, to = s.lastFast + 0.1;
    const frames = this.history.filter((h) => h.t >= from && h.t <= to && present(h.f)).map((h) => h.f);
    return frames.length >= 5 ? { frames, start: s.start, end: t } : null;
  }
}

/**
 * Words mode: a sign runs from when a hand shows up until the hands drop out of view
 * (or stay still for a moment after moving). push() returns {frames} when a sign has ended.
 */
export class SignSegmenter {
  constructor({ goneHold = 0.45, stillHold = 0.9, stillSpeed = 0.7, moveSpeed = 1.6, minDur = 0.35, maxDur = 5 } = {}) {
    Object.assign(this, { goneHold, stillHold, stillSpeed, moveSpeed, minDur, maxDur });
    this.reset();
  }

  reset() {
    this.frames = [];      // {t, f}
    this.active = false;
    this.lastSeen = -Infinity;
    this.lastMove = -Infinity;
    this.moved = false;
    this.waitForMove = false; // after a still-hands ending, wait for new movement
  }

  get isActive() { return this.active; }

  push(t, f, aspect) {
    const has = present(f);
    if (has) this.lastSeen = t;
    this.recent = (this.recent || []).filter((h) => t - h.t <= 1);
    this.recent.push({ t, f });
    const prev = earlier(this.recent, t);
    let v = 0;
    if (prev && has && present(prev.f)) v = handSpeed(prev.f, f, Math.max(t - prev.t, 1e-3), aspect);

    if (!this.active) {
      if (!has) { this.waitForMove = false; this.frames = []; return null; }
      if (this.waitForMove) {
        this.frames = [{ t, f }];
        if (v > this.moveSpeed) { this.waitForMove = false; this.begin(t); }
        return null;
      }
      this.begin(t);
    }
    this.frames.push({ t, f });
    if (v > this.stillSpeed) this.lastMove = t;
    if (v > this.moveSpeed) this.moved = true;

    const start = this.frames[0].t;
    if (t - this.lastSeen >= this.goneHold) return this.end(false);
    if (this.moved && has && t - this.lastMove >= this.stillHold) return this.end(true);
    if (t - start > this.maxDur) return this.end(true);
    return null;
  }

  begin(t) {
    this.active = true; this.moved = false; this.lastMove = t;
    const keep = this.frames.slice(-3);
    this.frames = keep;
  }

  end(stillHands) {
    const frames = this.frames.filter((h) => present(h.f));
    const dur = frames.length ? frames[frames.length - 1].t - frames[0].t : 0;
    this.active = false;
    this.frames = [];
    this.waitForMove = stillHands;
    this.moved = false;
    if (dur < this.minDur || frames.length < 5) return null;
    return { frames: frames.map((h) => h.f) };
  }
}
