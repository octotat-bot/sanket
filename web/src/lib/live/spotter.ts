// Sign spotting (CLAUDE.md Phase 5, ADR 009): decide when a sign starts and ends in the live
// stream, so the model classifies one sign-sized clip instead of an endless stream.
//
//   idle ──(a hand is raised AND moving for startFrames)──▶ signing
//   signing ──(hands lowered for endFrames, OR still for stillMs, OR maxMs reached)──▶ cooldown
//   cooldown ──(cooldownMs)──▶ idle
//
// When signing ends, the frames from `prerollMs` before the start to the end form one clip
// (too-short ones, < minMs, are dropped). Units are body-relative: shoulder widths (sw) from the
// shoulder midpoint, x made aspect-correct with the processed frame size (ADR 007), so the
// thresholds don't depend on camera distance or resolution. Pure and deterministic: unit-tested
// with synthetic motion in tests/unit/spotter.test.ts.
import type { StoredFrame } from "../mediapipe/holistic";
import type { Clip } from "../inference/features";

export interface SpotterConfig {
  /** A hand counts as raised when its wrist is less than this many sw below the shoulder line. */
  raisedBelowShoulderSw: number;
  /** Smoothed hand speed (sw/s) needed to start, and below which the hands count as still. */
  startSpeed: number;
  stillSpeed: number;
  startFrames: number;
  endFrames: number;
  stillMs: number;
  minMs: number;
  maxMs: number;
  prerollMs: number;
  cooldownMs: number;
  /** Exponential smoothing of the speed (0..1, higher = more responsive). */
  speedAlpha: number;
  visMin: number;
}

export const DEFAULT_SPOTTER: SpotterConfig = {
  raisedBelowShoulderSw: 1.6,
  startSpeed: 0.8,
  stillSpeed: 0.25,
  startFrames: 3,
  endFrames: 6,
  stillMs: 600,
  minMs: 500,
  maxMs: 4000,
  prerollMs: 250,
  cooldownMs: 400,
  speedAlpha: 0.35,
  visMin: 0.5,
};

export type SpotterState = "idle" | "signing" | "cooldown";

export interface SpotterUpdate {
  state: SpotterState;
  raised: boolean;
  speed: number; // smoothed, sw/s
  /** Set on the frame where a sign ends: the clip to classify. */
  clip?: Clip;
  /** Set when a sign ends: why (for the debug panel and tests). */
  endReason?: "lowered" | "still" | "max" | "too_short";
}

interface Buffered {
  t: number;
  f: StoredFrame;
}

const L_SH = 11;
const R_SH = 12;
const WRIST = 0;

export class Spotter {
  private state: SpotterState = "idle";
  private buf: Buffered[] = [];
  private speed = 0;
  private prev: { t: number; pts: [number, number][] } | null = null;
  private hot = 0; // consecutive frames raised + moving (idle)
  private lowered = 0; // consecutive frames with no raised hand (signing)
  private stillSince: number | null = null;
  private start = 0;
  private cooldownUntil = 0;

  /** `frameSize`: the (width, height) of the image MediaPipe processed (480 rows, ADR 007). */
  constructor(
    private readonly frameSize: () => [number, number],
    private readonly cfg: SpotterConfig = DEFAULT_SPOTTER,
  ) {}

  reset(): void {
    this.state = "idle";
    this.buf = [];
    this.speed = 0;
    this.prev = null;
    this.hot = 0;
    this.lowered = 0;
    this.stillSince = null;
  }

  /** Body frame: shoulder midpoint origin, shoulder-width units, aspect-correct x. */
  private body(f: StoredFrame): ((p: number[]) => [number, number]) | null {
    if (!f.pose_present) return null;
    const l = f.pose[L_SH];
    const r = f.pose[R_SH];
    if (l[3] < this.cfg.visMin || r[3] < this.cfg.visMin) return null;
    const [w, h] = this.frameSize();
    const ax = w / h;
    const cx = ((l[0] + r[0]) / 2) * ax;
    const cy = (l[1] + r[1]) / 2;
    const sw = Math.hypot((l[0] - r[0]) * ax, l[1] - r[1]);
    if (!(sw > 1e-6)) return null;
    return (p) => [(p[0] * ax - cx) / sw, (p[1] - cy) / sw];
  }

  update(tMs: number, f: StoredFrame): SpotterUpdate {
    const cfg = this.cfg;
    this.buf.push({ t: tMs, f });
    const keepFrom = (this.state === "signing" ? this.start : tMs) - cfg.prerollMs - 50;
    while (this.buf.length && this.buf[0].t < keepFrom) this.buf.shift();

    // Hand points (body coords) and whether a hand is raised.
    const toBody = this.body(f);
    const pts: [number, number][] = [];
    let raised = false;
    if (toBody) {
      for (const [present, hand] of [
        [f.left_present, f.left],
        [f.right_present, f.right],
      ] as const) {
        if (!present) continue;
        for (const p of hand) pts.push(toBody(p));
        if (toBody(hand[WRIST])[1] < cfg.raisedBelowShoulderSw) raised = true;
      }
    }
    // Speed: mean displacement of matching hand points per second, smoothed.
    let inst = 0;
    if (this.prev && pts.length && pts.length === this.prev.pts.length && tMs > this.prev.t) {
      let s = 0;
      for (let i = 0; i < pts.length; i++) s += Math.hypot(pts[i][0] - this.prev.pts[i][0], pts[i][1] - this.prev.pts[i][1]);
      inst = s / pts.length / ((tMs - this.prev.t) / 1000);
    }
    this.prev = pts.length ? { t: tMs, pts } : null;
    this.speed = cfg.speedAlpha * inst + (1 - cfg.speedAlpha) * this.speed;

    const out: SpotterUpdate = { state: this.state, raised, speed: this.speed };
    if (this.state === "cooldown") {
      if (tMs >= this.cooldownUntil) this.state = "idle";
    } else if (this.state === "idle") {
      this.hot = raised && this.speed >= cfg.startSpeed ? this.hot + 1 : 0;
      if (this.hot >= cfg.startFrames) {
        this.state = "signing";
        // The sign began a little before motion crossed the threshold.
        this.start = this.buf[Math.max(0, this.buf.length - cfg.startFrames)].t;
        this.lowered = 0;
        this.stillSince = null;
      }
    } else {
      this.lowered = raised ? 0 : this.lowered + 1;
      this.stillSince = this.speed < cfg.stillSpeed ? (this.stillSince ?? tMs) : null;
      let reason: SpotterUpdate["endReason"];
      if (this.lowered >= cfg.endFrames) reason = "lowered";
      else if (this.stillSince !== null && tMs - this.stillSince >= cfg.stillMs) reason = "still";
      else if (tMs - this.start >= cfg.maxMs) reason = "max";
      if (reason) {
        const from = this.start - cfg.prerollMs;
        const frames = this.buf.filter((b) => b.t >= from);
        this.state = "cooldown";
        this.cooldownUntil = tMs + cfg.cooldownMs;
        this.hot = 0;
        if (tMs - this.start < cfg.minMs || frames.length < 2) {
          out.endReason = "too_short";
        } else {
          out.endReason = reason;
          out.clip = toClip(frames, this.frameSize());
        }
      }
    }
    out.state = this.state;
    return out;
  }
}

function toClip(frames: Buffered[], frameSize: [number, number]): Clip {
  const t0 = frames[0].t;
  return {
    t_ms: frames.map((b) => b.t - t0),
    frame_size: frameSize,
    pose: frames.map((b) => b.f.pose),
    left: frames.map((b) => b.f.left),
    right: frames.map((b) => b.f.right),
    pose_present: frames.map((b) => b.f.pose_present),
    left_present: frames.map((b) => b.f.left_present),
    right_present: frames.map((b) => b.f.right_present),
  };
}
