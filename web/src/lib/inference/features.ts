// Features v1: TypeScript mirror of ml/src/sanket/keypoints/features.py (spec
// "normalization.features_v1", ADR 008). Must reproduce spec/golden/features_v1.json to 1e-9;
// see web/tests/unit/features.test.ts. Keep the operation order identical to the Python.

export interface Clip {
  t_ms: number[];
  /** (width, height) of the image MediaPipe actually processed (never the container's claim). */
  frame_size: [number, number];
  pose: number[][][]; // (n, 25, 4): x, y, z, visibility (normalized)
  left: number[][][]; // (n, 21, 3)
  right: number[][][];
  pose_present: boolean[];
  left_present: boolean[];
  right_present: boolean[];
}

export interface FeatureSpec {
  poseIdx: number[];
  visMin: number;
  wrist: number;
  mcp: number;
  maxGap: number;
  T: number;
}

const L_SHOULDER = 11;
const R_SHOULDER = 12;

interface SpecJson {
  normalization: {
    features_v1: {
      pose_feature_indices: number[];
      pose_visibility_min: number;
      palm: { wrist: number; middle_mcp: number };
      max_gap_frames: number;
      resample: { T: number };
    };
  };
}

export function featureSpecFrom(spec: SpecJson): FeatureSpec {
  const f = spec.normalization.features_v1;
  return {
    poseIdx: f.pose_feature_indices,
    visMin: f.pose_visibility_min,
    wrist: f.palm.wrist,
    mcp: f.palm.middle_mcp,
    maxGap: f.max_gap_frames,
    T: f.resample.T,
  };
}

type Part = { start: number; end: number; deps: number[] };

/** Column ranges and presence-flag dependencies, in the spec's feature order. */
export function layout(fs: FeatureSpec) {
  const nPose = 2 * fs.poseIdx.length;
  const nCoord = nPose + 168;
  const parts: Record<string, Part> = {
    pose: { start: 0, end: nPose, deps: [0] },
    left: { start: nPose, end: nPose + 42, deps: [1, 0] },
    right: { start: nPose + 42, end: nPose + 84, deps: [2, 0] },
    left_local: { start: nPose + 84, end: nPose + 126, deps: [1] },
    right_local: { start: nPose + 126, end: nPose + 168, deps: [2] },
  };
  return { parts, nCoord, nFeatures: nCoord + 3 };
}

const aspect = (x: number, width: number, height: number) => x * (width / height);

export function frameFeatures(
  pose: number[][],
  left: number[][],
  right: number[][],
  present: [boolean, boolean, boolean],
  width: number,
  height: number,
  fs: FeatureSpec,
): Float64Array {
  const { parts, nCoord, nFeatures } = layout(fs);
  const out = new Float64Array(nFeatures);
  const [poseOk, leftOk, rightOk] = present;
  let body: { mx: number; my: number; sw: number } | null = null;
  if (poseOk) {
    const px = (i: number) => aspect(pose[i][0], width, height);
    const py = (i: number) => pose[i][1];
    if (pose[L_SHOULDER][3] >= fs.visMin && pose[R_SHOULDER][3] >= fs.visMin) {
      const mx = (px(L_SHOULDER) + px(R_SHOULDER)) / 2;
      const my = (py(L_SHOULDER) + py(R_SHOULDER)) / 2;
      const sw = Math.sqrt((px(L_SHOULDER) - px(R_SHOULDER)) ** 2 + (py(L_SHOULDER) - py(R_SHOULDER)) ** 2);
      if (sw > 1e-6) {
        body = { mx, my, sw };
        fs.poseIdx.forEach((idx, k) => {
          if (pose[idx][3] < fs.visMin) return; // low-visibility point: missing (zeros)
          out[parts.pose.start + 2 * k] = (px(idx) - mx) / sw;
          out[parts.pose.start + 2 * k + 1] = (py(idx) - my) / sw;
        });
      }
    }
  }
  const flags = [body !== null ? 1 : 0, 0, 0];
  ([
    [left, leftOk, "left", 1],
    [right, rightOk, "right", 2],
  ] as const).forEach(([hand, ok, side, k]) => {
    if (!ok) return;
    const hx = (i: number) => aspect(hand[i][0], width, height);
    const hy = (i: number) => hand[i][1];
    const palm = Math.sqrt((hx(fs.mcp) - hx(fs.wrist)) ** 2 + (hy(fs.mcp) - hy(fs.wrist)) ** 2);
    if (palm <= 1e-6) return; // degenerate hand: missing
    flags[k] = 1;
    const local = parts[`${side}_local`];
    const rel = parts[side];
    for (let i = 0; i < 21; i++) {
      out[local.start + 2 * i] = (hx(i) - hx(fs.wrist)) / palm;
      out[local.start + 2 * i + 1] = (hy(i) - hy(fs.wrist)) / palm;
      if (body) {
        out[rel.start + 2 * i] = (hx(i) - body.mx) / body.sw;
        out[rel.start + 2 * i + 1] = (hy(i) - body.my) / body.sw;
      }
    }
  });
  out.set(flags, nCoord);
  return out;
}

const clone3 = (a: number[][][]) => a.map((f) => f.map((p) => [...p]));

/** Raw landmarks: fill runs of ≤ maxGap missing frames bounded by present frames, per part. */
export function fillShortGaps(clip: Clip, maxGap: number): Clip {
  const out: Clip = {
    ...clip,
    pose: clone3(clip.pose),
    left: clone3(clip.left),
    right: clone3(clip.right),
    pose_present: [...clip.pose_present],
    left_present: [...clip.left_present],
    right_present: [...clip.right_present],
  };
  const t = clip.t_ms;
  const n = t.length;
  for (const [arrKey, flagKey] of [
    ["pose", "pose_present"],
    ["left", "left_present"],
    ["right", "right_present"],
  ] as const) {
    const present = clip[flagKey];
    let i = 0;
    while (i < n) {
      if (present[i]) {
        i++;
        continue;
      }
      let j = i;
      while (j < n && !present[j]) j++;
      if (i > 0 && j < n && j - i <= maxGap) {
        const a = i - 1;
        const b = j;
        for (let g = i; g < j; g++) {
          const w = (t[g] - t[a]) / (t[b] - t[a]);
          out[arrKey][g] = out[arrKey][a].map((pa, p) => pa.map((va, c) => (1 - w) * va + w * out[arrKey][b][p][c]));
          out[flagKey][g] = true;
        }
      }
      i = j;
    }
  }
  return out;
}

export function clipFeatures(clip: Clip, fs: FeatureSpec): Float64Array[] {
  const [w, h] = clip.frame_size;
  return clip.t_ms.map((_, i) =>
    frameFeatures(
      clip.pose[i],
      clip.left[i],
      clip.right[i],
      [clip.pose_present[i], clip.left_present[i], clip.right_present[i]],
      w,
      h,
      fs,
    ),
  );
}

const valid = (row: Float64Array, deps: number[], nCoord: number) => deps.every((d) => row[nCoord + d] > 0.5);

/** numpy.linspace semantics: start + i*step, with the last value exactly `stop`. */
function linspace(start: number, stop: number, num: number): number[] {
  if (num === 1) return [start];
  const step = (stop - start) / (num - 1);
  return Array.from({ length: num }, (_, i) => (i === num - 1 ? stop : i * step + start));
}

/** Number of elements <= x (numpy.searchsorted(..., side="right")). */
function searchRight(sorted: number[], x: number): number {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] <= x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export function resample(feats: Float64Array[], tMs: number[], fs: FeatureSpec, T = fs.T): Float64Array[] {
  const { parts, nCoord, nFeatures } = layout(fs);
  const n = feats.length;
  if (n === 0) throw new Error("empty clip");
  if (n === 1) return Array.from({ length: T }, () => Float64Array.from(feats[0]));
  const targets = linspace(tMs[0], tMs[n - 1], T);
  return targets.map((tt) => {
    const out = new Float64Array(nFeatures);
    const j = searchRight(tMs, tt);
    const a = Math.max(j - 1, 0);
    const b = Math.min(j, n - 1);
    const w = b === a ? 0 : (tt - tMs[a]) / (tMs[b] - tMs[a]);
    const nearest = w < 0.5 ? a : b;
    for (const p of Object.values(parts)) {
      if (valid(feats[a], p.deps, nCoord) && valid(feats[b], p.deps, nCoord)) {
        for (let c = p.start; c < p.end; c++) out[c] = (1 - w) * feats[a][c] + w * feats[b][c];
      } else if (valid(feats[nearest], p.deps, nCoord)) {
        for (let c = p.start; c < p.end; c++) out[c] = feats[nearest][c];
      }
    }
    for (let f = 0; f < 3; f++) {
      const pa = feats[a][nCoord + f] > 0.5;
      const pb = feats[b][nCoord + f] > 0.5;
      out[nCoord + f] = pa && pb ? 1 : feats[nearest][nCoord + f];
    }
    return out;
  });
}

export function velocities(seq: Float64Array[], fs: FeatureSpec): Float64Array[] {
  const { parts, nCoord } = layout(fs);
  return seq.map((row, i) => {
    const v = new Float64Array(nCoord);
    if (i === 0) return v;
    for (const p of Object.values(parts)) {
      if (!(valid(row, p.deps, nCoord) && valid(seq[i - 1], p.deps, nCoord))) continue;
      for (let c = p.start; c < p.end; c++) v[c] = row[c] - seq[i - 1][c];
    }
    return v;
  });
}

/** The full shared transform (float64 here; the model input is cast to float32 by the caller). */
export function clipToModelInput(clip: Clip, fs: FeatureSpec, useVelocity = false, T = fs.T): Float64Array[] {
  const filled = fillShortGaps(clip, fs.maxGap);
  const seq = resample(clipFeatures(filled, fs), filled.t_ms, fs, T);
  if (!useVelocity) return seq;
  const vel = velocities(seq, fs);
  return seq.map((row, i) => {
    const out = new Float64Array(row.length + vel[i].length);
    out.set(row, 0);
    out.set(vel[i], row.length);
    return out;
  });
}
