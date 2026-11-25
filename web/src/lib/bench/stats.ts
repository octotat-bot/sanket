// Small, dependency-free statistics for benchmark timings.

/** Percentile with linear interpolation between closest ranks (p in [0, 100]). */
export function percentile(values: readonly number[], p: number): number {
  if (values.length === 0) return Number.NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const rank = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(rank);
  const hi = Math.ceil(rank);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo);
}

export function mean(values: readonly number[]): number {
  if (values.length === 0) return Number.NaN;
  return values.reduce((s, v) => s + v, 0) / values.length;
}

/** Hex sha256 of bytes, via the browser's (or Node's) WebCrypto. */
export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes as BufferSource);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

/** Small seeded PRNG (mulberry32): same seed → same sequence. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface MeanCI {
  mean: number;
  lo: number;
  hi: number;
  n: number;
  excludesZero: boolean;
}

/**
 * Mean with a moving-block bootstrap 95% CI. Consecutive video frames are correlated, so
 * resampling single frames would make the CI too narrow. Each resample instead concatenates
 * random runs of `block` consecutive values (Künsch 1989) until it has n values.
 */
export function blockBootstrapMeanCI(
  values: readonly number[],
  { block = 10, resamples = 2000, seed = 0, alpha = 0.05 } = {},
): MeanCI {
  const n = values.length;
  if (n === 0) return { mean: Number.NaN, lo: Number.NaN, hi: Number.NaN, n, excludesZero: false };
  const b = Math.max(1, Math.min(block, n));
  const rand = mulberry32(seed);
  const means: number[] = [];
  for (let r = 0; r < resamples; r++) {
    let sum = 0;
    let taken = 0;
    while (taken < n) {
      const start = Math.floor(rand() * (n - b + 1));
      for (let k = 0; k < b && taken < n; k++, taken++) sum += values[start + k];
    }
    means.push(sum / n);
  }
  const lo = percentile(means, (100 * alpha) / 2);
  const hi = percentile(means, 100 * (1 - alpha / 2));
  return { mean: mean(values), lo, hi, n, excludesZero: lo > 0 || hi < 0 };
}
