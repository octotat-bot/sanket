// From model logits to what the UI shows (ADR 009): calibrated probabilities, top-k, and
// rejection. A wrong word shown with confidence is worse than "not sure", so the app answers only
// when the top class is both probable enough and clearly ahead of the runner-up.

export interface DecideConfig {
  /** Temperature from calibration on INCLUDE-50 val (1 = uncalibrated). */
  temperature: number;
  minProb: number;
  minMargin: number;
  topK: number;
}

export const DEFAULT_DECIDE: DecideConfig = { temperature: 1, minProb: 0.45, minMargin: 0.15, topK: 3 };

export interface Ranked {
  id: number;
  label: string;
  p: number;
}

export interface Decision {
  top: Ranked[];
  accepted: boolean;
  reason?: "low_confidence" | "ambiguous";
}

/** Numerically stable softmax of logits / temperature. */
export function softmax(logits: ArrayLike<number>, temperature = 1): Float64Array {
  const n = logits.length;
  let max = -Infinity;
  for (let i = 0; i < n; i++) max = Math.max(max, logits[i] / temperature);
  const out = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) sum += out[i] = Math.exp(logits[i] / temperature - max);
  for (let i = 0; i < n; i++) out[i] /= sum;
  return out;
}

export function decide(logits: ArrayLike<number>, labels: string[], cfg: DecideConfig = DEFAULT_DECIDE): Decision {
  const p = softmax(logits, cfg.temperature);
  const order = Array.from(p.keys()).sort((a, b) => p[b] - p[a]);
  const top = order.slice(0, Math.max(2, cfg.topK)).map((id) => ({ id, label: labels[id], p: p[id] }));
  const [a, b] = top;
  let reason: Decision["reason"];
  if (a.p < cfg.minProb) reason = "low_confidence";
  else if (a.p - (b?.p ?? 0) < cfg.minMargin) reason = "ambiguous";
  return { top: top.slice(0, cfg.topK), accepted: !reason, reason };
}

/** sha256 of the labels in id order as compact JSON: same bytes as Python's labels_sha256. */
export async function labelsSha256(labels: string[]): Promise<string> {
  const d = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(JSON.stringify(labels)));
  return Array.from(new Uint8Array(d), (b) => b.toString(16).padStart(2, "0")).join("");
}
