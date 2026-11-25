// The live keypoint path (ADR 002 + ADR 007): frames downscaled to 480 rows, then MediaPipe
// Holistic. Shared by the recording tool and the app, and produces the same per-frame layout as
// offline extraction (spec "stored"): pose 0-24 × (x, y, z, visibility), 21 × (x, y, z) per hand,
// presence flags. Left/right are the signer's own hands (frames are never mirrored; only the
// on-screen preview is).
import { HolisticLandmarker, type HolisticLandmarkerResult, type NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { VerifiedModel, WasmFileset } from "./assets";

export type Delegate = "GPU" | "CPU";
export const N_POSE = 25; // spec stored.pose.indices = 0..24
export const N_HAND = 21;

export interface StoredFrame {
  pose: number[][]; // (25, 4)
  left: number[][]; // (21, 3)
  right: number[][];
  pose_present: boolean;
  left_present: boolean;
  right_present: boolean;
}

/** Create Holistic in VIDEO mode, GPU first and CPU as fallback. Returns the delegate in use. */
export async function createHolistic(
  fileset: WasmFileset,
  models: Record<string, VerifiedModel>,
  prefer: Delegate = "GPU",
): Promise<{ landmarker: HolisticLandmarker; delegate: Delegate }> {
  const order: Delegate[] = prefer === "GPU" ? ["GPU", "CPU"] : ["CPU"];
  let lastError: unknown;
  for (const delegate of order) {
    try {
      const landmarker = await HolisticLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetBuffer: models.holistic_landmarker.bytes.slice(), delegate },
        runningMode: "VIDEO",
        outputFaceBlendshapes: false,
        outputPoseSegmentationMasks: false,
      });
      return { landmarker, delegate };
    } catch (e) {
      lastError = e;
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

const zeros = (n: number, d: number) => Array.from({ length: n }, () => new Array<number>(d).fill(0));

function hand(lms: NormalizedLandmark[] | undefined): [number[][], boolean] {
  if (!lms || lms.length !== N_HAND) return [zeros(N_HAND, 3), false];
  return [lms.map((p) => [p.x, p.y, p.z]), true];
}

/** One Holistic result → the stored per-frame layout (missing parts are zeros, never NaN). */
export function toStoredFrame(r: HolisticLandmarkerResult): StoredFrame {
  const p = r.poseLandmarks[0];
  const posePresent = !!p && p.length >= N_POSE;
  const pose = posePresent ? p.slice(0, N_POSE).map((q) => [q.x, q.y, q.z, q.visibility ?? 0]) : zeros(N_POSE, 4);
  const [left, leftPresent] = hand(r.leftHandLandmarks[0]);
  const [right, rightPresent] = hand(r.rightHandLandmarks[0]);
  return { pose, left, right, pose_present: posePresent, left_present: leftPresent, right_present: rightPresent };
}
