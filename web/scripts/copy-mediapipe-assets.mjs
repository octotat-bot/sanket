// Copy the MediaPipe and onnxruntime-web WASM runtimes (from the pinned npm packages) and the shared spec
// into public/, so the browser loads them from our own origin, never a CDN.
// The .task model files are copied separately by `make models`, which verifies their sha256.
import { cpSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const wasmSrc = resolve(web, "node_modules/@mediapipe/tasks-vision/wasm");
const wasmDst = resolve(web, "public/mediapipe/wasm");
const specSrc = resolve(web, "../spec/keypoints.json");
const vocabSrc = resolve(web, "../spec/vocab_include50.json");
const specDst = resolve(web, "public/spec/keypoints.json");

if (!existsSync(wasmSrc)) {
  console.error(`missing ${wasmSrc}: run pnpm install first`);
  process.exit(1);
}
mkdirSync(wasmDst, { recursive: true });
cpSync(wasmSrc, wasmDst, { recursive: true });
mkdirSync(dirname(specDst), { recursive: true });
cpSync(specSrc, specDst);
cpSync(vocabSrc, resolve(dirname(specDst), "vocab_include50.json"));
cpSync(resolve(web, "../spec/recording_v1.json"), resolve(dirname(specDst), "recording_v1.json"));
// onnxruntime-web runtime files (WASM build, and the Asyncify build its WebGPU entry point loads).
const ortSrc = resolve(web, "node_modules/onnxruntime-web/dist");
const ortDst = resolve(web, "public/ort");
mkdirSync(ortDst, { recursive: true });
for (const name of ["ort-wasm-simd-threaded", "ort-wasm-simd-threaded.asyncify"]) {
  for (const ext of [".mjs", ".wasm"]) cpSync(resolve(ortSrc, name + ext), resolve(ortDst, name + ext));
}
console.log("copied MediaPipe wasm, onnxruntime-web runtime and spec files into public/");

if (!existsSync(resolve(web, "public/models/mediapipe/holistic_landmarker.task"))) {
  console.warn("note: public/models/mediapipe/*.task missing; run `make models` from the repo root");
}
