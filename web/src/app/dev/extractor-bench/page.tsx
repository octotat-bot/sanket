import type { Metadata } from "next";
import BenchClient from "./bench-client";

export const metadata: Metadata = {
  title: "Extractor benchmark (dev) · Sanket",
  description: "Holistic vs Hand+Pose landmark extraction speed on this device (ADR 002).",
};

export default function ExtractorBenchPage() {
  return (
    <main className="mx-auto flex w-full max-w-4xl flex-col gap-6 px-6 py-10">
      <header className="flex flex-col gap-2">
        <h1 className="text-3xl font-semibold tracking-tight">Keypoint extractor benchmark</h1>
        <p className="text-zinc-600 dark:text-zinc-300">
          Developer page for ADR 002. It compares MediaPipe Holistic with Hand + Pose landmarkers,
          on the GPU and CPU delegates, using your camera or a local video file. Nothing leaves this
          device. Results are shown here and can be saved as a local JSON file.
        </p>
      </header>
      <BenchClient />
    </main>
  );
}
