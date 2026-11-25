import type { Metadata } from "next";
import RecordClient from "./record-client";

export const metadata: Metadata = {
  title: "Record test set (dev) · Sanket",
  description: "Records keypoints only (no video) for the real-webcam test set (Phase 6).",
};

export default function RecordPage() {
  return (
    <main className="mx-auto flex w-full max-w-4xl flex-col gap-6 px-6 py-10">
      <header className="flex flex-col gap-2">
        <h1 className="text-3xl font-semibold tracking-tight">Record the real-webcam test set</h1>
        <p className="text-zinc-600 dark:text-zinc-300">
          Developer tool for Phase 6. It records <strong>keypoints only</strong> (numbers describing
          body and hand positions). No images, video or audio are recorded, and nothing leaves this
          device: you save the session as a file at the end.
        </p>
      </header>
      <RecordClient />
    </main>
  );
}
