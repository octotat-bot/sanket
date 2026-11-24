// Phase 0 placeholder. The real pages (Live, Practice, Dictionary, Model card, Benchmark)
// arrive in Phase 7; see CLAUDE.md section 9.
export default function Home() {
  return (
    <main className="mx-auto flex max-w-2xl flex-1 flex-col justify-center gap-6 px-6 py-24">
      <h1 className="text-4xl font-semibold tracking-tight">Sanket</h1>
      <p className="text-lg leading-8 text-zinc-600 dark:text-zinc-300">
        Real-time Indian Sign Language (ISL) <strong>word</strong> recognition that runs entirely
        in your browser. It recognizes isolated signs from a limited vocabulary. It is not a
        translator and not a replacement for interpreters.
      </p>
      <p className="text-zinc-500 dark:text-zinc-400">
        Status: in development (Phase 0: setup and verification). No camera access yet.
      </p>
      <p className="text-sm text-zinc-500 dark:text-zinc-400">
        Trained on the INCLUDE dataset by AI4Bharat / IIT Madras.
      </p>
    </main>
  );
}
