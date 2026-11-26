"use client";
// DEV ONLY: a word picker that loops one INCLUDE-50 train-split clip per word, so a non-signer can
// learn the sign before trying it on the Live page. Loaded by live-client.tsx only when
// NODE_ENV === "development" (the import is removed from production builds; the e2e test
// tests/e2e/live.spec.ts checks the build output for REFERENCE_PANEL_MARKER). Clips come from the
// dev-only route /dev/reference/<label> (`make reference-clips`), never from a deployed server.
import { useState } from "react";

export const REFERENCE_PANEL_MARKER = "sanket-dev-reference-panel";

interface Word {
  id: number;
  label: string;
  display: string;
  category: string;
}

export default function ReferencePanel({ words }: { words: Word[] }) {
  const [label, setLabel] = useState(words[0]?.label ?? "");
  // On by default: people copy a clip like a mirror, so a mirrored clip makes them use the
  // same hand as the INCLUDE signer.
  const [mirror, setMirror] = useState(true);
  const [missing, setMissing] = useState<string | null>(null);
  const categories = [...new Set(words.map((w) => w.category))].sort();

  return (
    <section data-testid={REFERENCE_PANEL_MARKER} aria-label="Reference sign" className="rounded-2xl bg-surface p-4 ring-1 ring-line">
      <div className="flex flex-wrap items-center gap-3">
        <label htmlFor="reference-word" className="font-bold">
          Learn a sign
        </label>
        <select
          id="reference-word"
          value={label}
          onChange={(e) => {
            setLabel(e.target.value);
            setMissing(null);
          }}
          className="rounded-lg bg-ink px-3 py-2 ring-1 ring-line"
        >
          {categories.map((c) => (
            <optgroup key={c} label={c}>
              {words
                .filter((w) => w.category === c)
                .map((w) => (
                  <option key={w.label} value={w.label}>
                    {w.display}
                  </option>
                ))}
            </optgroup>
          ))}
        </select>
        <label className="flex items-center gap-2 text-muted">
          <input type="checkbox" checked={mirror} onChange={(e) => setMirror(e.target.checked)} />
          Mirror (copy it like a mirror)
        </label>
      </div>
      {missing ? (
        <p role="alert" className="mt-3 text-muted">
          {missing}
        </p>
      ) : (
        <video
          key={label}
          src={`/dev/reference/${label}`}
          autoPlay
          loop
          muted
          playsInline
          onError={() => setMissing("No clip for this word yet. Run `make reference-clips` and reload.")}
          className={`mt-3 aspect-video w-full rounded-lg bg-ink object-contain ${mirror ? "-scale-x-100" : ""}`}
        />
      )}
      <p className="mt-2 text-sm text-muted">
        Clip: INCLUDE (CC BY 4.0), AI4Bharat / IIT Madras. One training-set video per word, shown on this computer only.
      </p>
    </section>
  );
}
