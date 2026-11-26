import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import LiveClient from "./live-client";

interface Word {
  id: number;
  label: string;
  display: string;
  category: string;
}

// Read at build time (the page is prerendered): spec/ is the single source of truth.
const vocab = JSON.parse(readFileSync(resolve(process.cwd(), "../spec/vocab_include50.json"), "utf8")) as {
  words: Word[];
};
const words = vocab.words;
const byCategory = Object.entries(
  words.reduce<Record<string, Word[]>>((acc, w) => {
    (acc[w.category] ??= []).push(w);
    return acc;
  }, {}),
).sort(([a], [b]) => a.localeCompare(b));

export default function Home() {
  return (
    <main className="mx-auto w-full max-w-7xl flex-1 px-5 py-8 sm:px-8 lg:py-12">
      <header className="mb-8 max-w-3xl">
        <h1 className="font-display text-4xl font-extrabold tracking-tight sm:text-5xl">Sanket</h1>
        <p className="mt-3 text-xl leading-relaxed text-muted">
          Sign an Indian Sign Language word to your webcam and see it recognised as you sign. It runs entirely in your
          browser: your video never leaves this device.
        </p>
      </header>

      <LiveClient words={words} />

      <section aria-labelledby="words" className="mt-16">
        <h2 id="words" className="font-display text-3xl font-bold">
          The {words.length} words it knows
        </h2>
        <p className="mt-2 max-w-2xl text-lg text-muted">
          These are the INCLUDE-50 signs. Sign one at a time, starting and ending with your hands lowered.
        </p>
        <div className="mt-6 grid gap-x-10 gap-y-6 sm:grid-cols-2 lg:grid-cols-3">
          {byCategory.map(([category, ws]) => (
            <div key={category}>
              <h3 className="text-lg font-bold">{category}</h3>
              <p className="mt-1 leading-relaxed text-muted">{ws.map((w) => w.display).join(", ")}</p>
            </div>
          ))}
        </div>
      </section>

      <section aria-labelledby="limits" className="mt-16 grid gap-10 lg:grid-cols-3">
        <div>
          <h2 id="limits" className="font-display text-2xl font-bold">
            What it is, and isn&apos;t
          </h2>
          <p className="mt-3 leading-relaxed text-muted">
            Sanket recognises single, isolated ISL words from a vocabulary of {words.length}. It does not understand
            sentences or fingerspelling, and it is not a translator or a replacement for an interpreter. ISL is a full
            language with its own grammar; this covers a tiny part of it. Accuracy on everyday webcams is lower than on
            the lab-recorded dataset it was trained on.
          </p>
        </div>
        <div>
          <h2 className="font-display text-2xl font-bold">Privacy</h2>
          <p className="mt-3 leading-relaxed text-muted">
            The camera feed is turned into body and hand keypoints by MediaPipe, and a small model classifies them, all
            inside this tab. The page is only allowed to contact its own server, which sends it the model files. No
            video, images or keypoints are uploaded or stored.
          </p>
        </div>
        <div>
          <h2 className="font-display text-2xl font-bold">Credit</h2>
          <p className="mt-3 leading-relaxed text-muted">
            Trained on the{" "}
            <a className="underline underline-offset-4" href="https://zenodo.org/records/4010759">
              INCLUDE dataset
            </a>{" "}
            by Sridhar, Ganesan, Kumar and Khapra (AI4Bharat and IIT Madras, ACM Multimedia 2020), licensed CC BY 4.0.
            Keypoints by Google MediaPipe. Code is{" "}
            <a className="underline underline-offset-4" href="https://github.com/octotat-bot/sanket">
              open source
            </a>{" "}
            under the MIT license.
          </p>
        </div>
      </section>
    </main>
  );
}
