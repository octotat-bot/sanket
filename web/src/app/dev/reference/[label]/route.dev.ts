// DEV ONLY: serves INCLUDE reference clips from the local cache (ml/.cache/reference, made by
// `make reference-clips`) to the recorder at /dev/record. The `.dev.ts` extension is only
// recognised by the dev server (next.config.ts), so production builds never contain this route.
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { referenceFileName } from "@/lib/record/reference";

if (process.env.NODE_ENV === "production") {
  // Defence in depth: if this module is ever pulled into a production build, fail loudly.
  throw new Error("dev-only route /dev/reference was loaded in a production build: refusing");
}

const repo = resolve(process.cwd(), "..");

export async function GET(_req: Request, { params }: { params: Promise<{ label: string }> }) {
  const { label } = await params;
  const vocab = JSON.parse(await readFile(resolve(repo, "spec/vocab_include50.json"), "utf8")) as {
    words: { label: string }[];
  };
  const name = referenceFileName(
    label,
    vocab.words.map((w) => w.label),
  );
  if (!name) return new Response("unknown word", { status: 404 });
  try {
    const bytes = await readFile(resolve(repo, "ml/.cache/reference", name));
    return new Response(new Uint8Array(bytes), {
      headers: { "content-type": "video/mp4", "cache-control": "no-store" },
    });
  } catch {
    return new Response("reference clip not prepared: run `make reference-clips`", { status: 404 });
  }
}
