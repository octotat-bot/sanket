// Build-time: make sure the pinned MediaPipe .task files are in public/models/mediapipe, with the
// exact sha256 from spec/keypoints.json. Locally `make models` (Python) already put them there;
// on Vercel (no Python) this downloads them from the pinned, versioned URLs. A mismatch fails the
// build: the browser must use the same model bytes as training (ADR 001).
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const spec = JSON.parse(readFileSync(resolve(web, "../spec/keypoints.json"), "utf8"));
const dest = resolve(web, "public/models/mediapipe");
mkdirSync(dest, { recursive: true });
const sha = (b) => createHash("sha256").update(b).digest("hex");

for (const [name, pin] of Object.entries(spec.extractor.models)) {
  const file = resolve(dest, `${name}.task`);
  if (existsSync(file) && sha(readFileSync(file)) === pin.sha256) continue;
  const res = await fetch(pin.url);
  if (!res.ok) throw new Error(`${name}: HTTP ${res.status} from ${pin.url}`);
  const bytes = Buffer.from(await res.arrayBuffer());
  const got = sha(bytes);
  if (got !== pin.sha256) throw new Error(`${name}: sha256 ${got} != pinned ${pin.sha256}; refusing to build`);
  writeFileSync(file, bytes);
  console.log(`fetched ${name}.task (${bytes.length} bytes, sha256 ok)`);
}
