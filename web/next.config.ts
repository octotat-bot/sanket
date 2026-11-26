import type { NextConfig } from "next";
import { PHASE_DEVELOPMENT_SERVER } from "next/constants";

// Privacy guarantee enforced by the browser, not by trust in libraries: the page may only
// make network requests (fetch / XHR / WebSocket) to its own origin. @mediapipe/tasks-vision
// 1.2.3 otherwise POSTs usage metrics to https://odml.pa.googleapis.com/v1/log
// (see docs/decisions/006-block-third-party-requests.md).
//
// COOP + COEP make the page cross-origin isolated (self.crossOriginIsolated === true), which
// SharedArrayBuffer and multi-threaded WASM need (onnxruntime-web, CLAUDE.md section 8). All our
// assets are same-origin, so require-corp blocks nothing we load.
const SECURITY_HEADERS = [
  { key: "Content-Security-Policy", value: "connect-src 'self'" },
  { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
  { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
];

// Dev-only routes (e.g. the INCLUDE reference clips in /dev/record) are files named
// `*.dev.ts(x)`. Only the dev server recognises that extension, so a production build never
// contains them; the route modules also throw if ever loaded in production.
const BASE_EXTENSIONS = ["tsx", "ts", "jsx", "js"];
const DEV_ONLY_EXTENSIONS = ["dev.tsx", "dev.ts"];

const config = (phase: string): NextConfig => ({
  pageExtensions: phase === PHASE_DEVELOPMENT_SERVER ? [...DEV_ONLY_EXTENSIONS, ...BASE_EXTENSIONS] : BASE_EXTENSIONS,
  async headers() {
    return [{ source: "/:path*", headers: SECURITY_HEADERS }];
  },
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
});

export default config;
