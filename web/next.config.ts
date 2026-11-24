import type { NextConfig } from "next";

// Privacy guarantee enforced by the browser, not by trust in libraries: the page may only
// make network requests (fetch / XHR / WebSocket) to its own origin. @mediapipe/tasks-vision
// 1.2.3 otherwise POSTs usage metrics to https://odml.pa.googleapis.com/v1/log
// (see docs/decisions/006-block-third-party-requests.md).
const SECURITY_HEADERS = [{ key: "Content-Security-Policy", value: "connect-src 'self'" }];

const nextConfig: NextConfig = {
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
};

export default nextConfig;
