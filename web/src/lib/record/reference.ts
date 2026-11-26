// INCLUDE reference clips for the recorder: dev-only (served by src/app/dev/reference/[label]/route.dev.ts,
// which production builds never include). Kept pure so the label check is unit-testable.

export const REFERENCE_CREDIT =
  "Reference: INCLUDE dataset (Sridhar, Ganesan, Kumar, Khapra; AI4Bharat / IIT Madras; ACM MM 2020), " +
  "CC BY 4.0. Shown locally for recording only.";

/** The cache file for a word, or null if `label` isn't one of the known vocabulary labels. */
export function referenceFileName(label: string, knownLabels: readonly string[]): string | null {
  if (!/^[a-z]+$/.test(label) || !knownLabels.includes(label)) return null; // no path tricks
  return `${label}.mp4`;
}
