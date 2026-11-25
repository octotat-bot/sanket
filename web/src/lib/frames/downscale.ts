// Downscale frames before MediaPipe exactly as offline extraction does (ADR 007, spec "downscale").
//
// Offline (Python): cv2.resize(..., INTER_AREA) to 480 rows. In the browser,
// createImageBitmap(..., { resizeQuality: "high" }) is the closest match we measured (pixel MAE
// 0.29 vs INTER_AREA, identical in Chromium and Chrome; canvas drawImage differed between
// browsers). The live app always feeds MediaPipe 480-row frames, whatever the camera delivers.

export const EXTRACT_HEIGHT = 480;

/** Spec "downscale.width_rule": keep aspect, even width, halves round UP; never upscale. */
export function targetSize(width: number, height: number, targetHeight = EXTRACT_HEIGHT): [number, number] {
  if (height <= targetHeight) return [width, height];
  return [2 * Math.floor((width * targetHeight) / height / 2 + 0.5), targetHeight];
}

/**
 * The image to hand MediaPipe: `source` downscaled to EXTRACT_HEIGHT rows (or unchanged if it
 * is already that small). The caller owns the returned bitmap and must close() it.
 * `width`/`height` are the true processed size, which normalization must use.
 */
export async function frameForMediaPipe(
  source: ImageBitmapSource,
  sourceWidth: number,
  sourceHeight: number,
): Promise<{ bitmap: ImageBitmap; width: number; height: number }> {
  const [width, height] = targetSize(sourceWidth, sourceHeight);
  const bitmap =
    width === sourceWidth && height === sourceHeight
      ? await createImageBitmap(source)
      : await createImageBitmap(source, { resizeWidth: width, resizeHeight: height, resizeQuality: "high" });
  return { bitmap, width, height };
}
