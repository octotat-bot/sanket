"""Does OpenCV decode INCLUDE videos correctly? (Phase 1 pre-check, decision 3)

    uv run python -m sanket.checks.video_metadata

For each video in `video_metadata` of ml/data/samples/phase1_checks.json, compare:

* ffprobe (the reference; reads the container directly): display rotation, frame count,
  per-frame presentation timestamps (pts), and whether frame intervals vary (VFR).
* OpenCV with the FFmpeg backend forced (`cv2.CAP_FFMPEG`, the same backend Kaggle/Linux
  uses): reported fps and frame count, the size of decoded frames, and CAP_PROP_POS_MSEC
  per decoded frame.

OpenCV "mishandles" a file if any of these hold:
  1. its decoded frames are not in display orientation (rotation ignored), or
  2. it decodes a different number of frames than ffprobe counts, or
  3. its per-frame timestamps differ from ffprobe's pts by more than half a frame interval.
PyAV is proposed only if at least one sampled file is mishandled.

ffprobe is a local verification tool only (Homebrew `ffmpeg`), not a project dependency.
"""

import itertools
import json
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any

import cv2

from sanket.checks.sample import SAMPLES_FILE, VIDEO_CACHE
from sanket.paths import REPORTS_DIR


def ffprobe_info(path: Path) -> dict[str, Any]:
    exe = shutil.which("ffprobe")
    if exe is None:
        raise RuntimeError("ffprobe not found (brew install ffmpeg)")
    streams = json.loads(
        subprocess.run(
            [
                exe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )["streams"][0]
    pts = json.loads(
        subprocess.run(
            [
                exe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "packet=pts_time",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )["packets"]
    times = sorted(float(p["pts_time"]) for p in pts if "pts_time" in p)
    rotation = 0
    for sd in streams.get("side_data_list", []):
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    rotation = rotation or int(streams.get("tags", {}).get("rotate", 0))
    w, h = int(streams["width"]), int(streams["height"])
    if abs(rotation) % 180 == 90:
        w, h = h, w
    deltas = [b - a for a, b in itertools.pairwise(times)]
    return {
        "codec": streams["codec_name"],
        "coded_size": [int(streams["width"]), int(streams["height"])],
        "display_size": [w, h],
        "rotation": rotation,
        "r_frame_rate": streams.get("r_frame_rate"),
        "avg_frame_rate": streams.get("avg_frame_rate"),
        "n_frames": len(times),
        "pts_s": times,
        "frame_interval_ms": {
            "min": 1000 * min(deltas) if deltas else None,
            "max": 1000 * max(deltas) if deltas else None,
            "median": 1000 * statistics.median(deltas) if deltas else None,
        },
        # More than 1 ms spread between frame intervals = variable frame rate.
        "vfr": bool(deltas) and (max(deltas) - min(deltas)) > 0.001,
    }


def opencv_info(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return {"opened": False}
    info: dict[str, Any] = {
        "opened": True,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count_prop": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "orientation_meta": cap.get(cv2.CAP_PROP_ORIENTATION_META),
        "orientation_auto": cap.get(cv2.CAP_PROP_ORIENTATION_AUTO),
    }
    times_ms: list[float] = []
    size: list[int] | None = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        times_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        if size is None:
            size = [int(frame.shape[1]), int(frame.shape[0])]
    cap.release()
    info["decoded_frames"] = len(times_ms)
    info["decoded_size"] = size
    info["pos_msec"] = times_ms
    return info


def compare(ff: dict[str, Any], cv: dict[str, Any]) -> dict[str, Any]:
    if not cv.get("opened"):
        return {"mishandled": True, "reasons": ["OpenCV could not open the file"]}
    reasons = []
    if cv["decoded_size"] != ff["display_size"]:
        reasons.append(f"orientation: decoded {cv['decoded_size']} vs display {ff['display_size']}")
    if cv["decoded_frames"] != ff["n_frames"]:
        reasons.append(f"frames: decoded {cv['decoded_frames']} vs ffprobe {ff['n_frames']}")
    n = min(len(cv["pos_msec"]), len(ff["pts_s"]))
    t0 = ff["pts_s"][0] if ff["pts_s"] else 0.0
    errs = [abs(cv["pos_msec"][i] - 1000 * (ff["pts_s"][i] - t0)) for i in range(n)]
    max_err = max(errs) if errs else 0.0
    half_frame = 0.5 * (ff["frame_interval_ms"]["median"] or 0.0)
    if max_err > half_frame:
        reasons.append(f"timestamps: max error {max_err:.1f} ms > half frame {half_frame:.1f} ms")
    return {"mishandled": bool(reasons), "reasons": reasons, "max_timestamp_error_ms": max_err}


def main() -> None:
    samples = json.loads(SAMPLES_FILE.read_text())
    rows = []
    for rel in samples["video_metadata"]:
        path = VIDEO_CACHE / rel
        ff, cv = ffprobe_info(path), opencv_info(path)
        cmp = compare(ff, cv)
        rows.append(
            {
                "path": rel,
                "ffprobe": {k: v for k, v in ff.items() if k != "pts_s"},
                "opencv": {k: v for k, v in cv.items() if k != "pos_msec"},
                **cmp,
            }
        )
        print(("MISHANDLED " if cmp["mishandled"] else "ok         ") + rel, *cmp["reasons"])

    ffv = subprocess.run(["ffprobe", "-version"], capture_output=True, text=True).stdout.split(
        "\n"
    )[0]
    summary = {
        "n_files": len(rows),
        "n_mov": sum(r["path"].upper().endswith(".MOV") for r in rows),
        "n_mp4": sum(r["path"].upper().endswith(".MP4") for r in rows),
        "n_rotated": sum(r["ffprobe"]["rotation"] != 0 for r in rows),
        "n_vfr": sum(r["ffprobe"]["vfr"] for r in rows),
        "n_mishandled": sum(r["mishandled"] for r in rows),
        "opencv_version": cv2.__version__,
        "ffprobe_version": ffv,
    }
    summary["pyav_needed"] = summary["n_mishandled"] > 0
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "phase1_video_metadata.json").write_text(
        json.dumps({"summary": summary, "files": rows}, indent=2) + "\n"
    )
    lines = [
        "# Phase 1 pre-check: OpenCV vs container metadata (generated)",
        "",
        "Generated by `uv run python -m sanket.checks.video_metadata`. Do not edit by hand.",
        "",
        f"- Files: {summary['n_files']} ({summary['n_mov']} .MOV, {summary['n_mp4']} .MP4)",
        f"- With rotation metadata: {summary['n_rotated']}",
        f"- Variable frame rate (interval spread > 1 ms): {summary['n_vfr']}",
        f"- Mishandled by OpenCV {summary['opencv_version']} (CAP_FFMPEG): {summary['n_mishandled']}",
        f"- Reference: {summary['ffprobe_version']}",
        f"- **PyAV needed: {'yes' if summary['pyav_needed'] else 'no'}**",
        "",
        "| file | codec | coded size | rotation | fps (r / avg) | frames ffprobe / OpenCV | "
        "interval ms min/med/max | max ts error ms | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        f, c, iv = r["ffprobe"], r["opencv"], r["ffprobe"]["frame_interval_ms"]
        lines.append(
            f"| `{r['path']}` | {f['codec']} | {f['coded_size'][0]}x{f['coded_size'][1]} | {f['rotation']} "
            f"| {f['r_frame_rate']} / {f['avg_frame_rate']} | {f['n_frames']} / {c.get('decoded_frames')} "
            f"| {iv['min']:.2f} / {iv['median']:.2f} / {iv['max']:.2f} "
            f"| {r.get('max_timestamp_error_ms', float('nan')):.2f} "
            f"| {'MISHANDLED: ' + '; '.join(r['reasons']) if r['mishandled'] else 'ok'} |"
        )
    (REPORTS_DIR / "phase1_video_metadata.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
