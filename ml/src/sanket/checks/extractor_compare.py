"""ADR 002, measure (a): Holistic vs Hand+Pose on INCLUDE videos, in Python.

    uv run python -m sanket.checks.extractor_compare

For every video in `extractor_bench` (ml/data/samples/phase1_checks.json), decode each frame
once (OpenCV, FFmpeg backend, unmirrored, RGB) and feed the *same* frames and timestamps to:

  A. HolisticLandmarker                   (holistic_landmarker.task)
  B. HandLandmarker(num_hands=2) + PoseLandmarker
     (hand_landmarker.task + pose_landmarker_full.task)

Both run in VIDEO mode with default confidence thresholds. Metrics are defined in
sanket.keypoints.hand_tracks: hand detection rates, label agreement with the body
(pose wrists 15/16), and temporal label swaps per 100 judged frame transitions.
Python CPU time per frame is recorded too, but browser fps (measure b) is measured in the
browser, not here.
"""

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from sanket.checks.sample import SAMPLES_FILE, VIDEO_CACHE
from sanket.keypoints.hand_tracks import Frame, body_agreement, count_swaps, detection_rates
from sanket.paths import CACHE_DIR, REPORTS_DIR
from sanket.spec import fetch_model, load_spec, model_pins

POSE_LEFT_WRIST, POSE_RIGHT_WRIST = 15, 16
POSE_VISIBILITY_MIN = 0.5


def read_frames(path: Path) -> tuple[list[np.ndarray], list[int], float]:
    """All frames as RGB plus strictly increasing integer timestamps (ms)."""
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    frames: list[np.ndarray] = []
    stamps: list[int] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        ts = round(cap.get(cv2.CAP_PROP_POS_MSEC))
        if stamps and ts <= stamps[-1]:
            ts = stamps[-1] + 1  # VIDEO mode requires strictly increasing timestamps
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        stamps.append(ts)
    aspect = frames[0].shape[1] / frames[0].shape[0] if frames else 1.0
    cap.release()
    return frames, stamps, aspect


def _pt(lm: Any, aspect: float) -> tuple[float, float]:
    return (float(lm.x) * aspect, float(lm.y))


def _pose_wrists(pose: list[Any], aspect: float) -> dict[str, Any]:
    if not pose:
        return {}
    out = {}
    for key, idx in (("pose_left", POSE_LEFT_WRIST), ("pose_right", POSE_RIGHT_WRIST)):
        lm = pose[idx]
        vis = getattr(lm, "visibility", None)
        if vis is None or vis >= POSE_VISIBILITY_MIN:
            out[key] = _pt(lm, aspect)
    return out


def _flat(landmarks: Any) -> list[Any]:
    """Holistic returns one person; accept either [lm, ...] or [[lm, ...]]."""
    if landmarks and isinstance(landmarks[0], list):
        return list(landmarks[0])
    return list(landmarks or [])


def run_holistic(
    model: str, frames: list[np.ndarray], stamps: list[int], aspect: float
) -> tuple[list[Frame], float]:
    opts = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model), running_mode=vision.RunningMode.VIDEO
    )
    out, spent = [], 0.0
    with vision.HolisticLandmarker.create_from_options(opts) as lm:
        for rgb, ts in zip(frames, stamps, strict=True):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            t0 = time.perf_counter()
            r = lm.detect_for_video(img, ts)
            spent += time.perf_counter() - t0
            hands = {}
            for side, lms in (("left", r.left_hand_landmarks), ("right", r.right_hand_landmarks)):
                flat = _flat(lms)
                if flat:
                    hands[side] = _pt(flat[0], aspect)
            out.append(Frame(hands, **_pose_wrists(_flat(r.pose_landmarks), aspect)))
    return out, 1000 * spent / max(len(frames), 1)


def run_hand_pose(
    hand_model: str, pose_model: str, frames: list[np.ndarray], stamps: list[int], aspect: float
) -> tuple[list[Frame], float]:
    hopts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    popts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_model), running_mode=vision.RunningMode.VIDEO
    )
    out, spent = [], 0.0
    with (
        vision.HandLandmarker.create_from_options(hopts) as hl,
        vision.PoseLandmarker.create_from_options(popts) as pl,
    ):
        for rgb, ts in zip(frames, stamps, strict=True):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            t0 = time.perf_counter()
            hr = hl.detect_for_video(img, ts)
            pr = pl.detect_for_video(img, ts)
            spent += time.perf_counter() - t0
            hands: dict[str, tuple[float, float]] = {}
            # Two hands can get the same label; keep the more confident one for that side.
            detected = sorted(
                zip(hr.hand_landmarks, hr.handedness, strict=True), key=lambda h: -h[1][0].score
            )
            for lms, cats in detected:
                side = cats[0].category_name.lower()
                if side not in hands:
                    hands[side] = _pt(lms[0], aspect)
            pose = pr.pose_landmarks[0] if pr.pose_landmarks else []
            out.append(Frame(hands, **_pose_wrists(pose, aspect)))
    return out, 1000 * spent / max(len(frames), 1)


def flip_labels(frames: list[Frame]) -> list[Frame]:
    swap = {"left": "right", "right": "left"}
    return [
        Frame({swap[k]: v for k, v in f.hands.items()}, f.pose_left, f.pose_right) for f in frames
    ]


def metrics(frames: list[Frame]) -> dict[str, Any]:
    ba, sw = body_agreement(frames), count_swaps(frames)
    return {
        **detection_rates(frames),
        "body_judged": ba.judged,
        "body_agree": ba.agree,
        "swap_transitions": sw.transitions,
        "swaps": sw.swaps,
    }


def pooled(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    n = sum(r["n_frames"] for r in rows)
    m = [r[key] for r in rows]
    judged, agree = sum(x["body_judged"] for x in m), sum(x["body_agree"] for x in m)
    trans, swaps = sum(x["swap_transitions"] for x in m), sum(x["swaps"] for x in m)
    rate = {
        k: sum(x[k] * r["n_frames"] for x, r in zip(m, rows, strict=True)) / n
        for k in ("any_hand", "both_hands", "left", "right", "pose")
    }
    return {
        **rate,
        "body_agreement": agree / judged if judged else None,
        "body_judged": judged,
        "swaps_per_100": 100 * swaps / trans if trans else None,
        "swap_transitions": trans,
        "swaps": swaps,
        "videos_with_zero_hand_frames": sum(x["any_hand"] == 0 for x in m),
        "ms_per_frame_python_cpu": sum(r[key + "_ms"] * r["n_frames"] for r in rows) / n,
    }


def main() -> None:
    models = {p.name: str(fetch_model(p, CACHE_DIR / "models")) for p in model_pins(load_spec())}
    samples = json.loads(SAMPLES_FILE.read_text())
    rows = []
    for i, rel in enumerate(samples["extractor_bench"]):
        frames, stamps, aspect = read_frames(VIDEO_CACHE / rel)
        a, a_ms = run_holistic(models["holistic_landmarker"], frames, stamps, aspect)
        b, b_ms = run_hand_pose(
            models["hand_landmarker"], models["pose_landmarker_full"], frames, stamps, aspect
        )
        row = {
            "path": rel,
            "n_frames": len(frames),
            "size": [int(frames[0].shape[1]), int(frames[0].shape[0])] if frames else None,
            "holistic": metrics(a),
            "holistic_ms": a_ms,
            "hand_pose": metrics(b),
            "hand_pose_ms": b_ms,
            "hand_pose_flipped": metrics(flip_labels(b)),
            "hand_pose_flipped_ms": b_ms,
        }
        rows.append(row)
        print(
            f"[{i + 1}/{len(samples['extractor_bench'])}] {rel}: {len(frames)} frames, "
            f"A {a_ms:.0f} ms, B {b_ms:.0f} ms"
        )

    summary: dict[str, Any] = {
        k: pooled(rows, k) for k in ("holistic", "hand_pose", "hand_pose_flipped")
    }
    summary["n_videos"] = len(rows)
    summary["n_frames"] = sum(r["n_frames"] for r in rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "phase1_extractor_compare.json").write_text(
        json.dumps({"summary": summary, "videos": rows}, indent=2) + "\n"
    )

    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{100 * x:.1f}%"

    lines = [
        "# ADR 002 measure (a): Holistic vs Hand+Pose in Python (generated)",
        "",
        "Generated by `uv run python -m sanket.checks.extractor_compare`. Do not edit by hand.",
        f"{summary['n_videos']} INCLUDE videos, {summary['n_frames']} frames, identical frames and "
        "timestamps for both extractors, VIDEO mode, default thresholds. Rates are pooled over frames.",
        "",
        "| metric | A: Holistic | B: Hand+Pose (raw labels) | B with labels flipped |",
        "|---|---|---|---|",
    ]
    rowsdef: list[tuple[str, str, Callable[[Any], str]]] = [
        ("frames with any hand", "any_hand", pct),
        ("frames with both hands", "both_hands", pct),
        ("frames with a 'left' hand", "left", pct),
        ("frames with a 'right' hand", "right", pct),
        ("frames with both pose wrists visible", "pose", pct),
        ("label agrees with nearer pose wrist", "body_agreement", pct),
        ("hands judged for body agreement", "body_judged", str),
        (
            "label swaps per 100 judged transitions",
            "swaps_per_100",
            lambda x: "n/a" if x is None else f"{x:.2f}",
        ),
        ("judged transitions", "swap_transitions", str),
        ("videos with zero hand frames", "videos_with_zero_hand_frames", str),
        ("Python CPU ms/frame (info only)", "ms_per_frame_python_cpu", lambda x: f"{x:.1f}"),
    ]
    for name, key, fmt in rowsdef:
        lines.append(
            f"| {name} | "
            + " | ".join(
                fmt(summary[m][key]) for m in ("holistic", "hand_pose", "hand_pose_flipped")
            )
            + " |"
        )
    lines += [
        "",
        "## Per video",
        "",
        "| video | frames | A any hand | B any hand | A agree | B agree (raw) | A swaps | B swaps (raw) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        a, b = r["holistic"], r["hand_pose"]

        def agree(x: dict[str, Any]) -> str:
            return f"{x['body_agree']}/{x['body_judged']}"

        lines.append(
            f"| `{r['path']}` | {r['n_frames']} | {pct(a['any_hand'])} | {pct(b['any_hand'])} "
            f"| {agree(a)} | {agree(b)} | {a['swaps']}/{a['swap_transitions']} | {b['swaps']}/{b['swap_transitions']} |"
        )
    (REPORTS_DIR / "phase1_extractor_compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
