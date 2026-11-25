"""How many frames per second does recognition need? (evidence for ADR 002's fps rule)

    uv run python -m sanket.checks.fps_sensitivity [--extractor holistic]

INCLUDE is 25 fps. For each lower rate r, keep only the frames a camera at r fps would have
delivered (selected by timestamp: keep a frame once t >= next sample time), and run a
*fresh* extractor on just those frames with their real timestamps (VIDEO mode then tracks
across bigger gaps, as it would live). Compare with the full 25 fps run:

1. Tracking: share of kept frames with a hand, vs the 25 fps run on the same frames.
2. Motion fidelity: interpolate the r-fps hand keypoints linearly back to every 25 fps
   timestamp, and measure the distance to the 25 fps keypoints. That is the information a
   lower frame rate loses. Errors are in shoulder widths (the unit normalization will use)
   and only judged where both runs have the hand and interpolation is bracketed.

Limits: a 25 fps source can't give perfectly even spacing for 20 fps (gaps of 40/40/40/80 ms),
and this is not accuracy. Accuracy vs fps is re-measured with the Phase 2 baseline model.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from sanket.checks.sample import SAMPLES_FILE, VIDEO_CACHE
from sanket.keypoints.extractor import EXTRACTORS, Sequence
from sanket.keypoints.video import iter_frames, open_video
from sanket.paths import CACHE_DIR, REPORTS_DIR
from sanket.spec import fetch_model, load_spec, model_pins

RATES = [25.0, 20.0, 15.0, 12.5, 10.0]
L_SHOULDER, R_SHOULDER = 11, 12  # positions in the stored pose array (indices 0-24 kept as-is)


def run_at_rate(video: Path, make: Any, rate: float) -> dict[str, np.ndarray]:
    cap, info = open_video(video)
    seq = Sequence()
    next_t = 0.0
    try:
        with make() as ex:
            for rgb, t_ms in iter_frames(cap):
                if t_ms + 1e-6 >= next_t:
                    ex.add(rgb, t_ms, seq)
                    next_t += 1000.0 / rate
    finally:
        cap.release()
    out = seq.arrays()
    out["aspect"] = np.array(info.width / info.height)
    return out


def shoulder_width(ref: dict[str, np.ndarray]) -> float:
    a = float(ref["aspect"])
    p = ref["pose"][ref["pose_present"]]
    if len(p) == 0:
        return float("nan")
    d = (p[:, L_SHOULDER, :2] - p[:, R_SHOULDER, :2]) * np.array([a, 1.0])
    return float(np.median(np.linalg.norm(d, axis=1)))


def interp_errors(
    ref: dict[str, np.ndarray], sub: dict[str, np.ndarray], sw: float
) -> tuple[list[float], list[float]]:
    """Per (frame, hand): mean landmark distance / shoulder width. Split into frames the
    subsampled run also saw (tracking effect only) vs frames it skipped (interpolated)."""
    a = float(ref["aspect"])
    scale = np.array([a, 1.0])
    seen: list[float] = []
    skipped: list[float] = []
    kept_t = sub["t_ms"]
    kept_set = set(kept_t.tolist())
    for side in ("left", "right"):
        rp, sp = ref[f"{side}_present"], sub[f"{side}_present"]
        if sp.sum() < 2:
            continue
        st, sx = kept_t[sp], sub[side][sp][:, :, :2] * scale
        for i, t in enumerate(ref["t_ms"]):
            if not rp[i] or t < st[0] or t > st[-1]:
                continue
            j = int(np.searchsorted(st, t))
            if st[min(j, len(st) - 1)] == t:
                est = sx[min(j, len(st) - 1)]
            else:
                t0, t1 = st[j - 1], st[j]
                if t1 - t0 > 2 * 1000 / 10:  # don't interpolate across real detection gaps
                    continue
                w = (t - t0) / (t1 - t0)
                est = (1 - w) * sx[j - 1] + w * sx[j]
            err = float(np.linalg.norm(est - ref[side][i][:, :2] * scale, axis=1).mean() / sw)
            (seen if t in kept_set else skipped).append(err)
    return seen, skipped


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--extractor", default="holistic", choices=sorted(EXTRACTORS))
    args = ap.parse_args()
    models = {p.name: fetch_model(p, CACHE_DIR / "models") for p in model_pins(load_spec())}
    cls = EXTRACTORS[args.extractor]
    videos = json.loads(SAMPLES_FILE.read_text())["extractor_bench"]

    acc: dict[float, dict[str, list[float]]] = {
        r: {"seen": [], "skipped": [], "kept": [], "ref_on_kept": [], "hand_kept": []}
        for r in RATES
    }
    for n, rel in enumerate(videos, 1):
        path = VIDEO_CACHE / rel
        ref = run_at_rate(path, lambda: cls(models), 25.0)
        sw = shoulder_width(ref)
        ref_any = ref["left_present"] | ref["right_present"]
        for r in RATES:
            sub = ref if r == 25.0 else run_at_rate(path, lambda: cls(models), r)
            idx = np.searchsorted(ref["t_ms"], sub["t_ms"])
            sub_any = sub["left_present"] | sub["right_present"]
            acc[r]["kept"].append(len(sub["t_ms"]))
            acc[r]["hand_kept"].append(float(sub_any.sum()))
            acc[r]["ref_on_kept"].append(float(ref_any[idx].sum()))
            if r != 25.0 and np.isfinite(sw):
                seen, skipped = interp_errors(ref, sub, sw)
                acc[r]["seen"] += seen
                acc[r]["skipped"] += skipped
        print(f"[{n}/{len(videos)}] {rel}")

    def pct(xs: list[float], q: float) -> float:
        return float(np.percentile(xs, q)) if xs else float("nan")

    rows = []
    for r in RATES:
        a = acc[r]
        rows.append(
            {
                "fps": r,
                "frames_kept": int(sum(a["kept"])),
                "hand_rate_kept": sum(a["hand_kept"]) / sum(a["kept"]),
                "hand_rate_25fps_same_frames": sum(a["ref_on_kept"]) / sum(a["kept"]),
                "err_seen_median": pct(a["seen"], 50),
                "err_seen_p90": pct(a["seen"], 90),
                "err_skipped_median": pct(a["skipped"], 50),
                "err_skipped_p90": pct(a["skipped"], 90),
                "n_seen": len(a["seen"]),
                "n_skipped": len(a["skipped"]),
            }
        )
    REPORTS_DIR.mkdir(exist_ok=True)
    stem = f"phase1_fps_sensitivity_{args.extractor}"
    (REPORTS_DIR / f"{stem}.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        f"# How much frame rate does recognition need? ({args.extractor}, generated)",
        "",
        f"Generated by `python -m sanket.checks.fps_sensitivity --extractor {args.extractor}` on the {len(videos)} "
        "`extractor_bench` INCLUDE videos (25 fps source). Errors are in **shoulder widths**: mean distance "
        "over the 21 hand landmarks, vs the 25 fps run. *Seen* = frames the lower-rate run also processed "
        "(tracking effect only). *Skipped* = frames it never saw, so linear interpolation from neighbours "
        "(information lost to the lower rate). Not accuracy. That is re-measured with the Phase 2 model.",
        "",
        "| fps | frames kept | hand rate (kept frames) | hand rate, 25 fps run, same frames | error seen: median / p90 | error skipped: median / p90 |",
        "|---|---|---|---|---|---|",
    ]
    for x in rows:
        lines.append(
            f"| {x['fps']:g} | {x['frames_kept']} | {100 * x['hand_rate_kept']:.1f}% | {100 * x['hand_rate_25fps_same_frames']:.1f}% "
            f"| {x['err_seen_median']:.3f} / {x['err_seen_p90']:.3f} | {x['err_skipped_median']:.3f} / {x['err_skipped_p90']:.3f} |"
        )
    (REPORTS_DIR / f"{stem}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
