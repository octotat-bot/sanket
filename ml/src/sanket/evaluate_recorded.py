"""Run the SHIPPED model on recorder clips, offline, with no spotter: is the model or the live
segmentation to blame for bad live results?

    uv run python -m sanket.evaluate_recorded --sessions S-ABC234 [S-...]   # or: make eval-recorded

Clips come from `make import-recordings` (data/realworld/<session>/*.npz, gitignored). The model is
the exported ONNX fp32 file named in web/public/models/include50.labels.json, with that file's T,
labels and decision thresholds, so this is exactly what the browser runs, minus the spotter.

Variants, each scored separately:
  recorded           the fixed-length clip as recorded
  mirrored           left/right flipped (hands swapped, x mirrored): tests "used the other hand"
  trimmed            cut to the span where a hand is visible (± 5 frames): INCLUDE clips are
                     trimmed signs, the recorder's fixed window also holds rest time
  trimmed+mirrored

Reports top-1/top-5, how often the app would say "not sure" (and accuracy when it doesn't), per
word, and for "no sign" clips how often they would be rejected. Aggregate numbers only are written
to reports/diagnostics/; per-clip predictions go to stdout.
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from sanket.augment.transforms import mirror
from sanket.data.recorded import OUT as RECORDED
from sanket.keypoints.features import FeatureSpec, clip_to_model_input
from sanket.paths import REPO_ROOT

META = REPO_ROOT / "web" / "public" / "models" / "include50.labels.json"
REPORT_DIR = REPO_ROOT / "reports" / "diagnostics"
TRIM_MARGIN = 5  # frames kept before the first / after the last frame with a hand

Clip = dict[str, np.ndarray]


def trim_to_hands(clip: Clip, margin: int = TRIM_MARGIN) -> Clip:
    hands = np.flatnonzero(clip["left_present"] | clip["right_present"])
    if len(hands) == 0:
        return clip
    a = max(0, int(hands[0]) - margin)
    b = min(len(clip["t_ms"]), int(hands[-1]) + margin + 1)
    out = {k: (v[a:b] if v.ndim and len(v) == len(clip["t_ms"]) else v) for k, v in clip.items()}
    out["t_ms"] = out["t_ms"] - out["t_ms"][0]
    return out


VARIANTS: dict[str, Callable[[Clip], Clip]] = {
    "recorded": lambda c: c,
    "mirrored": mirror,
    "trimmed": trim_to_hands,
    "trimmed+mirrored": lambda c: mirror(trim_to_hands(c)),
}


def softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = z / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    out: np.ndarray = e / e.sum(axis=-1, keepdims=True)
    return out


def score(
    items: list[tuple[Clip, dict[str, Any]]],
    predict: Callable[[np.ndarray], np.ndarray],
    fs: FeatureSpec,
    T: int,
    labels: list[str],
    cal: dict[str, float],
    variant: Callable[[Clip], Clip],
) -> dict[str, Any]:
    """predict: (n, T, F) float32 → (n, C) logits."""
    if not items:
        return {"n": 0}
    x = np.stack([clip_to_model_input(variant(c), fs, T=T) for c, _ in items]).astype(np.float32)
    p = softmax(predict(x), cal["temperature"])
    order = np.argsort(-p, axis=1)
    top = p[np.arange(len(p)), order[:, 0]]
    second = p[np.arange(len(p)), order[:, 1]]
    accepted = (top >= cal["min_prob"]) & (top - second >= cal["min_margin"])
    kinds = np.array([m["kind"] for _, m in items])
    y = np.array([m["label_id"] for _, m in items])
    sign = kinds == "sign"
    out: dict[str, Any] = {"n_sign": int(sign.sum()), "n_no_sign": int((~sign).sum())}
    if sign.any():
        top1 = order[sign, 0] == y[sign]
        top5 = (order[sign, :5] == y[sign, None]).any(axis=1)
        acc_s = accepted[sign]
        per_word: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for ok, lab in zip(top1, y[sign], strict=True):
            per_word[labels[lab]][0] += int(ok)
            per_word[labels[lab]][1] += 1
        out |= {
            "top1": float(top1.mean()),
            "top5": float(top5.mean()),
            "accepted_rate": float(acc_s.mean()),
            "accuracy_when_accepted": float(top1[acc_s].mean()) if acc_s.any() else None,
            "correct_but_rejected": int((top1 & ~acc_s).sum()),
            "mean_top_prob": float(top[sign].mean()),
            "per_word": {w: f"{a}/{n}" for w, (a, n) in sorted(per_word.items())},
            "clips": [
                {
                    "label": labels[lab],
                    "pred": [labels[i] for i in o[:3]],
                    "p": [round(float(pp[i]), 3) for i in o[:3]],
                    "accepted": bool(a),
                }
                for lab, o, pp, a in zip(y[sign], order[sign], p[sign], accepted[sign], strict=True)
            ],
        }
    if (~sign).any():
        out["no_sign_rejected_rate"] = float((~accepted[~sign]).mean())
    return out


def load_clips(sessions: list[str]) -> list[tuple[Clip, dict[str, Any]]]:
    items = []
    for s in sessions:
        files = sorted((RECORDED / s).glob("*.npz"))
        if not files:
            raise SystemExit(f"no clips for {s} in {RECORDED}: run `make import-recordings` first")
        for f in files:
            d = dict(np.load(f))
            meta = json.loads(str(d.pop("meta")))
            meta["label_id"] = int(d.pop("label_id"))
            items.append((d, meta))
    return items


def main() -> None:
    import onnxruntime as ort

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sessions", nargs="+", required=True, help="session ids (S-XXXXXX)")
    ap.add_argument("--meta", type=Path, default=META)
    ap.add_argument("--tag", default=None, help="report name (default: the session ids)")
    args = ap.parse_args()

    meta = json.loads(args.meta.read_text())
    labels: list[str] = meta["labels"]
    cal = meta.get("calibration") or {"temperature": 1.0, "min_prob": 0.45, "min_margin": 0.15}
    sess = ort.InferenceSession(
        str(args.meta.parent / meta["files"]["fp32"]), providers=["CPUExecutionProvider"]
    )

    def predict(x: np.ndarray) -> np.ndarray:
        out: np.ndarray = sess.run(["logits"], {"x": x})[0]
        return out

    fs = FeatureSpec.from_spec()
    items = load_clips(args.sessions)
    report: dict[str, Any] = {
        "sessions": args.sessions,
        "model": meta["files"]["fp32"],
        "model_sha256": meta["sha256"][meta["files"]["fp32"]],
        "decision": cal,
        "variants": {},
    }
    for name, fn in VARIANTS.items():
        r = score(items, predict, fs, meta["T"], labels, cal, fn)
        clips = r.pop("clips", [])
        report["variants"][name] = r
        if "top1" in r:
            print(
                f"{name:18s} top-1 {100 * r['top1']:5.1f}%  top-5 {100 * r['top5']:5.1f}%  "
                f"accepted {100 * r['accepted_rate']:5.1f}%  mean top p {r['mean_top_prob']:.2f}"
                + (
                    f"  no-sign rejected {100 * r['no_sign_rejected_rate']:.0f}%"
                    if "no_sign_rejected_rate" in r
                    else ""
                )
            )
        if name == "recorded":
            for c in clips:
                mark = "ok " if c["pred"][0] == c["label"] else "ERR"
                print(
                    f"    {mark} {c['label']:12s} -> "
                    + ", ".join(f"{w} {p:.2f}" for w, p in zip(c["pred"], c["p"], strict=True))
                    + ("" if c["accepted"] else "  (not sure)")
                )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or "_".join(args.sessions)
    (REPORT_DIR / f"recorded_{tag}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"→ {REPORT_DIR / f'recorded_{tag}.json'}")


if __name__ == "__main__":
    main()
