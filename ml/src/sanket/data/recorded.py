"""Real-webcam test set (Phase 6): read session files from the recording tool.

    uv run python -m sanket.data.recorded path/to/sanket-rec-*.json [...]

Validates each file against `spec/recording_v1.json` (the same rules as the browser's
`validateSession`), refuses anything with image/video/name fields, and writes one .npz per clip
into `data/realworld/<session_id>/` in the **same stored layout as INCLUDE extraction** (plus the
label), so the real-world set goes through the identical features and evaluation code.
"""

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from sanket.paths import REPO_ROOT

SPEC = REPO_ROOT / "spec" / "recording_v1.json"
OUT = REPO_ROOT / "data" / "realworld"


def load_spec() -> dict[str, Any]:
    spec: dict[str, Any] = json.loads(SPEC.read_text())
    return spec


def validate_session(s: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    """Errors (empty = valid). Mirrors web/src/lib/record/session.ts `validateSession`."""
    errs: list[str] = []
    text = json.dumps(s)
    errs += [f"forbidden key {k}" for k in spec["forbidden_keys"] if f'"{k}":' in text]
    if s.get("format") != spec["format"]:
        errs.append(f"format {s.get('format')}")
    errs += [f"missing {k}" for k in spec["session_required"] if k not in s]
    if not re.fullmatch(spec["participant_id_pattern"].strip("^$"), str(s.get("participant_id"))):
        errs.append("participant_id")
    consent = s.get("consent") or {}
    errs += [f"consent.{k}" for k in spec["consent_required"] if k not in consent]
    for i, c in enumerate(s.get("clips") or []):
        errs += [f"clip {i}: missing {k}" for k in spec["clip_required"] if k not in c]
        if c.get("kind") not in spec["kinds"]:
            errs.append(f"clip {i}: kind")
        t = c.get("t_ms") or []
        if any(b <= a for a, b in itertools.pairwise(t)):
            errs.append(f"clip {i}: t_ms not increasing")
        for key, (a, b) in spec["shapes"].items():
            arr = c.get(key) or []
            if len(arr) != len(t) or any(len(f) != a or any(len(p) != b for p in f) for f in arr):
                errs.append(f"clip {i}: {key} shape")
    return errs


def clip_arrays(c: dict[str, Any]) -> dict[str, np.ndarray]:
    """One recorded clip → the stored .npz layout (missing parts already zeros in the file)."""
    return {
        "t_ms": np.asarray(c["t_ms"], np.float64),
        "frame_size": np.asarray(c["frame_size"], np.int32),
        "pose": np.asarray(c["pose"], np.float32),
        "left": np.asarray(c["left"], np.float32),
        "right": np.asarray(c["right"], np.float32),
        "pose_present": np.asarray(c["pose_present"], bool),
        "left_present": np.asarray(c["left_present"], bool),
        "right_present": np.asarray(c["right_present"], bool),
        "label_id": np.asarray(c["label_id"], np.int64),
    }


def convert(path: Path, out_root: Path = OUT) -> list[Path]:
    s = json.loads(path.read_text())
    errs = validate_session(s, load_spec())
    if errs:
        raise ValueError(f"{path.name}: invalid session file: {errs[:5]}")
    out_dir = out_root / s["session_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for c in s["clips"]:
        meta = {
            k: c[k]
            for k in (
                "clip_id",
                "kind",
                "label",
                "label_id",
                "take",
                "prompt",
                "started_at",
                "camera_size",
            )
        }
        meta |= {
            k: s[k]
            for k in (
                "session_id",
                "participant_id",
                "spec_version",
                "mediapipe_version",
                "delegate",
            )
        }
        dest = out_dir / f"{c['clip_id']}.npz"
        payload: dict[str, Any] = {
            "meta": np.array(json.dumps(meta, sort_keys=True)),
            **clip_arrays(c),
        }
        np.savez_compressed(dest, **payload)
        written.append(dest)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()
    for f in args.files:
        out = convert(f)
        print(f"{f.name}: {len(out)} clips → {out[0].parent if out else '-'}")


if __name__ == "__main__":
    main()
