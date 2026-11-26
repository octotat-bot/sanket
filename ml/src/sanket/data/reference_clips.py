"""INCLUDE reference clips for the recorder (dev only, never deployed).

    uv run python -m sanket.data.reference_clips        # or: make reference-clips

For each INCLUDE-50 word, one **train-split** video (the first by path; never val or test) is pulled
out of its Zenodo zip by HTTP Range requests, transcoded to a small 480-row H.264 clip with the
local ffmpeg, and stored in `ml/.cache/reference/<label>.mp4` (gitignored). The original is deleted.
`manifest.json` records the source path, zip, sha256 and the CC BY 4.0 credit shown on screen.

Only the dev server's `/dev/reference/<label>` route serves these files (web/src/app/dev/reference,
a `*.dev.ts` route that production builds never include). Participants imitate these clips, which is
why the real-webcam set is labelled "non-native signers imitating INCLUDE references".
"""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from sanket.checks.sample import MANIFEST, fetch
from sanket.data.sessions import Video, load_manifest
from sanket.data.splits import label_of, load_label_map, load_split, video_key
from sanket.paths import CACHE_DIR

OUT = CACHE_DIR / "reference"
SRC = CACHE_DIR / "reference_src"
CREDIT = (
    "Reference: INCLUDE dataset (Sridhar, Ganesan, Kumar, Khapra; AI4Bharat / IIT Madras; "
    "ACM MM 2020), CC BY 4.0. Shown locally for recording only."
)


def select(videos: list[Video]) -> dict[str, Video]:
    """One train-split video per INCLUDE-50 label: the first by path, skipping renamed copies."""
    by_key = {video_key(v.path): v for v in videos if not v.is_renamed_copy}
    chosen: dict[str, Video] = {}
    for path in sorted(load_split("include50", "train")):
        v = by_key.get(video_key(path))
        if v is not None:
            chosen.setdefault(label_of(path), v)
    missing = set(load_label_map("include50")) - set(chosen)
    if missing:
        raise RuntimeError(f"no train video found for {sorted(missing)}")
    return chosen


def transcode(src: Path, dest: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found (brew install ffmpeg)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            str(src),
            "-vf",
            "scale=-2:480:flags=area",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "28",  # small files: 50 clips should stay around 10 MB in total
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(dest),
        ],
        check=True,
    )


def main() -> None:
    videos = load_manifest(MANIFEST)
    chosen = select(videos)
    mpath = OUT / "manifest.json"
    prior = json.loads(mpath.read_text())["clips"] if mpath.exists() else {}
    manifest: dict[str, Any] = {"credit": CREDIT, "clips": {}}
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [
        v.path
        for label, v in chosen.items()
        if not (
            (OUT / f"{label}.mp4").exists() and prior.get(label, {}).get("source_path") == v.path
        )
    ]
    if todo:  # all missing clips at once: one Range stream per zip, several zips in parallel
        fetch(todo, videos, dest=SRC, workers=6)
    for label, v in sorted(chosen.items()):
        dest = OUT / f"{label}.mp4"
        entry = prior.get(label)
        if not (dest.exists() and entry and entry["source_path"] == v.path):
            src = SRC / v.path
            entry = {
                "source_path": v.path,
                "zip": v.zip,
                "split": "include50/train",
                "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                "file": dest.name,
            }
            transcode(src, dest)
            src.unlink()
        manifest["clips"][label] = entry
        mpath.write_text(json.dumps(manifest, indent=1) + "\n")  # after every clip: crash-safe
        print(f"{label}: {v.path}")
    shutil.rmtree(SRC, ignore_errors=True)
    total = sum(f.stat().st_size for f in OUT.glob("*.mp4"))
    print(f"{len(manifest['clips'])} clips, {total / 1e6:.1f} MB in {OUT}")


if __name__ == "__main__":
    main()
