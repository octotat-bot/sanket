"""Pick (deterministically) and fetch the small video samples used by the Phase 1 pre-checks.

    uv run python -m sanket.checks.sample            # write ml/data/samples/phase1_checks.json
    uv run python -m sanket.checks.sample --fetch    # also download those videos (~1 GB)

Three samples, all seeded:

* session_spotcheck: 5 classes x 2 camera-number runs. For each run, the first and last take,
  plus the take that follows the run in a *different* class of the same category, if one
  exists. That tests "one run = one signer/session" and whether sessions span classes.
* extractor_bench: 50 random videos for Holistic vs Hand+Pose (ADR 002, measure a).
* video_metadata: 30 .MOV files (+ the .MP4 files) from extractor_bench, for the
  rotation / variable-fps check.

Videos go to ml/.cache/videos/<zip path> (gitignored). Each one is pulled out of its zip
with HTTP Range requests and CRC-checked, so no full zip is downloaded.
"""

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from sanket.data import remote_zip, zenodo
from sanket.data.sessions import Video, class_runs, load_manifest
from sanket.paths import CACHE_DIR, REPO_ROOT

MANIFEST = REPO_ROOT / "ml" / "data" / "manifests" / "zenodo_4010759_videos.tsv"
SAMPLES_FILE = REPO_ROOT / "ml" / "data" / "samples" / "phase1_checks.json"
VIDEO_CACHE = CACHE_DIR / "videos"
SEED = 0


def _neighbor_after(run: list[Video], videos: list[Video]) -> Video | None:
    """The closest take numbered just after the run, in another class of the same category."""
    last = run[-1]
    assert last.number is not None
    cands = [
        v
        for v in videos
        if v.category == last.category
        and v.ext == last.ext
        and v.class_folder != last.class_folder
        and v.number is not None
        and not v.is_renamed_copy
        and 0 < v.number - last.number <= 3
    ]
    return min(cands, key=lambda v: (v.number, v.path)) if cands else None


def select(videos: list[Video], seed: int = SEED) -> dict[str, Any]:
    rng = random.Random(seed)
    runs = [r for r in class_runs(videos) if len(r) >= 3 and not r[0].is_renamed_copy]
    by_class: dict[str, list[list[Video]]] = defaultdict(list)
    for r in runs:
        by_class[r[0].class_folder].append(r)
    eligible = sorted(c for c, rs in by_class.items() if len(rs) >= 2)
    with_mp4 = [c for c in eligible if any(r[0].ext == ".MP4" for r in by_class[c])]

    # 1 class that has an .MP4 run (to cover the second camera) + 4 others.
    mp4_class = rng.choice(with_mp4)
    classes = [mp4_class, *rng.sample([c for c in eligible if c != mp4_class], 4)]

    spot: list[dict[str, Any]] = []
    for ci, cls in enumerate(classes):
        cls_runs = by_class[cls]
        if cls == mp4_class:
            picked = [
                rng.choice([r for r in cls_runs if r[0].ext == ".MP4"]),
                rng.choice([r for r in cls_runs if r[0].ext == ".MOV"]),
            ]
        else:
            picked = rng.sample(cls_runs, 2)
        for ri, run in enumerate(picked):
            run_id = f"c{ci}r{ri}"
            span = f"{run[0].number}-{run[-1].number}"
            spot.append({"run": run_id, "span": span, "role": "first", "path": run[0].path})
            spot.append({"run": run_id, "span": span, "role": "last", "path": run[-1].path})
            nb = _neighbor_after(run, videos)
            if nb is not None:
                spot.append({"run": run_id, "span": span, "role": "neighbor", "path": nb.path})

    plain = sorted((v for v in videos if not v.is_renamed_copy), key=lambda v: v.path)
    bench = sorted(v.path for v in rng.sample(plain, 50))
    movs = [p for p in bench if p.upper().endswith(".MOV")]
    mp4s = [p for p in bench if p.upper().endswith(".MP4")]
    return {
        "seed": seed,
        "session_spotcheck": spot,
        "extractor_bench": bench,
        "video_metadata": sorted(rng.sample(movs, min(30, len(movs))) + mp4s),
    }


def all_paths(samples: dict[str, Any]) -> list[str]:
    paths = {s["path"] for s in samples["session_spotcheck"]}
    paths |= set(samples["extractor_bench"]) | set(samples["video_metadata"])
    return sorted(paths)


def fetch(
    paths: list[str], videos: list[Video], dest: Path = VIDEO_CACHE, workers: int = 4
) -> None:
    """Fetch videos, one zip per worker thread (a few parallel Range streams; Zenodo is slow
    per connection). Files already on disk are skipped, so an interrupted fetch resumes."""
    zip_of = {v.path: v.zip for v in videos}
    by_zip: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        by_zip[zip_of[p]].append(p)

    def one(item: tuple[str, list[str]]) -> str:
        z, names = item
        remote_zip.extract_remote_members(zenodo.file_url(z), names, dest)
        return f"{z}: {len(names)} videos"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(one, sorted(by_zip.items())):
            print(line, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fetch", action="store_true", help="download the sampled videos")
    args = parser.parse_args()
    videos = load_manifest(MANIFEST)
    samples = select(videos)
    SAMPLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SAMPLES_FILE.write_text(json.dumps(samples, indent=2) + "\n")
    paths = all_paths(samples)
    mb = sum(v.bytes for v in videos if v.path in set(paths)) / 1e6
    print(f"wrote {SAMPLES_FILE}: {len(paths)} distinct videos, {mb:.0f} MB")
    if args.fetch:
        fetch(paths, videos)


if __name__ == "__main__":
    main()
