"""Disk-safe keypoint extraction over the whole INCLUDE dataset (Phase 1).

    uv run python -m sanket.pipeline.extract --config configs/extract.yaml [--zips A.zip B.zip]

Per zip, one at a time: check disk → download (resumable, md5) → for each video:
unpack to a temp file → stream frames through a *fresh* extractor → write a compact .npz →
cache one small middle frame (ADR 005, local only) → log → delete the temp video. Then delete
the zip and mark it done. The full dataset (57 GB) is never on disk at once. Peak use is
about one zip (≤ 1.85 GB) plus one video.

Edge cases (CLAUDE.md section 8):
* An unreadable or corrupt video is logged with status "error" and skipped. The batch never crashes.
* A video with near-zero hand detection is still saved, and listed in `_review.jsonl`.
* Every video's sha256 is logged, so duplicates across classes or splits can be found.
* Free disk below the floor (10 GB) → LowDiskSpace: stop, delete our partial download.
* Rerunning resumes: finished zips are skipped, finished videos inside a zip are skipped.
"""

import argparse
import collections
import hashlib
import io
import json
import math
import platform
import subprocess
import tempfile
import threading
import time
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from sanket import net
from sanket.data import remote_zip, zenodo
from sanket.data.disk import GB, DiskGuard
from sanket.data.download import download
from sanket.data.splits import label_of, video_key
from sanket.keypoints.extractor import EXTRACTORS, Extractor, Sequence
from sanket.keypoints.video import iter_frames, open_video
from sanket.paths import CACHE_DIR, REPO_ROOT
from sanket.spec import fetch_model, load_spec, model_pins

FILES_MANIFEST = REPO_ROOT / "ml" / "data" / "manifests" / "zenodo_4010759_files.json"
VIDEOS_MANIFEST = REPO_ROOT / "ml" / "data" / "manifests" / "zenodo_4010759_videos.tsv"
VIDEO_EXTS = (".mov", ".mp4")
FRAME_CACHE_WIDTH = 320


@dataclass(frozen=True)
class Config:
    extractor: str
    raw_dir: Path
    out_dir: Path
    frame_cache_dir: Path | None
    min_free_gb: float
    review_below_hand_rate: float
    extract_height: int | None = None  # None = native resolution (ADR 007)

    @staticmethod
    def load(path: Path) -> "Config":
        d = yaml.safe_load(path.read_text())
        if d.get("extractor") not in EXTRACTORS:
            choices = sorted(EXTRACTORS)
            raise SystemExit(
                f"config 'extractor' must be one of {choices} (ADR 002), got {d.get('extractor')!r}"
            )

        def p(v: str) -> Path:
            return (REPO_ROOT / v) if not Path(v).is_absolute() else Path(v)

        return Config(
            extractor=d["extractor"],
            raw_dir=p(d["raw_dir"]),
            out_dir=p(d["out_dir"]),
            frame_cache_dir=p(d["frame_cache_dir"]) if d.get("frame_cache_dir") else None,
            min_free_gb=float(d["min_free_gb"]),
            review_below_hand_rate=float(d["review_below_hand_rate"]),
            extract_height=int(d["extract_height"]) if d.get("extract_height") else None,
        )


def resize_to_height(rgb: np.ndarray, target_height: int | None) -> np.ndarray:
    """Downscale (never upscale) to `target_height` rows, keeping aspect; width rounded to even.
    INTER_AREA averages pixels, the closest OpenCV analogue to a camera's own downscaling."""
    h, w = rgb.shape[:2]
    if target_height is None or h <= target_height:
        return rgb
    return cv2.resize(rgb, target_size(w, h, target_height), interpolation=cv2.INTER_AREA)


def target_size(width: int, height: int, target_height: int) -> tuple[int, int]:
    """Spec "downscale.width_rule": even width, halves round UP (Python's round() would round
    halves to even and disagree with JavaScript's Math.round). Never upscales."""
    if height <= target_height:
        return width, height
    return 2 * math.floor(width * target_height / height / 2 + 0.5), target_height


def extract_video(
    video: Path, make_extractor: Any, frame_out: Path | None, target_height: int | None = None
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run one video through a fresh extractor. Returns (arrays, stats).

    `frame_size` (stored in the arrays) is the size of the image MediaPipe actually processed:
    the decoded frame after any downscaling, not the container's claim. Normalization must use
    it (sanket.keypoints.normalize).
    """
    cap, info = open_video(video)
    seq = Sequence()
    middle = max(info.frame_count // 2, 0)
    frame_size: tuple[int, int] | None = None
    try:
        ex: Extractor
        with make_extractor() as ex:
            for i, (decoded, t_ms) in enumerate(iter_frames(cap)):
                rgb = resize_to_height(decoded, target_height)
                size = (int(rgb.shape[1]), int(rgb.shape[0]))
                if frame_size is None:
                    frame_size = size
                elif size != frame_size:
                    raise RuntimeError(f"frame size changed mid-video: {frame_size} -> {size}")
                ex.add(rgb, t_ms, seq)
                if frame_out is not None and i == middle:
                    h, w = rgb.shape[:2]
                    small = cv2.resize(rgb, (FRAME_CACHE_WIDTH, round(h * FRAME_CACHE_WIDTH / w)))
                    frame_out.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(frame_out),
                        cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 80],
                    )
    finally:
        cap.release()
    arrays = seq.arrays()
    arrays["frame_size"] = np.asarray(
        frame_size or (0, 0), np.int32
    )  # (width, height) as processed
    n = len(seq.t_ms)
    lp, rp = arrays["left_present"], arrays["right_present"]
    stats = {
        "fps": info.fps,
        "width": frame_size[0] if frame_size else 0,  # as processed by MediaPipe
        "height": frame_size[1] if frame_size else 0,
        "container_width": info.width,  # what the container claims (logged, never used)
        "container_height": info.height,
        "frame_count_container": info.frame_count,
        "n_frames": n,
        "duration_s": (seq.t_ms[-1] - seq.t_ms[0]) / 1000 if n > 1 else 0.0,
        "rate_pose": float(arrays["pose_present"].mean()) if n else 0.0,
        "rate_left": float(lp.mean()) if n else 0.0,
        "rate_right": float(rp.mean()) if n else 0.0,
        "rate_any_hand": float((lp | rp).mean()) if n else 0.0,
        "rate_both_hands": float((lp & rp).mean()) if n else 0.0,
        "dropped_duplicate_hands": seq.dropped_duplicate_hands,
    }
    return arrays, stats


def save_npz(path: Path, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    payload: dict[str, Any] = {"meta": np.array(json.dumps(meta, sort_keys=True)), **arrays}
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)


def _git_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def process_video(
    rel: str,
    data: bytes,
    zip_key: str,
    cfg: Config,
    make_extractor: Any,
    base_meta: dict[str, Any],
    tmpd: Path,
    log: Any,
) -> str:
    """One video's bytes → .npz + log record. Processing errors are recorded, never raised."""
    out = cfg.out_dir / Path(rel).with_suffix(".npz")
    record: dict[str, Any] = {
        "path": rel,
        "key": video_key(rel),
        "label": label_of(rel),
        "zip": zip_key,
    }
    tmp = tmpd / f"video{Path(rel).suffix}"
    t0 = time.perf_counter()
    try:
        record["sha256"] = hashlib.sha256(data).hexdigest()
        tmp.write_bytes(data)
        frame_out = (
            cfg.frame_cache_dir / Path(rel).with_suffix(".jpg") if cfg.frame_cache_dir else None
        )
        arrays, stats = extract_video(tmp, make_extractor, frame_out, cfg.extract_height)
        if stats["n_frames"] == 0:
            raise RuntimeError("decoded 0 frames")
        record.update(stats)
        save_npz(out, arrays, {**base_meta, **record})
        record["status"] = "ok"
        if stats["rate_any_hand"] < cfg.review_below_hand_rate:
            record["review"] = True
            with (cfg.out_dir / "_review.jsonl").open("a") as f:
                f.write(json.dumps({"path": rel, "rate_any_hand": stats["rate_any_hand"]}) + "\n")
    except Exception as e:  # one bad video must never stop the batch
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {e}"
    finally:
        tmp.unlink(missing_ok=True)
    record["seconds"] = round(time.perf_counter() - t0, 3)
    with (cfg.out_dir / "_log.jsonl").open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    log(
        f"  {record['status']:5s} {rel}"
        + (f"  ({record.get('error')})" if record["status"] == "error" else "")
    )
    return "review" if record.get("review") else str(record["status"])


def _count(counts: dict[str, int], status: str) -> None:
    counts["ok" if status == "review" else status] += 1
    if status == "review":
        counts["review"] += 1


def process_zip(
    zip_path: Path,
    zip_key: str,
    cfg: Config,
    make_extractor: Any,
    guard: DiskGuard,
    base_meta: dict[str, Any],
    log: Any,
) -> dict[str, int]:
    counts = {"ok": 0, "error": 0, "skipped": 0, "review": 0}
    with zipfile.ZipFile(zip_path) as zf, tempfile.TemporaryDirectory(dir=cfg.raw_dir) as tmpd:
        members = sorted(
            (i for i in zf.infolist() if i.filename.lower().endswith(VIDEO_EXTS)),
            key=lambda i: i.filename,
        )
        for info in members:
            if (cfg.out_dir / Path(info.filename).with_suffix(".npz")).exists():
                counts["skipped"] += 1
                continue
            guard.check(about_to_write=info.file_size)
            try:
                data = zf.read(info)  # CRC-checked; a corrupt member is a per-video error
            except Exception as e:
                data = b""
                log(f"  read error {info.filename}: {e!r}")
            _count(
                counts,
                process_video(
                    info.filename, data, zip_key, cfg, make_extractor, base_meta, Path(tmpd), log
                ),
            )
    return counts


def process_remote(
    zip_key: str,
    size: int,
    fetch: remote_zip.RangeFetcher,
    names: list[str],
    cfg: Config,
    make_extractor: Any,
    guard: DiskGuard,
    base_meta: dict[str, Any],
    log: Any,
    workers: int = 4,
) -> dict[str, int]:
    """Only `names` from a remote zip, each pulled by HTTP Range (CRC-checked).

    Zenodo caps each connection (~0.5-1 MB/s measured 2026-11-26) but not the total, so `workers`
    threads download the next videos in parallel, each through its own ZipFile, while this thread
    runs MediaPipe on them one at a time, in order. Network errors propagate (the run stops and
    resumes later from the .npz files already written)."""
    counts = {"ok": 0, "error": 0, "skipped": 0, "review": 0}
    todo = []
    for name in sorted(names):
        if (cfg.out_dir / Path(name).with_suffix(".npz")).exists():
            counts["skipped"] += 1
        else:
            todo.append(name)
    if not todo:
        return counts

    local = threading.local()

    def open_zip() -> zipfile.ZipFile:
        reader = io.BufferedReader(remote_zip.RangeReader(size, fetch), buffer_size=1 << 16)
        return zipfile.ZipFile(reader)

    def download(name: str) -> bytes:
        if not hasattr(local, "zf"):  # zipfile objects are not thread-safe: one per worker
            local.zf = open_zip()
        data: bytes = local.zf.read(name)
        return data

    index = open_zip()  # sizes for the disk guard
    pending: collections.deque[tuple[str, Future[bytes]]] = collections.deque()
    queue_names = iter(todo)
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"fetch-{zip_key}")

    def submit_next() -> None:
        name = next(queue_names, None)
        if name is not None:
            guard.check(about_to_write=index.getinfo(name).file_size * (len(pending) + 1))
            pending.append((name, pool.submit(download, name)))

    try:
        with tempfile.TemporaryDirectory(dir=cfg.raw_dir) as tmpd:
            for _ in range(workers + 1):
                submit_next()
            while pending:
                name, fut = pending.popleft()
                data = fut.result()  # re-raises the worker's network error
                submit_next()
                status = process_video(
                    name, data, zip_key, cfg, make_extractor, base_meta, Path(tmpd), log
                )
                _count(counts, status)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return counts


def dataset_members(dataset: str) -> dict[str, list[str]]:
    """zip key → member paths of every video in `dataset`'s official splits found on Zenodo."""
    from sanket.data.sessions import load_manifest
    from sanket.data.splits import SPLITS, load_split

    videos = {video_key(v.path): v for v in load_manifest(VIDEOS_MANIFEST)}
    out: dict[str, list[str]] = {}
    for split in SPLITS:
        for entry in load_split(dataset, split):  # type: ignore[arg-type]
            v = videos.get(video_key(entry))
            if v is not None:
                out.setdefault(v.zip, []).append(v.path)
    return {z: sorted(set(ms)) for z, ms in sorted(out.items())}


def run(
    cfg: Config,
    only_zips: list[str] | None = None,
    log: Any = print,
    allow_dirty: bool = False,
    only_dataset: str | None = None,
) -> None:
    sha = _git_sha()
    if sha.endswith("-dirty") and not allow_dirty:
        raise SystemExit(
            "working tree has uncommitted changes: commit first so this run is reproducible "
            "from its git SHA (or pass --allow-dirty for a throwaway dry run)"
        )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    guard = DiskGuard(cfg.raw_dir, min_free=int(cfg.min_free_gb * GB))
    guard.check()

    spec = load_spec()
    models = {p.name: fetch_model(p, CACHE_DIR / "models") for p in model_pins(spec)}
    cls = EXTRACTORS[cfg.extractor]

    def make_extractor() -> Extractor:
        return cls(models)

    import mediapipe

    base_meta = {
        "extractor": cfg.extractor,
        "spec_version": spec["spec_version"],
        "mediapipe_version": mediapipe.__version__
        if hasattr(mediapipe, "__version__")
        else spec["extractor"]["mediapipe_version"],
        "model_sha256": {p.name: p.sha256 for p in model_pins(spec)},
        "opencv_version": cv2.__version__,
        "git_sha": sha,
    }
    run_header = {
        **base_meta,
        "started_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "config": {k: str(v) for k, v in cfg.__dict__.items()},
        "only_dataset": only_dataset,
    }
    with (cfg.out_dir / "_runs.jsonl").open("a") as f:
        f.write(json.dumps(run_header, sort_keys=True) + "\n")

    files = [zenodo.RemoteFile(**f) for f in json.loads(FILES_MANIFEST.read_text())]
    if only_dataset:
        # Only this dataset's videos, each pulled from its zip by Range requests (no full zips).
        sizes = {f.key: f.size for f in files}
        wanted = dataset_members(only_dataset)
        total = sum(len(v) for v in wanted.values())
        log(f"{only_dataset}: {total} videos in {len(wanted)} zips (Range requests per video)")
        for i, (zkey, names) in enumerate(wanted.items(), 1):
            url = zenodo.file_url(zkey)
            fetch = lambda a, b, u=url: net.with_retries(  # noqa: E731
                lambda: net.get_range(u, a, b, timeout=600), attempts=40, log=log
            )
            counts = process_remote(
                zkey, sizes[zkey], fetch, names, cfg, make_extractor, guard, base_meta, log
            )
            log(f"[{i}/{len(wanted)}] {zkey}: {counts}")
        return

    zips = [
        f for f in files if f.key.endswith(".zip") and (only_zips is None or f.key in only_zips)
    ]
    done_file = cfg.out_dir / "_done_zips.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()

    for i, z in enumerate(zips, 1):
        if z.key in done:
            log(f"[{i}/{len(zips)}] {z.key}: already done")
            continue
        log(f"[{i}/{len(zips)}] {z.key}: downloading {z.size / GB:.2f} GB")
        path = download(z, cfg.raw_dir, guard, log=log)
        counts = process_zip(path, z.key, cfg, make_extractor, guard, base_meta, log)
        path.unlink()
        with done_file.open("a") as f:
            f.write(z.key + "\n")
        log(f"[{i}/{len(zips)}] {z.key}: {counts}; zip deleted")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "ml" / "configs" / "extract.yaml")
    ap.add_argument("--zips", nargs="*", help="only these zip names (dry runs)")
    ap.add_argument(
        "--only", choices=["include50"], help="only this dataset's videos (Range requests)"
    )
    ap.add_argument(
        "--allow-dirty", action="store_true", help="dry runs only: allow uncommitted changes"
    )
    args = ap.parse_args()
    run(Config.load(args.config), args.zips, allow_dirty=args.allow_dirty, only_dataset=args.only)


if __name__ == "__main__":
    main()
