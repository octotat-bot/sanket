"""Upload the extracted keypoints as a PRIVATE Kaggle dataset (Phase 1).

    make upload KAGGLE_DATASET=<kaggle-user>/sanket-include-keypoints-v1

Credentials: the Kaggle CLI reads ~/.kaggle/access_token or $KAGGLE_API_TOKEN (never the repo).
The upload contains only the .npz keypoints and their logs. No videos and no cached frames:
the frame cache lives in ml/.cache/frames, outside the uploaded folder.
"""

import argparse
import json
import subprocess
from pathlib import Path

from sanket.paths import REPO_ROOT
from sanket.pipeline.extract import FILES_MANIFEST, Config

N_ZIPS = sum(1 for f in json.loads(FILES_MANIFEST.read_text()) if f["key"].endswith(".zip"))


def check_complete(out_dir: Path, only: str | None = None) -> None:
    """Full run: all zips done. `only="include50"`: every split video of that dataset has a .npz."""
    if only:
        from sanket.data.keypoint_index import load_split_index
        from sanket.data.splits import SPLITS

        missing = [m for s in SPLITS for m in load_split_index(only, s, out_dir).missing]  # type: ignore[arg-type]
        if missing:
            raise SystemExit(
                f"extraction incomplete: {len(missing)} {only} videos missing, e.g. {missing[:3]}"
            )
    else:
        done_file = out_dir / "_done_zips.txt"
        done = set(done_file.read_text().split()) if done_file.exists() else set()
        if len(done) != N_ZIPS:
            raise SystemExit(f"extraction incomplete: {len(done)}/{N_ZIPS} zips done in {out_dir}")
    if any(out_dir.rglob("*.jpg")) or any(out_dir.rglob("*.MOV")) or any(out_dir.rglob("*.mov")):
        raise SystemExit(f"{out_dir} contains images or videos; refusing to upload")


def metadata(dataset_id: str, extractor: str) -> dict[str, object]:
    return {
        "title": "Sanket INCLUDE keypoints v1",  # INCLUDE-50 subset when extracted with --only
        "id": dataset_id,
        "licenses": [{"name": "CC-BY-4.0"}],
        "subtitle": f"MediaPipe 1.2.3 {extractor} keypoints extracted from INCLUDE (ISL)",
        "description": (
            "Per-video MediaPipe keypoints (no video, no images) extracted from the INCLUDE Indian "
            "Sign Language dataset by Sridhar, Ganesan, Kumar and Khapra (AI4Bharat / IIT Madras, "
            "ACM MM 2020, doi:10.1145/3394171.3413528), Zenodo record 4010759, CC BY 4.0. "
            "Layout: spec/keypoints.json in the Sanket repository."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True, help="<kaggle-username>/<slug>")
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "ml" / "configs" / "extract.yaml")
    ap.add_argument("--only", choices=["include50"], help="dataset subset that was extracted")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    check_complete(cfg.out_dir, args.only)
    (cfg.out_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata(args.dataset, cfg.extractor), indent=2)
    )
    cmd = [
        "kaggle",
        "datasets",
        "create",
        "-p",
        str(cfg.out_dir),
        "--dir-mode",
        "zip",
    ]  # private by default
    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
