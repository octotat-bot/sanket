"""Load `spec/keypoints.json` and fetch the pinned MediaPipe model files.

The spec is the single source of truth shared by Python and the web app. In Phase 0 it
only pins the keypoint extractor (library version + exact model bytes by sha256).
"""

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sanket import net
from sanket.paths import CACHE_DIR, SPEC_PATH, WEB_DIR


@dataclass(frozen=True)
class ModelPin:
    name: str
    url: str
    sha256: str
    bytes: int


def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    spec: dict[str, Any] = json.loads(path.read_text())
    return spec


def model_pins(spec: dict[str, Any]) -> list[ModelPin]:
    models = spec["extractor"]["models"]
    return [
        ModelPin(name=k, url=v["url"], sha256=v["sha256"], bytes=v["bytes"])
        for k, v in models.items()
    ]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_model(pin: ModelPin, dest_dir: Path) -> Path:
    """Download `pin` into dest_dir unless a file with the right sha256 is already there.

    Raises ValueError on a hash mismatch: we never silently use different model bytes,
    because train/serve skew starts exactly there.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{pin.name}.task"
    if dest.exists() and sha256_of(dest) == pin.sha256:
        return dest
    data = net.get_bytes(pin.url, timeout=300)
    got = hashlib.sha256(data).hexdigest()
    if got != pin.sha256:
        raise ValueError(f"{pin.name}: sha256 mismatch (expected {pin.sha256}, got {got})")
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch pinned MediaPipe models from spec/keypoints.json"
    )
    parser.add_argument("--dest", type=Path, default=CACHE_DIR / "models")
    parser.add_argument(
        "--copy-to-web",
        action="store_true",
        help="also copy to web/public/models/mediapipe so the browser loads identical bytes",
    )
    args = parser.parse_args()
    for pin in model_pins(load_spec()):
        path = fetch_model(pin, args.dest)
        print(f"ok  {pin.name}  {pin.sha256[:12]}  {path}")
        if args.copy_to_web:
            web_dest = WEB_DIR / "public" / "models" / "mediapipe"
            web_dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, web_dest / path.name)


if __name__ == "__main__":
    main()
