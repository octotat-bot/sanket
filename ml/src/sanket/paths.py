"""Well-known repository paths, resolved from this file's location (no cwd dependence)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = REPO_ROOT / "spec" / "keypoints.json"
SPLITS_DIR = REPO_ROOT / "ml" / "data" / "splits"
REPORTS_DIR = REPO_ROOT / "reports"
CACHE_DIR = REPO_ROOT / "ml" / ".cache"
WEB_DIR = REPO_ROOT / "web"
