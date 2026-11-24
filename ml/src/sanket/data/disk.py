"""Free-disk guard: extraction aborts before free space drops below a floor (Mukund: 10 GB).

The laptop's free space also moves for reasons outside our control (macOS swap grew by
14 GB on 2026-11-24), so the guard checks *actual* free space often, not just once at start.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

GB = 1_000_000_000


class LowDiskSpace(RuntimeError):
    pass


def free_bytes(path: Path) -> int:
    """Free bytes on the volume holding `path` (nearest existing parent)."""
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


@dataclass(frozen=True)
class DiskGuard:
    path: Path
    min_free: int = 10 * GB

    def check(self, about_to_write: int = 0) -> int:
        """Raise LowDiskSpace if free space, minus what we're about to write, is under the floor."""
        free = free_bytes(self.path)
        if free - about_to_write < self.min_free:
            raise LowDiskSpace(
                f"free {free / GB:.1f} GB - planned {about_to_write / GB:.1f} GB "
                f"< floor {self.min_free / GB:.0f} GB on {self.path}"
            )
        return free
