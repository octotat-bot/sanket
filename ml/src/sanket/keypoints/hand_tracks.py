"""Left/right hand label quality metrics (used by ADR 002, and later by swap correction).

All positions are wrist points in *isotropic* image units: x is multiplied by width/height so
a step of 0.1 means the same distance horizontally and vertically (1.0 = image height).

Two independent ways to catch a wrong left/right label:

1. **Body agreement.** The pose model also predicts the person's left wrist (pose landmark 15)
   and right wrist (16). A hand labelled "left" should be nearest the pose's left wrist. We
   only judge hands that are *clearly* nearer one pose wrist (`clear_ratio`), so hands close
   to the body midline or crossing over don't count either way.
2. **Temporal swaps.** Between two consecutive frames where both hands are present, the labels
   swapped if pairing each hand with the *other* label's previous position is much cheaper
   than keeping labels (`swap_ratio`). With one hand in both frames, a label change counts as
   a swap if the wrist barely moved (`still_dist`): the same hand was relabelled.
"""

import itertools
import math
from dataclasses import dataclass

Point = tuple[float, float]
SIDES = ("left", "right")


@dataclass(frozen=True)
class Frame:
    hands: dict[str, Point]  # label -> wrist; a missing key means not detected
    pose_left: Point | None = None  # pose landmark 15 (the person's left wrist), if visible
    pose_right: Point | None = None  # pose landmark 16


def _d(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass(frozen=True)
class BodyAgreement:
    judged: int  # hands clearly nearer one pose wrist
    agree: int  # ... whose label matches that pose wrist

    @property
    def rate(self) -> float:
        return self.agree / self.judged if self.judged else float("nan")


def body_agreement(frames: list[Frame], clear_ratio: float = 0.6) -> BodyAgreement:
    """Count hands whose label matches the nearer pose wrist.

    A hand is judged only if  d(near) < clear_ratio * d(far).
    """
    judged = agree = 0
    for f in frames:
        if f.pose_left is None or f.pose_right is None:
            continue
        for label, wrist in f.hands.items():
            dl, dr = _d(wrist, f.pose_left), _d(wrist, f.pose_right)
            near, far, side = (dl, dr, "left") if dl < dr else (dr, dl, "right")
            if near < clear_ratio * far:
                judged += 1
                agree += side == label
    return BodyAgreement(judged, agree)


@dataclass(frozen=True)
class SwapCount:
    transitions: int  # consecutive frame pairs where a swap could be judged
    swaps: int

    @property
    def per_100(self) -> float:
        return 100 * self.swaps / self.transitions if self.transitions else float("nan")


def count_swaps(
    frames: list[Frame], swap_ratio: float = 0.5, still_dist: float = 0.05
) -> SwapCount:
    transitions = swaps = 0
    for prev, cur in itertools.pairwise(frames):
        p, c = prev.hands, cur.hands
        if len(p) == 2 and len(c) == 2:
            transitions += 1
            keep = _d(p["left"], c["left"]) + _d(p["right"], c["right"])
            cross = _d(p["left"], c["right"]) + _d(p["right"], c["left"])
            swaps += cross < swap_ratio * keep
        elif len(p) == 1 and len(c) == 1:
            (pl, pw), (cl, cw) = next(iter(p.items())), next(iter(c.items()))
            if _d(pw, cw) < still_dist:  # same physical hand (it barely moved)
                transitions += 1
                swaps += pl != cl
    return SwapCount(transitions, swaps)


def detection_rates(frames: list[Frame]) -> dict[str, float]:
    n = len(frames) or 1
    return {
        "any_hand": sum(bool(f.hands) for f in frames) / n,
        "both_hands": sum(len(f.hands) == 2 for f in frames) / n,
        "left": sum("left" in f.hands for f in frames) / n,
        "right": sum("right" in f.hands for f in frames) / n,
        "pose": sum(f.pose_left is not None and f.pose_right is not None for f in frames) / n,
    }
