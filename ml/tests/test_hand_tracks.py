"""Crafted cases for left/right label metrics (CLAUDE.md section 8: crafted hand-swap tests)."""

import math

from sanket.keypoints.hand_tracks import Frame, body_agreement, count_swaps, detection_rates

# The person's left wrist appears on the image RIGHT (camera faces the signer).
POSE = {"pose_left": (0.8, 0.5), "pose_right": (0.2, 0.5)}


def test_body_agreement_correct_and_inverted_labels() -> None:
    correct = [Frame({"left": (0.8, 0.5), "right": (0.2, 0.5)}, **POSE)]
    inverted = [Frame({"left": (0.2, 0.5), "right": (0.8, 0.5)}, **POSE)]
    assert body_agreement(correct).rate == 1.0
    assert body_agreement(inverted).rate == 0.0
    assert body_agreement(correct).judged == 2


def test_body_agreement_ignores_ambiguous_midline_hands_and_missing_pose() -> None:
    midline = [Frame({"left": (0.5, 0.5)}, **POSE)]
    no_pose = [Frame({"left": (0.8, 0.5)})]
    assert body_agreement(midline).judged == 0
    assert body_agreement(no_pose).judged == 0
    assert math.isnan(body_agreement(no_pose).rate)


def test_swap_detected_when_two_hands_exchange_labels_without_moving() -> None:
    a = Frame({"left": (0.7, 0.5), "right": (0.3, 0.5)})
    swapped = Frame({"left": (0.3, 0.5), "right": (0.7, 0.5)})
    assert count_swaps([a, a, swapped, swapped]).swaps == 1
    assert count_swaps([a, a, swapped, swapped]).transitions == 3


def test_hands_crossing_slowly_is_not_a_swap() -> None:
    # Each hand moves 0.05 per frame toward the other; labels follow the hands.
    frames = [
        Frame({"left": (0.7 - 0.05 * i, 0.5), "right": (0.3 + 0.05 * i, 0.5)}) for i in range(9)
    ]
    assert count_swaps(frames).swaps == 0


def test_single_hand_relabelled_in_place_is_a_swap_but_a_new_hand_is_not() -> None:
    still = [Frame({"left": (0.5, 0.5)}), Frame({"right": (0.51, 0.5)})]
    jumped = [Frame({"left": (0.2, 0.5)}), Frame({"right": (0.8, 0.5)})]  # other hand came in
    assert count_swaps(still).swaps == 1
    assert count_swaps(jumped).transitions == 0


def test_detection_rates() -> None:
    frames = [
        Frame({"left": (0.1, 0.1), "right": (0.2, 0.2)}, **POSE),
        Frame({"right": (0.2, 0.2)}),
        Frame({}),
        Frame({}),
    ]
    r = detection_rates(frames)
    assert r == {"any_hand": 0.5, "both_hands": 0.25, "left": 0.25, "right": 0.5, "pose": 0.25}
