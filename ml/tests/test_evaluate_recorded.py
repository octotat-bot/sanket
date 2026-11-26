import numpy as np
from test_recorded import session  # same synthetic session as the converter tests

from sanket.data.recorded import clip_arrays
from sanket.evaluate_recorded import VARIANTS, score, trim_to_hands
from sanket.keypoints.features import FeatureSpec

LABELS = [f"w{i}" for i in range(50)]
CAL = {"temperature": 1.0, "min_prob": 0.45, "min_margin": 0.15}


def clip(hand_frames: slice | None = None, n: int = 30) -> dict[str, np.ndarray]:
    c = clip_arrays(session(n)["clips"][0])
    c.pop("label_id")
    if hand_frames is not None:
        c["left_present"][:] = False
        c["left_present"][hand_frames] = True
    return c


def test_trim_keeps_the_hand_span_plus_margin_and_restarts_time() -> None:
    t = trim_to_hands(clip(slice(10, 20)))
    assert len(t["t_ms"]) == (20 + 5) - (10 - 5) and t["t_ms"][0] == 0
    assert t["pose"].shape[0] == len(t["t_ms"]) and t["frame_size"].tolist() == [640, 480]
    assert len(trim_to_hands(clip(slice(0, 0)))["t_ms"]) == 30  # no hands: unchanged


def test_mirrored_variant_swaps_hands() -> None:
    m = VARIANTS["mirrored"](clip())
    assert m["right_present"].all() and not m["left_present"].any()


def test_score_counts_accuracy_rejection_and_no_sign() -> None:
    fs = FeatureSpec.from_spec()
    items = [(clip(), {"kind": "sign", "label_id": 22}), (clip(), {"kind": "sign", "label_id": 3})]
    items.append((clip(), {"kind": "no_sign", "label_id": -1}))

    def confident_hello(x: np.ndarray) -> np.ndarray:
        z = np.zeros((len(x), 50), np.float32)
        z[:, 22] = 10
        return z

    r = score(items, confident_hello, fs, 16, LABELS, CAL, VARIANTS["recorded"])
    assert r["n_sign"] == 2 and r["top1"] == 0.5 and r["accepted_rate"] == 1.0
    assert r["no_sign_rejected_rate"] == 0.0 and r["per_word"] == {"w22": "1/1", "w3": "0/1"}

    def unsure(x: np.ndarray) -> np.ndarray:
        return np.zeros((len(x), 50), np.float32)

    r = score(items, unsure, fs, 16, LABELS, CAL, VARIANTS["recorded"])
    assert r["accepted_rate"] == 0.0 and r["no_sign_rejected_rate"] == 1.0
