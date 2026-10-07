"""
FoundYou — Feature Utilities
==============================
Pure numpy distance helpers used by tracks.py and reid.py.
No circular imports: nothing here imports tracks or reid.

TYPE_CHECKING guard is used for the GlobalTrack type hint so that
tracks.py can import features.py without creating a cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tracks import GlobalTrack


# ======================== FEATURE UTILS =========================
def _norm(v):
    """Return unit-norm copy of v, or None on bad input."""
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-6 or not np.isfinite(n):
        return None
    return v / n


def _dist(a, b) -> float:
    """Cosine distance in [0, 2]. 0 = identical, 1 = orthogonal."""
    if a is None or b is None:
        return 2.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 2.0
    return float(1.0 - np.clip(np.dot(a/na, b/nb), -1, 1))


def _robust_dist_to_gallery(feat, gt: 'GlobalTrack') -> float:
    """
    FP-1 fix: Robust distance from feat to gt's gallery.

    Algorithm:
      For each camera that has stored frames:
        - Compute cosine distance from feat to every stored frame.
        - Take the MEDIAN (robust to single outlier/blurred frames).
      Average the per-camera medians.
      Also compute distance to the merged gallery mean.
      Return the average of (mean-cam-median, merged-mean-dist).

    Why:
      - Median per camera kills the effect of a single noisy frame that
        happens to be close to any incoming query — which is exactly what
        the previous min() was vulnerable to.
      - Averaging camera medians prevents one well-lit camera from dominating.
      - Including the merged mean adds noise-averaged stability.
      - The resulting score is conservative (higher than min), making the
        threshold meaningful again.
    """
    if feat is None:
        return 2.0

    cam_medians = []
    for cam_buf in gt._buf.values():
        if not cam_buf:
            continue
        dists = [_dist(feat, v) for v in cam_buf]
        cam_medians.append(float(np.median(dists)))

    merged_dist = _dist(feat, gt.feature)   # 2.0 if gt.feature is None

    if not cam_medians:
        return merged_dist

    mean_cam_median = float(np.mean(cam_medians))

    if gt.feature is None:
        return mean_cam_median

    return (mean_cam_median + merged_dist) / 2.0