"""
FoundYou — Metric Unit Tests
==============================
Self-contained tests using only synthetic data.
No cameras, no YOLO, no GPU required.

Run with:
    python -m tests
    # or
    python benchmarks/tests.py
"""

from __future__ import annotations

import sys
import traceback

from metrics import (
    Detection, TrackingResult,
    compute_distmat, compute_cmc, rank_k_accuracy, mean_ap,
    mota, motp, idf1, fragmentation,
    gallery_dist_stats, resolve_step_breakdown, merge_effectiveness, throughput,
)
from data import generate_synthetic_sequence
from ablation import run_ablation

# ================================================================
# Test harness
# ================================================================

_passed = 0
_failed = 0


def _test(name: str):
    def decorator(fn):
        def wrapper():
            global _passed, _failed
            try:
                fn()
                print(f"  ✓  {name}")
                _passed += 1
            except Exception as e:
                print(f"  ✗  {name}")
                traceback.print_exc()
                _failed += 1
        return wrapper
    return decorator


def _assert(cond: bool, msg: str = ''):
    if not cond:
        raise AssertionError(msg or 'Assertion failed')


def _assert_close(a: float, b: float, tol: float = 1e-3, msg: str = ''):
    if abs(a - b) > tol:
        raise AssertionError(msg or f"{a} not close to {b} (tol={tol})")


# ================================================================
# Perfect prediction helpers
# ================================================================

def _perfect_pair(n_people=3, n_cams=2, n_frames=50):
    """GT and prediction are identical — all metrics should be near-perfect."""
    gt, _ = generate_synthetic_sequence(
        n_people=n_people, n_cameras=n_cams, n_frames=n_frames,
        n_id_switches=0, n_fragmentations=0, seed=1,
    )
    # Build perfect predictions: pred_id == true_id, same bboxes
    pr = [Detection(d.frame, d.cam, 0, d.true_id, d.bbox) for d in gt]
    return gt, pr


# ================================================================
# MOTA tests
# ================================================================

@_test("mota: perfect prediction → mota ≈ 1.0")
def test_mota_perfect():
    gt, pr = _perfect_pair()
    r = mota(gt, pr)
    _assert_close(r['mota'], 1.0, tol=0.01)
    _assert(r['fp'] == 0, f"Expected 0 FP, got {r['fp']}")
    _assert(r['fn'] == 0, f"Expected 0 FN, got {r['fn']}")
    _assert(r['id_sw'] == 0, f"Expected 0 IDS, got {r['id_sw']}")


@_test("mota: all FP (predictions with no GT) → mota < 0")
def test_mota_all_fp():
    gt, _ = generate_synthetic_sequence(n_people=2, n_cameras=1, n_frames=20, seed=2)
    # Shift all prediction bboxes far away so IoU = 0
    pr = [Detection(d.frame, d.cam, 0, d.true_id, (1000, 1000, 1060, 1120))
          for d in gt]
    r = mota(gt, pr)
    _assert(r['mota'] < 0.5, f"Expected low MOTA, got {r['mota']}")


@_test("mota: injected id_switches are counted")
def test_mota_id_switches():
    gt, pr = generate_synthetic_sequence(
        n_people=3, n_cameras=1, n_frames=100,
        n_id_switches=5, n_fragmentations=0, seed=3,
    )
    r = mota(gt, pr)
    _assert(r['id_sw'] > 0, "Expected > 0 ID switches")


@_test("mota: injected fragmentations reduce fn count")
def test_mota_fn():
    gt, pr = generate_synthetic_sequence(
        n_people=3, n_cameras=1, n_frames=100,
        n_id_switches=0, n_fragmentations=5, seed=4,
    )
    r = mota(gt, pr)
    _assert(r['fn'] > 0, "Expected FN > 0 due to injected fragmentations")


# ================================================================
# MOTP tests
# ================================================================

@_test("motp: perfect bboxes → motp ≈ 1.0")
def test_motp_perfect():
    gt, pr = _perfect_pair()
    r = motp(gt, pr)
    _assert_close(r['motp'], 1.0, tol=0.01)


@_test("motp: no matches → matched_count == 0")
def test_motp_no_matches():
    gt, _ = generate_synthetic_sequence(n_people=2, n_cameras=1, n_frames=10, seed=5)
    pr = [Detection(d.frame, d.cam, 0, d.true_id, (2000, 2000, 2060, 2120))
          for d in gt]
    r = motp(gt, pr)
    _assert(r['matched_count'] == 0)


# ================================================================
# IDF1 tests
# ================================================================

@_test("idf1: perfect prediction → idf1 ≈ 1.0")
def test_idf1_perfect():
    gt, pr = _perfect_pair()
    r = idf1(gt, pr)
    _assert_close(r['idf1'], 1.0, tol=0.05)
    _assert_close(r['idp'],  1.0, tol=0.05)
    _assert_close(r['idr'],  1.0, tol=0.05)


@_test("idf1: all wrong IDs → idf1 ≈ 0")
def test_idf1_wrong_ids():
    gt, pr = _perfect_pair(n_people=3, n_frames=30)
    # Shift all predicted IDs by 1000 so none match GT
    pr = [Detection(d.frame, d.cam, 0, d.pred_id + 1000, d.bbox) for d in pr]
    r = idf1(gt, pr)
    _assert(r['idf1'] < 0.1, f"Expected IDF1 near 0, got {r['idf1']}")


@_test("idf1: idf1 = 2*idp*idr / (idp+idr)")
def test_idf1_formula():
    gt, pr = generate_synthetic_sequence(
        n_people=3, n_cameras=2, n_frames=60,
        n_id_switches=2, n_fragmentations=2, seed=6,
    )
    r = idf1(gt, pr)
    idp, idr = r['idp'], r['idr']
    expected = 2 * idp * idr / max(idp + idr, 1e-9)
    _assert_close(r['idf1'], expected, tol=0.01)


# ================================================================
# Fragmentation tests
# ================================================================

@_test("fragmentation: perfect → 0 fragmentations")
def test_frag_perfect():
    gt, pr = _perfect_pair()
    _assert(fragmentation(gt, pr) == 0)


@_test("fragmentation: injected drops are detected")
def test_frag_injected():
    n_inject = 4
    gt, pr = generate_synthetic_sequence(
        n_people=3, n_cameras=1, n_frames=150,
        n_id_switches=0, n_fragmentations=n_inject, seed=7,
    )
    f = fragmentation(gt, pr)
    _assert(f > 0, f"Expected > 0 fragmentations with {n_inject} injected")


# ================================================================
# CMC / mAP tests
# ================================================================

@_test("compute_distmat: diagonal is 0 for identical features")
def test_distmat_diagonal():
    import numpy as np
    feats = np.random.randn(5, 64).astype(np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    d = compute_distmat(feats, feats)
    _assert(np.allclose(np.diag(d), 0, atol=1e-5),
            f"Diagonal not zero: {np.diag(d)}")


@_test("rank_k_accuracy: trivial perfect retrieval → rank1 = 1.0")
def test_rank1_perfect():
    import numpy as np
    # Each identity has one query and one gallery sample; they are identical
    n = 4
    feats   = np.eye(n, dtype=np.float32)
    pids    = np.arange(n)
    camids  = np.zeros(n, dtype=int)
    distmat = compute_distmat(feats, feats)
    cmc     = compute_cmc(distmat, pids, pids, camids, camids,
                          cross_cam=False)
    _assert_close(rank_k_accuracy(cmc, k=1), 1.0)


@_test("mean_ap: perfect retrieval → mAP = 1.0")
def test_map_perfect():
    import numpy as np
    n = 4
    feats   = np.eye(n, dtype=np.float32)
    pids    = np.arange(n)
    camids  = np.zeros(n, dtype=int)
    distmat = compute_distmat(feats, feats)
    ap      = mean_ap(distmat, pids, pids, camids, camids, cross_cam=False)
    _assert_close(ap, 1.0)


# ================================================================
# Gallery quality tests
# ================================================================

@_test("gallery_dist_stats: returns expected keys")
def test_gallery_stats_keys():
    from collections import defaultdict

    import numpy as np

    # Build two minimal synthetic GlobalTrack objects
    class FakeTrack:
        def __init__(self, gid, vecs):
            self.gid     = gid
            self.feature = vecs.mean(0)
            self.feature /= np.linalg.norm(self.feature)
            self._buf    = defaultdict(list)
            self._buf[0] = list(vecs)

    vecs_a = np.random.randn(6, 64).astype(np.float32)
    vecs_a /= np.linalg.norm(vecs_a, axis=1, keepdims=True)
    vecs_b = np.random.randn(6, 64).astype(np.float32)
    vecs_b /= np.linalg.norm(vecs_b, axis=1, keepdims=True)

    tracks = [FakeTrack(1, vecs_a), FakeTrack(2, vecs_b)]
    stats  = gallery_dist_stats(tracks)

    for key in ['intra_mean', 'intra_std', 'inter_mean',
                'inter_std', 'separation_ratio', 'n_tracks']:
        _assert(key in stats, f"Missing key: {key}")

    _assert(stats['n_tracks'] == 2)
    _assert(stats['separation_ratio'] >= 0)


# ================================================================
# Pipeline metrics tests
# ================================================================

@_test("resolve_step_breakdown: fractions sum to 1.0")
def test_step_fractions():
    r = TrackingResult(step_a=30, step_b=10, step_c=60)
    bd = resolve_step_breakdown(r)
    total = bd['step_a'] + bd['step_b'] + bd['step_c']
    _assert_close(total, 1.0)


@_test("resolve_step_breakdown: all zeros → all zeros")
def test_step_all_zero():
    r  = TrackingResult()
    bd = resolve_step_breakdown(r)
    _assert(bd['total_resolves'] == 0)


@_test("merge_effectiveness: correction_rate capped at 1.0")
def test_merge_correction_rate():
    r = TrackingResult(merges=100)
    m = merge_effectiveness(r, gt_frag=10)
    _assert(m['correction_rate'] <= 1.0)
    _assert(m['over_merge_rate'] > 0)


@_test("throughput: returns positive dets_per_sec")
def test_throughput():
    r = TrackingResult(total_time=2.0)
    t = throughput(r, total_detections=1000)
    _assert(t['dets_per_sec'] == 500.0, f"Got {t['dets_per_sec']}")


# ================================================================
# Ablation smoke test
# ================================================================

@_test("run_ablation: returns one row per variant")
def test_ablation_rows():
    gt, pr = generate_synthetic_sequence(
        n_people=2, n_cameras=1, n_frames=30, seed=8)
    variants = [
        {'name': 'v1', 'CROSS_CAM_DIST': 0.40},
        {'name': 'v2', 'CROSS_CAM_DIST': 0.30},
    ]
    results = run_ablation(gt, [pr, pr], variants)
    _assert(len(results) == 2)
    _assert(results[0]['name'] == 'v1')
    _assert(results[1]['name'] == 'v2')


# ================================================================
# Runner
# ================================================================

def run_all():
    print("\n" + "=" * 50)
    print("  FoundYou metric unit tests")
    print("=" * 50)

    # Execute all decorated test functions
    for name, obj in list(globals().items()):
        if callable(obj) and name.startswith('test_'):
            obj()

    print("=" * 50)
    print(f"  {_passed} passed  |  {_failed} failed")
    print("=" * 50 + "\n")
    return _failed == 0


if __name__ == '__main__':
    ok = run_all()
    sys.exit(0 if ok else 1)
