"""
FoundYou — Benchmarking Metrics
=================================
All metric computations used for evaluating ReID and multi-camera
tracking quality. Pure numpy — no dependency on the live system.

Metrics implemented
-------------------
  ReID (ranking)
    rank_k_accuracy   Rank-1 / Rank-5 / Rank-10 accuracy
    mean_ap           mean Average Precision (mAP)
    compute_cmc       full Cumulative Matching Characteristic curve

  Multi-camera tracking
    idf1              IDF1  — identity-aware F1 over matched detections
    mota              MOTA  — Multi-Object Tracking Accuracy
    motp              MOTP  — Multi-Object Tracking Precision
    id_switches       raw ID-switch count
    fragmentation     raw fragmentation count (track interruptions)

  Gallery quality
    gallery_dist_stats  per-track intra/inter-gallery distance summary
    probe_convergence   how quickly gallery distance stabilises vs probe count

  System
    resolve_step_breakdown   fraction of resolves via step A / B / C
    merge_effectiveness      merges fired vs fragmentation prevented
    throughput               detections processed per second

Usage
-----
  All functions accept plain Python lists or numpy arrays.
  Ground-truth format is described per function.

  Quick end-to-end example:

    from metrics import rank_k_accuracy, mean_ap, idf1
    from data    import load_ground_truth, build_result_sequence

    gt   = load_ground_truth("path/to/annotations.csv")
    res  = build_result_sequence("foundyou.pkl")
    cmc  = compute_cmc(embeddings, labels, query_idx, gallery_idx)
    print(rank_k_accuracy(cmc, k=1))
    print(idf1(gt, res))
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


# ================================================================
# Shared types
# ================================================================

@dataclass
class Detection:
    """One detection as seen by the ground-truth annotator or FoundYou."""
    frame    : int        # frame index (0-based)
    cam      : int        # camera id
    true_id  : int        # ground-truth person ID  (0 = unknown / ignored)
    pred_id  : int        # predicted global ID assigned by FoundYou
    bbox     : Tuple[int, int, int, int]  # x1 y1 x2 y2


@dataclass
class TrackingResult:
    """Accumulated per-sequence results fed to MOTA/MOTP/IDF1."""
    detections : List[Detection] = field(default_factory=list)
    # resolve step histogram  — filled by ReIDManager instrumentation
    step_a     : int = 0   # matched active track
    step_b     : int = 0   # reacquired archived track
    step_c     : int = 0   # new track created
    merges     : int = 0
    total_time : float = 0.0   # wall seconds of the run


# ================================================================
# ReID ranking metrics
# ================================================================

def compute_distmat(query_feats: np.ndarray,
                    gallery_feats: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine distance matrix.

    Parameters
    ----------
    query_feats   : (Q, D) float32 — L2-normalised query embeddings
    gallery_feats : (G, D) float32 — L2-normalised gallery embeddings

    Returns
    -------
    distmat : (Q, G) float32   cosine distance in [0, 2]
    """
    q = query_feats  / (np.linalg.norm(query_feats,   axis=1, keepdims=True) + 1e-12)
    g = gallery_feats / (np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-12)
    return 1.0 - (q @ g.T).astype(np.float32)


def compute_cmc(distmat    : np.ndarray,
                q_pids     : np.ndarray,
                g_pids     : np.ndarray,
                q_camids   : np.ndarray,
                g_camids   : np.ndarray,
                max_rank   : int = 50,
                cross_cam  : bool = True) -> np.ndarray:
    """
    Cumulative Matching Characteristic (CMC) curve.

    Follows the standard Market-1501 evaluation protocol:
      - For each query, remove gallery samples from the same camera
        with the same identity (junk removal) when cross_cam=True.
      - A match is correct if the gallery identity equals the query identity.

    Parameters
    ----------
    distmat   : (Q, G) cosine distance matrix
    q_pids    : (Q,)   query person IDs
    g_pids    : (G,)   gallery person IDs
    q_camids  : (Q,)   query camera IDs
    g_camids  : (G,)   gallery camera IDs
    max_rank  : int    maximum rank to compute
    cross_cam : bool   if True, apply same-cam junk removal

    Returns
    -------
    cmc : (max_rank,) float   CMC[k] = fraction of queries with a correct
                               match in top-k gallery results (0.0 – 1.0)
    """
    num_q, num_g = distmat.shape
    max_rank = min(max_rank, num_g)

    all_cmc   = np.zeros((num_q, max_rank), dtype=np.float32)
    all_ap    = np.zeros(num_q,             dtype=np.float32)
    valid_q   = 0

    for q_idx in range(num_q):
        q_pid   = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = np.argsort(distmat[q_idx])

        # Mask gallery items: same person AND same camera → junk
        if cross_cam:
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        else:
            remove = np.zeros(num_g, dtype=bool)

        keep = ~remove
        sorted_pids = g_pids[order][keep]

        matches   = (sorted_pids == q_pid).astype(np.float32)
        num_valid = matches.sum()

        if num_valid == 0:
            # No valid gallery match for this query — skip
            continue

        valid_q += 1

        # CMC
        cmc = matches[:max_rank].cumsum()
        cmc[cmc > 1] = 1
        all_cmc[q_idx] = cmc

        # AP
        num_rel   = 0
        tmp_cmc   = 0.0
        for i, m in enumerate(matches):
            if m:
                num_rel  += 1
                tmp_cmc  += num_rel / (i + 1)
        all_ap[q_idx] = tmp_cmc / num_valid if num_valid else 0.0

    if valid_q == 0:
        return np.zeros(max_rank, dtype=np.float32)

    return all_cmc[:valid_q].mean(0)


def rank_k_accuracy(cmc: np.ndarray, k: int = 1) -> float:
    """
    Return Rank-k accuracy from a CMC curve.

    Parameters
    ----------
    cmc : (max_rank,) CMC curve from compute_cmc()
    k   : rank to query (1-based)

    Returns
    -------
    float in [0, 1]
    """
    if k < 1 or k > len(cmc):
        raise ValueError(f"k={k} out of range [1, {len(cmc)}]")
    return float(cmc[k - 1])


def mean_ap(distmat  : np.ndarray,
            q_pids   : np.ndarray,
            g_pids   : np.ndarray,
            q_camids : np.ndarray,
            g_camids : np.ndarray,
            cross_cam: bool = True) -> float:
    """
    Mean Average Precision (mAP) over all queries.

    Uses the same junk-removal logic as compute_cmc.

    Returns
    -------
    float in [0, 1]
    """
    num_q = distmat.shape[0]
    aps   = []

    for q_idx in range(num_q):
        q_pid   = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order   = np.argsort(distmat[q_idx])

        if cross_cam:
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        else:
            remove = np.zeros(distmat.shape[1], dtype=bool)

        keep        = ~remove
        sorted_pids = g_pids[order][keep]
        matches     = (sorted_pids == q_pid).astype(np.float32)
        num_valid   = matches.sum()

        if num_valid == 0:
            continue

        num_rel = 0
        ap      = 0.0
        for i, m in enumerate(matches):
            if m:
                num_rel += 1
                ap      += num_rel / (i + 1)
        aps.append(ap / num_valid)

    return float(np.mean(aps)) if aps else 0.0


# ================================================================
# Multi-object tracking metrics
# ================================================================

def _iou(box_a: Tuple, box_b: Tuple) -> float:
    """IoU between two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _match_detections(gt_dets: List[Detection],
                      pr_dets: List[Detection],
                      iou_thr: float = 0.5
                      ) -> Tuple[List[Tuple], List[int], List[int]]:
    """
    Greedy IoU matching between GT and predicted detections on a single frame.

    Returns
    -------
    matched   : list of (gt_idx, pr_idx) pairs
    unmatched_gt : indices into gt_dets with no prediction
    unmatched_pr : indices into pr_dets with no ground truth
    """
    if not gt_dets or not pr_dets:
        return [], list(range(len(gt_dets))), list(range(len(pr_dets)))

    iou_mat = np.zeros((len(gt_dets), len(pr_dets)), dtype=np.float32)
    for i, g in enumerate(gt_dets):
        for j, p in enumerate(pr_dets):
            iou_mat[i, j] = _iou(g.bbox, p.bbox)

    matched       = []
    used_gt       = set()
    used_pr       = set()

    # Greedy: highest IoU first
    for _ in range(min(len(gt_dets), len(pr_dets))):
        idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        gi, pi = idx
        if iou_mat[gi, pi] < iou_thr:
            break
        matched.append((gi, pi))
        used_gt.add(gi); used_pr.add(pi)
        iou_mat[gi, :] = -1
        iou_mat[:, pi] = -1

    unmatched_gt = [i for i in range(len(gt_dets)) if i not in used_gt]
    unmatched_pr = [j for j in range(len(pr_dets)) if j not in used_pr]
    return matched, unmatched_gt, unmatched_pr


def mota(gt_detections : List[Detection],
         pr_detections : List[Detection],
         iou_thr       : float = 0.5) -> Dict:
    """
    Multi-Object Tracking Accuracy (MOTA).

        MOTA = 1 - (FN + FP + IDS) / GT

    where GT  = total ground-truth detections
          FN  = false negatives (missed detections)
          FP  = false positives (spurious detections)
          IDS = identity switches

    Parameters
    ----------
    gt_detections : list of Detection  (true_id is the GT person ID)
    pr_detections : list of Detection  (pred_id is FoundYou's global ID)
    iou_thr       : IoU threshold for a spatial match

    Returns
    -------
    dict with keys: mota, fp, fn, id_sw, gt_count, mota_score
    """
    # Group by (frame, cam)
    gt_by_frame : Dict = defaultdict(list)
    pr_by_frame : Dict = defaultdict(list)
    for d in gt_detections:
        gt_by_frame[(d.frame, d.cam)].append(d)
    for d in pr_detections:
        pr_by_frame[(d.frame, d.cam)].append(d)

    all_frames = sorted(set(gt_by_frame) | set(pr_by_frame))

    fp = fn = id_sw = 0
    gt_count = len(gt_detections)

    # Track the last predicted ID seen for each GT ID
    last_pred : Dict[int, int] = {}

    for fkey in all_frames:
        gt_f = gt_by_frame.get(fkey, [])
        pr_f = pr_by_frame.get(fkey, [])

        matched, unmatched_gt, unmatched_pr = _match_detections(
            gt_f, pr_f, iou_thr)

        fn += len(unmatched_gt)
        fp += len(unmatched_pr)

        for gi, pi in matched:
            true_id = gt_f[gi].true_id
            pred_id = pr_f[pi].pred_id
            if true_id in last_pred and last_pred[true_id] != pred_id:
                id_sw += 1
            last_pred[true_id] = pred_id

    mota_score = 1.0 - (fn + fp + id_sw) / max(gt_count, 1)
    return {
        'mota'      : round(mota_score, 4),
        'fp'        : fp,
        'fn'        : fn,
        'id_sw'     : id_sw,
        'gt_count'  : gt_count,
    }


def motp(gt_detections : List[Detection],
         pr_detections : List[Detection],
         iou_thr       : float = 0.5) -> Dict:
    """
    Multi-Object Tracking Precision (MOTP).

        MOTP = sum(IoU of matched pairs) / number of matches

    Higher is better (max 1.0).

    Returns
    -------
    dict with keys: motp, matched_count
    """
    gt_by_frame : Dict = defaultdict(list)
    pr_by_frame : Dict = defaultdict(list)
    for d in gt_detections:
        gt_by_frame[(d.frame, d.cam)].append(d)
    for d in pr_detections:
        pr_by_frame[(d.frame, d.cam)].append(d)

    all_frames    = sorted(set(gt_by_frame) | set(pr_by_frame))
    total_iou     = 0.0
    matched_count = 0

    for fkey in all_frames:
        gt_f = gt_by_frame.get(fkey, [])
        pr_f = pr_by_frame.get(fkey, [])
        matched, _, _ = _match_detections(gt_f, pr_f, iou_thr)
        for gi, pi in matched:
            total_iou     += _iou(gt_f[gi].bbox, pr_f[pi].bbox)
            matched_count += 1

    return {
        'motp'          : round(total_iou / max(matched_count, 1), 4),
        'matched_count' : matched_count,
    }


def idf1(gt_detections : List[Detection],
         pr_detections : List[Detection],
         iou_thr       : float = 0.5) -> Dict:
    """
    IDF1 — Identification F1 score.

        IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)

    where:
      IDTP = correctly identified detections (same spatial match AND same ID)
      IDFP = predictions with wrong ID
      IDFN = GT detections with no correct ID prediction

    This is the primary metric for multi-camera ReID quality. Unlike MOTA,
    it directly penalises persistent ID confusion rather than just switches.

    Returns
    -------
    dict with keys: idf1, idtp, idfp, idfn, idp, idr
    """
    gt_by_frame : Dict = defaultdict(list)
    pr_by_frame : Dict = defaultdict(list)
    for d in gt_detections:
        gt_by_frame[(d.frame, d.cam)].append(d)
    for d in pr_detections:
        pr_by_frame[(d.frame, d.cam)].append(d)

    all_frames = sorted(set(gt_by_frame) | set(pr_by_frame))

    idtp = idfp = idfn = 0

    for fkey in all_frames:
        gt_f = gt_by_frame.get(fkey, [])
        pr_f = pr_by_frame.get(fkey, [])
        matched, unmatched_gt, unmatched_pr = _match_detections(
            gt_f, pr_f, iou_thr)

        idfn += len(unmatched_gt)
        idfp += len(unmatched_pr)

        for gi, pi in matched:
            if gt_f[gi].true_id == pr_f[pi].pred_id:
                idtp += 1
            else:
                idfp += 1
                idfn += 1

    idp   = idtp / max(idtp + idfp, 1)
    idr   = idtp / max(idtp + idfn, 1)
    idf1_ = 2 * idtp / max(2 * idtp + idfp + idfn, 1)

    return {
        'idf1' : round(idf1_, 4),
        'idp'  : round(idp,   4),
        'idr'  : round(idr,   4),
        'idtp' : idtp,
        'idfp' : idfp,
        'idfn' : idfn,
    }


def id_switches(gt_detections : List[Detection],
                pr_detections : List[Detection],
                iou_thr       : float = 0.5) -> int:
    """
    Count raw identity switches: how many times a ground-truth person's
    predicted global ID changes between consecutive frames.

    Lower is better.
    """
    return mota(gt_detections, pr_detections, iou_thr)['id_sw']


def fragmentation(gt_detections : List[Detection],
                  pr_detections : List[Detection],
                  iou_thr       : float = 0.5) -> int:
    """
    Count track fragmentations: how many times a correctly-tracked GT person
    loses their predicted track and a new one starts for them later.

    A fragmentation is counted each time a GT ID's matching prediction
    goes from present → absent → present again (each interruption = +1).

    Lower is better.
    """
    gt_by_frame : Dict = defaultdict(list)
    pr_by_frame : Dict = defaultdict(list)
    for d in gt_detections:
        gt_by_frame[(d.frame, d.cam)].append(d)
    for d in pr_detections:
        pr_by_frame[(d.frame, d.cam)].append(d)

    all_frames = sorted(set(gt_by_frame) | set(pr_by_frame))

    # Track active / inactive state per GT ID per camera
    was_tracked : Dict[Tuple[int, int], bool] = defaultdict(lambda: False)
    frags = 0

    for fkey in all_frames:
        gt_f = gt_by_frame.get(fkey, [])
        pr_f = pr_by_frame.get(fkey, [])
        matched, _, _ = _match_detections(gt_f, pr_f, iou_thr)
        matched_gt_ids = {gt_f[gi].true_id for gi, _ in matched}

        for d in gt_f:
            key        = (d.true_id, d.cam)
            is_tracked = d.true_id in matched_gt_ids
            if not is_tracked and was_tracked[key]:
                frags += 1
            was_tracked[key] = is_tracked

    return frags


# ================================================================
# Gallery quality metrics
# ================================================================

def gallery_dist_stats(tracks: list) -> Dict:
    """
    Intra- and inter-gallery distance statistics across all GlobalTrack objects.

    Intra-gallery distance: average pairwise cosine distance between feature
    vectors stored in the same track's buffer. Low = consistent appearance.

    Inter-gallery distance: average cosine distance between the mean feature
    vectors of different tracks. High = well-separated identities.

    Parameters
    ----------
    tracks : list of GlobalTrack  (from DB.tracks or mgr.active.values())

    Returns
    -------
    dict with keys:
      intra_mean, intra_std  — intra-gallery spread (lower = more consistent)
      inter_mean, inter_std  — inter-gallery separation (higher = better)
      separation_ratio       — inter_mean / intra_mean  (higher is better)
      n_tracks               — number of tracks with enough probes to measure
    """
    from features import _dist

    intra_dists = []
    mean_feats  = []

    for gt in tracks:
        all_vecs = []
        for buf in gt._buf.values():
            all_vecs.extend(buf)

        if len(all_vecs) < 2:
            continue

        # Intra: average all pairwise distances
        vecs = np.stack(all_vecs).astype(np.float32)
        n    = len(vecs)
        d_sum = 0.0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                d_sum += _dist(vecs[i], vecs[j])
                pairs += 1
        intra_dists.append(d_sum / max(pairs, 1))

        if gt.feature is not None:
            mean_feats.append(gt.feature)

    inter_dists = []
    for i in range(len(mean_feats)):
        for j in range(i + 1, len(mean_feats)):
            inter_dists.append(_dist(mean_feats[i], mean_feats[j]))

    intra_arr = np.array(intra_dists) if intra_dists else np.array([0.0])
    inter_arr = np.array(inter_dists) if inter_dists else np.array([0.0])

    sep = float(inter_arr.mean()) / max(float(intra_arr.mean()), 1e-9)

    return {
        'intra_mean'      : round(float(intra_arr.mean()), 4),
        'intra_std'       : round(float(intra_arr.std()),  4),
        'inter_mean'      : round(float(inter_arr.mean()), 4),
        'inter_std'       : round(float(inter_arr.std()),  4),
        'separation_ratio': round(sep, 4),
        'n_tracks'        : len(intra_dists),
    }


def probe_convergence(track, max_probes: int = 30) -> Dict:
    """
    Measure how quickly a track's gallery distance stabilises vs probe count.

    Simulates adding one probe at a time (in insertion order) and records the
    cosine distance from the final full-gallery mean to the partial-gallery mean
    at each step. A fast convergence means the gallery is reliable early.

    Parameters
    ----------
    track      : GlobalTrack
    max_probes : maximum probes to simulate (capped at track.total_probes())

    Returns
    -------
    dict with keys:
      probe_counts  : list[int]   x-axis — number of probes seen so far
      distances     : list[float] y-axis — dist from partial to final mean
      stable_at     : int   first probe count where distance < 0.05 and stays
    """
    from features import _dist, _norm

    all_vecs = []
    for buf in track._buf.values():
        all_vecs.extend(buf)

    if len(all_vecs) < 2:
        return {'probe_counts': [], 'distances': [], 'stable_at': None}

    all_vecs = all_vecs[:max_probes]
    final_mean = _norm(np.stack(all_vecs).mean(0))

    probe_counts = []
    distances    = []
    stable_at    = None
    stable_run   = 0

    for k in range(1, len(all_vecs) + 1):
        partial_mean = _norm(np.stack(all_vecs[:k]).mean(0))
        d = _dist(partial_mean, final_mean)
        probe_counts.append(k)
        distances.append(round(d, 4))

        if d < 0.05:
            stable_run += 1
            if stable_run >= 3 and stable_at is None:
                stable_at = k - 2   # first of the 3-run
        else:
            stable_run = 0

    return {
        'probe_counts' : probe_counts,
        'distances'    : distances,
        'stable_at'    : stable_at,
    }


# ================================================================
# System / pipeline metrics
# ================================================================

def resolve_step_breakdown(result: TrackingResult) -> Dict:
    """
    Fraction of new-tracklet resolves handled by each pipeline step.

    Step A = matched active track    (cross-cam link or same-cam re-entry)
    Step B = reacquired archived     (person returned after absence)
    Step C = new track created       (genuinely new or failed to match)

    A high Step-C fraction means the system is over-creating tracks —
    tune CROSS_CAM_DIST or MIN_PROBES_TO_MATCH.

    Returns
    -------
    dict with keys: step_a, step_b, step_c (fractions), total_resolves
    """
    total = result.step_a + result.step_b + result.step_c
    if total == 0:
        return {'step_a': 0.0, 'step_b': 0.0, 'step_c': 0.0, 'total_resolves': 0}
    return {
        'step_a'          : round(result.step_a / total, 4),
        'step_b'          : round(result.step_b / total, 4),
        'step_c'          : round(result.step_c / total, 4),
        'total_resolves'  : total,
    }


def merge_effectiveness(result    : TrackingResult,
                        gt_frag   : int) -> Dict:
    """
    Compare merges fired vs ground-truth fragmentations corrected.

    A merge that corresponds to a real GT fragmentation is a true positive.
    Merges fired when no GT fragmentation existed are spurious (over-merging).

    Parameters
    ----------
    result   : TrackingResult   from the live run
    gt_frag  : int              fragmentation() count from GT comparison

    Returns
    -------
    dict with keys: merges_fired, gt_frags, correction_rate, over_merge_rate
    """
    merges = result.merges
    tp     = min(merges, gt_frag)          # can't correct more than exist
    fp     = max(0, merges - gt_frag)      # merges beyond what GT needed

    return {
        'merges_fired'    : merges,
        'gt_frags'        : gt_frag,
        'correction_rate' : round(tp / max(gt_frag, 1), 4),
        'over_merge_rate' : round(fp / max(merges,  1), 4),
    }


def throughput(result: TrackingResult,
               total_detections: int) -> Dict:
    """
    Compute system throughput.

    Parameters
    ----------
    result           : TrackingResult
    total_detections : int   total raw detections processed (across all cameras)

    Returns
    -------
    dict with keys: dets_per_sec, resolves_per_sec, wall_seconds
    """
    t = max(result.total_time, 1e-6)
    return {
        'dets_per_sec'     : round(total_detections / t, 1),
        'resolves_per_sec' : round((result.step_a + result.step_b + result.step_c) / t, 1),
        'wall_seconds'     : round(t, 2),
    }


# ================================================================
# Convenience: full report
# ================================================================

def full_report(gt_detections  : List[Detection],
                pr_detections  : List[Detection],
                result         : TrackingResult,
                tracks         : list,
                iou_thr        : float = 0.5,
                total_raw_dets : int   = 0) -> Dict:
    """
    Run all metrics and return a single flat report dict.

    Suitable for logging, CSV export, or paper tables.
    """
    mot  = mota(gt_detections, pr_detections, iou_thr)
    motp_ = motp(gt_detections, pr_detections, iou_thr)
    idf  = idf1(gt_detections, pr_detections, iou_thr)
    frag = fragmentation(gt_detections, pr_detections, iou_thr)
    gal  = gallery_dist_stats(tracks)
    step = resolve_step_breakdown(result)
    mrg  = merge_effectiveness(result, frag)
    tput = throughput(result, total_raw_dets)

    return {
        # --- tracking ---
        'mota'            : mot['mota'],
        'motp'            : motp_['motp'],
        'idf1'            : idf['idf1'],
        'idp'             : idf['idp'],
        'idr'             : idf['idr'],
        'fp'              : mot['fp'],
        'fn'              : mot['fn'],
        'id_switches'     : mot['id_sw'],
        'fragmentations'  : frag,
        # --- gallery ---
        'intra_dist_mean' : gal['intra_mean'],
        'inter_dist_mean' : gal['inter_mean'],
        'separation_ratio': gal['separation_ratio'],
        # --- pipeline ---
        'step_a_frac'     : step['step_a'],
        'step_b_frac'     : step['step_b'],
        'step_c_frac'     : step['step_c'],
        'correction_rate' : mrg['correction_rate'],
        'over_merge_rate' : mrg['over_merge_rate'],
        # --- system ---
        'dets_per_sec'    : tput['dets_per_sec'],
        'wall_seconds'    : tput['wall_seconds'],
    }
