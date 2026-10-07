"""
FoundYou — Benchmark Data Utilities
=====================================
Helpers for loading ground-truth annotations and converting FoundYou's
internal state into the Detection / TrackingResult format expected by
metrics.py.

Supported annotation formats
-----------------------------
  CSV  (MOT-style)
    frame, cam, person_id, x1, y1, w, h
    e.g.  1,0,3,120,80,60,180

  MOT17/MOT20  (.txt per-sequence)
    frame, id, x1, y1, w, h, conf, -1, -1, -1
    (single-camera; cam_id supplied separately)

Synthetic data
--------------
  generate_synthetic_sequence() creates a controllable multi-camera
  scenario for unit-testing metrics without real video:
    - N people walking between cameras
    - Configurable ID switches and fragmentations injected
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import numpy as np

from metrics import Detection, TrackingResult


# ================================================================
# Loading ground truth
# ================================================================

def load_csv(path: str) -> List[Detection]:
    """
    Load a CSV annotation file.

    Expected columns (no header):
        frame, cam, person_id, x1, y1, w, h

    Lines starting with '#' are ignored.
    """
    dets = []
    with open(path, newline='') as f:
        for row in csv.reader(f):
            if not row or row[0].startswith('#'):
                continue
            if len(row) < 7:
                continue
            frame, cam, pid, x1, y1, w, h = (int(v) for v in row[:7])
            dets.append(Detection(
                frame   = frame,
                cam     = cam,
                true_id = pid,
                pred_id = 0,              # GT has no prediction
                bbox    = (x1, y1, x1 + w, y1 + h),
            ))
    return dets


def load_mot_txt(path: str, cam_id: int = 0) -> List[Detection]:
    """
    Load a MOT17/MOT20 ground-truth .txt file (single camera).

    Format: frame,id,x1,y1,w,h,conf,cx,cy,cz
    Rows where conf == 0 (ignored regions) are skipped.
    """
    dets = []
    with open(path, newline='') as f:
        for row in csv.reader(f):
            if not row or row[0].startswith('#'):
                continue
            if len(row) < 7:
                continue
            frame, pid, x1, y1, w, h, conf = (
                int(float(v)) for v in row[:7])
            if conf == 0:
                continue
            dets.append(Detection(
                frame   = frame,
                cam     = cam_id,
                true_id = pid,
                pred_id = 0,
                bbox    = (x1, y1, x1 + w, y1 + h),
            ))
    return dets


def load_mot_sequence(seq_dir: str) -> List[Detection]:
    """
    Load all cameras in a sequence directory.

    Expected layout:
        seq_dir/
          cam0/gt/gt.txt
          cam1/gt/gt.txt
          ...
    """
    dets = []
    for entry in sorted(os.listdir(seq_dir)):
        cam_path = os.path.join(seq_dir, entry, 'gt', 'gt.txt')
        if not os.path.exists(cam_path):
            continue
        try:
            cam_id = int(entry.replace('cam', ''))
        except ValueError:
            cam_id = len(dets)
        dets.extend(load_mot_txt(cam_path, cam_id))
    return dets


# ================================================================
# Converting FoundYou output → Detection list
# ================================================================

def detections_from_log(log_path: str) -> Tuple[List[Detection], TrackingResult]:
    """
    Parse a FoundYou debug log to extract predicted detections and pipeline
    step counts. Useful when you don't have live access to the manager.

    Log lines parsed:
      - "Link [cross-cam] G<gid> <- cam<c>/L<lid>"  → step_a
      - "Reacquired G<gid>"                          → step_b
      - "New G<gid>"                                 → step_c
      - "Merged G<loser> -> G<winner>"               → merge

    Note: bboxes are not available in the log; returned Detections have
    bbox=(0,0,1,1) as a placeholder. Use detections_from_snapshot() for
    real bboxes.

    Returns
    -------
    (detections, TrackingResult)
    """
    import re

    dets   = []
    result = TrackingResult()

    link_re   = re.compile(r'Link \[.+?\] G(\d+) <- cam(\d+)/L(\d+)')
    reacq_re  = re.compile(r'Reacquired G(\d+)')
    new_re    = re.compile(r'New G(\d+)\s+cam(\d+)/L(\d+)')
    merge_re  = re.compile(r'Merged G(\d+) -> G(\d+)')

    frame = 0
    with open(log_path) as f:
        for line in f:
            if m := link_re.search(line):
                result.step_a += 1
                gid, cam = int(m.group(1)), int(m.group(2))
                dets.append(Detection(frame, cam, 0, gid, (0, 0, 1, 1)))
                frame += 1
            elif reacq_re.search(line):
                result.step_b += 1
            elif m := new_re.search(line):
                result.step_c += 1
                gid, cam = int(m.group(1)), int(m.group(2))
                dets.append(Detection(frame, cam, 0, gid, (0, 0, 1, 1)))
                frame += 1
            elif merge_re.search(line):
                result.merges += 1

    return dets, result


def detections_from_snapshot(db_tracks    : list,
                              l2g          : dict,
                              frame_dets   : List[dict],
                              frame_idx    : int,
                              cam_id       : int) -> List[Detection]:
    """
    Convert one frame's raw detection dicts (as produced by CameraWorker)
    into Detection objects with pred_id filled in.

    Parameters
    ----------
    db_tracks  : list of GlobalTrack   (mgr.db.tracks)
    l2g        : dict   (mgr.l2g)
    frame_dets : list of dicts with keys: local_id, bbox, gid
    frame_idx  : int
    cam_id     : int
    """
    out = []
    for d in frame_dets:
        gid = d.get('gid')
        if gid is None:
            continue
        x1, y1, x2, y2 = d['bbox']
        out.append(Detection(
            frame   = frame_idx,
            cam     = cam_id,
            true_id = 0,          # ground truth not known here
            pred_id = gid,
            bbox    = (x1, y1, x2, y2),
        ))
    return out


# ================================================================
# Synthetic test data
# ================================================================

def generate_synthetic_sequence(
    n_people       : int  = 4,
    n_cameras      : int  = 2,
    n_frames       : int  = 200,
    n_id_switches  : int  = 2,
    n_fragmentations: int = 3,
    frame_w        : int  = 640,
    frame_h        : int  = 480,
    seed           : int  = 42,
) -> Tuple[List[Detection], List[Detection]]:
    """
    Generate a synthetic multi-camera sequence with known ground truth
    and a predicted sequence with injected errors.

    Use this to unit-test metrics without needing real video:

        gt, pr = generate_synthetic_sequence(n_id_switches=2)
        result = mota(gt, pr)
        assert result['id_sw'] == 2

    Parameters
    ----------
    n_people        : number of distinct people in the scene
    n_cameras       : number of cameras (each person appears on all cameras
                      for roughly equal time)
    n_frames        : total frame count per camera
    n_id_switches   : number of injected prediction ID switches
    n_fragmentations: number of injected prediction track drops
    frame_w/h       : frame dimensions for plausible bboxes
    seed            : random seed for reproducibility

    Returns
    -------
    (gt_detections, pr_detections)
    """
    rng = np.random.default_rng(seed)

    gt_dets : List[Detection] = []
    pr_dets : List[Detection] = []

    # Each person starts at a random position and walks randomly
    positions = {pid: rng.integers([50, 50], [frame_w - 100, frame_h - 100])
                 for pid in range(1, n_people + 1)}
    velocities = {pid: rng.integers(-5, 5, size=2)
                  for pid in range(1, n_people + 1)}

    # Track which predicted ID each person currently has per camera
    # Initially pred_id == true_id
    pred_ids : Dict[Tuple[int, int], int] = {}   # (cam, true_id) → pred_id
    next_pred_id = n_people + 1

    # Inject switches and fragmentations at random frames
    switch_frames = set(rng.choice(n_frames, min(n_id_switches, n_frames),
                                   replace=False).tolist())
    frag_frames   = set(rng.choice(n_frames, min(n_fragmentations, n_frames),
                                   replace=False).tolist())

    switches_left = n_id_switches
    frags_left    = n_fragmentations

    for cam in range(n_cameras):
        for frame in range(n_frames):
            for pid in range(1, n_people + 1):
                # Update position
                pos = positions[pid].copy()
                vel = velocities[pid]
                pos = np.clip(pos + vel, [10, 10], [frame_w - 70, frame_h - 120])
                positions[pid] = pos
                velocities[pid] += rng.integers(-2, 2, size=2)
                velocities[pid] = np.clip(velocities[pid], -8, 8)

                x1, y1 = int(pos[0]), int(pos[1])
                x2, y2 = x1 + 60, y1 + 120
                bbox = (x1, y1, x2, y2)

                # Ground truth detection
                gt_dets.append(Detection(frame, cam, pid, 0, bbox))

                # Predicted detection — start with correct ID
                key = (cam, pid)
                if key not in pred_ids:
                    pred_ids[key] = pid

                current_pred = pred_ids[key]

                # Inject ID switch
                if frame in switch_frames and switches_left > 0:
                    current_pred = next_pred_id
                    pred_ids[key] = current_pred
                    next_pred_id += 1
                    switches_left -= 1

                # Inject fragmentation (skip this detection)
                if frame in frag_frames and frags_left > 0:
                    frags_left -= 1
                    continue   # no predicted detection this frame

                pr_dets.append(Detection(frame, cam, 0, current_pred, bbox))

    return gt_dets, pr_dets
