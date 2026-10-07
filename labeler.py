"""
FoundYou — Ground Truth Labeler
=================================
OpenCV-based tool to assign ground-truth person IDs to a recorded session.
Opens each camera's recorded frames (or a video file), lets you click on
bounding boxes and type the real person ID, then saves a gt.csv file that
the benchmarks can use.

This is the minimum viable labeling workflow. For larger datasets use
a dedicated annotation tool (CVAT, Label Studio) and export in MOT format,
then use data.load_mot_sequence().

Usage
-----
  # Label from a saved session + original video:
  python -m labeler \
      --session recordings/session_20240101_120000 \
      --video   path/to/recording.mp4 \
      --cam     0

  # Label from session only (draws recorded bboxes on blank frames):
  python -m labeler \
      --session recordings/session_20240101_120000 \
      --cam 0 --no-video

Controls
--------
  Click a bounding box  → select it (turns yellow)
  Type a number         → assign that person ID to the selected box
  Enter                 → confirm assignment
  N                     → next frame
  P                     → previous frame
  S                     → save gt.csv and quit
  Q                     → quit without saving
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ================================================================
# Data structures
# ================================================================

class FrameAnnotation:
    """All ground-truth assignments for one (frame, cam) pair."""
    def __init__(self):
        # global_id → true_person_id
        self.assignments : Dict[int, int] = {}


# ================================================================
# Labeler UI
# ================================================================

class Labeler:
    BOX_COLOR      = (80,  200,  80)   # unselected
    SEL_COLOR      = (0,   220, 220)   # selected
    LABELED_COLOR  = (255, 140,   0)   # labeled
    TEXT_COLOR     = (255, 255, 255)
    BG_COLOR       = (20,   20,  20)
    FONT           = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, session_dir: str, cam_id: int,
                 video_path: Optional[str] = None):
        self.session_dir = session_dir
        self.cam_id      = cam_id
        self.video_path  = video_path

        # Load detections for this camera
        self.frames_dets = self._load_dets()
        self.frame_keys  = sorted(self.frames_dets.keys())

        if not self.frame_keys:
            raise RuntimeError(
                f"No detections found for cam {cam_id} in {session_dir}")

        # Video source
        self.cap : Optional[cv2.VideoCapture] = None
        if video_path and os.path.exists(video_path):
            self.cap = cv2.VideoCapture(video_path)

        # Annotation state
        self.annotations : Dict[int, FrameAnnotation] = defaultdict(FrameAnnotation)
        self.cursor       = 0
        self.selected_gid : Optional[int] = None
        self.input_buffer = ''

        # Load existing gt.csv if it exists
        self._load_existing_gt()

    # ------------------------------------------------------------------ #
    # Loading                                                              #
    # ------------------------------------------------------------------ #

    def _load_dets(self) -> Dict[int, List[dict]]:
        """Load detections.csv, return {frame_idx: [det_dict]}."""
        path = os.path.join(self.session_dir, 'detections.csv')
        by_frame : Dict[int, List[dict]] = defaultdict(list)
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                if int(row['cam']) != self.cam_id:
                    continue
                by_frame[int(row['frame'])].append({
                    'frame'    : int(row['frame']),
                    'cam'      : int(row['cam']),
                    'local_id' : int(row['local_id']),
                    'global_id': int(row['global_id']),
                    'bbox'     : (int(row['x1']), int(row['y1']),
                                  int(row['x2']), int(row['y2'])),
                })
        return dict(by_frame)

    def _load_existing_gt(self):
        gt_path = os.path.join(self.session_dir, f'gt_cam{self.cam_id}.csv')
        if not os.path.exists(gt_path):
            return
        with open(gt_path, newline='') as f:
            for row in csv.DictReader(f):
                frame = int(row['frame'])
                gid   = int(row['global_id'])
                pid   = int(row['true_id'])
                self.annotations[frame].assignments[gid] = pid
        print(f"[Labeler] Loaded existing GT: {gt_path}")

    # ------------------------------------------------------------------ #
    # Rendering                                                            #
    # ------------------------------------------------------------------ #

    def _get_frame_image(self, frame_idx: int) -> np.ndarray:
        """Return BGR image for this frame from video or blank canvas."""
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, img = self.cap.read()
            if ret and img is not None:
                return cv2.resize(img, (640, 480))
        return np.full((480, 640, 3), self.BG_COLOR, dtype=np.uint8)

    def _render(self, frame_idx: int) -> np.ndarray:
        img  = self._get_frame_image(frame_idx)
        dets = self.frames_dets.get(frame_idx, [])
        ann  = self.annotations[frame_idx]

        for d in dets:
            gid = d['global_id']
            x1, y1, x2, y2 = d['bbox']

            if gid == self.selected_gid:
                color = self.SEL_COLOR
            elif gid in ann.assignments:
                color = self.LABELED_COLOR
            else:
                color = self.BOX_COLOR

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Label: "G{gid} → P{true_id}" or just "G{gid}"
            if gid in ann.assignments:
                lbl = f"G{gid}->P{ann.assignments[gid]}"
            else:
                lbl = f"G{gid}"

            cv2.rectangle(img, (x1, y1 - 20), (x1 + len(lbl) * 9 + 4, y1),
                          color, -1)
            cv2.putText(img, lbl, (x1 + 2, y1 - 5),
                        self.FONT, 0.5, self.TEXT_COLOR, 1)

        # HUD
        total   = len(self.frame_keys)
        labeled = sum(1 for fk in self.frame_keys
                      if self.annotations[fk].assignments)
        progress = f"Frame {frame_idx}  [{self.cursor+1}/{total}]  " \
                   f"Labeled: {labeled}/{total}"
        cv2.putText(img, progress, (10, 20), self.FONT, 0.55,
                    (200, 200, 200), 1)

        # Input prompt
        if self.selected_gid is not None:
            prompt = f"G{self.selected_gid} → person ID: {self.input_buffer}_"
            cv2.rectangle(img, (0, 450), (640, 480), (40, 40, 40), -1)
            cv2.putText(img, prompt, (10, 470), self.FONT, 0.65,
                        (100, 255, 100), 2)

        # Controls reminder
        controls = "N=next  P=prev  click=select  enter=confirm  S=save  Q=quit"
        cv2.putText(img, controls, (10, 460), self.FONT, 0.38,
                    (120, 120, 120), 1)

        return img

    # ------------------------------------------------------------------ #
    # Mouse                                                                #
    # ------------------------------------------------------------------ #

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        frame_idx = self.frame_keys[self.cursor]
        for d in self.frames_dets.get(frame_idx, []):
            x1, y1, x2, y2 = d['bbox']
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selected_gid = d['global_id']
                self.input_buffer = ''
                return
        self.selected_gid = None
        self.input_buffer = ''

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #

    def run(self) -> Optional[str]:
        """
        Run the labeler UI. Returns path to saved gt.csv, or None if quit.
        """
        win = f"FoundYou Labeler — cam {self.cam_id}"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self._on_mouse)

        while True:
            frame_idx = self.frame_keys[self.cursor]
            img       = self._render(frame_idx)
            cv2.imshow(win, img)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                cv2.destroyAllWindows()
                return None

            elif key == ord('s'):
                path = self.save()
                cv2.destroyAllWindows()
                return path

            elif key == ord('n') or key == 83:   # N or right arrow
                self.cursor = min(self.cursor + 1, len(self.frame_keys) - 1)
                self.selected_gid = None
                self.input_buffer = ''

            elif key == ord('p') or key == 81:   # P or left arrow
                self.cursor = max(self.cursor - 1, 0)
                self.selected_gid = None
                self.input_buffer = ''

            elif key == 13:  # Enter — confirm assignment
                if self.selected_gid is not None and self.input_buffer.isdigit():
                    pid = int(self.input_buffer)
                    # Propagate to all frames where this global_id appears
                    for fk in self.frame_keys:
                        for d in self.frames_dets.get(fk, []):
                            if d['global_id'] == self.selected_gid:
                                self.annotations[fk].assignments[
                                    self.selected_gid] = pid
                    self.input_buffer = ''

            elif key == 8:   # Backspace
                self.input_buffer = self.input_buffer[:-1]

            elif 48 <= key <= 57:   # digits 0-9
                if self.selected_gid is not None:
                    self.input_buffer += chr(key)

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #

    def save(self) -> str:
        """Save ground-truth assignments as gt_cam{N}.csv."""
        gt_path = os.path.join(self.session_dir, f'gt_cam{self.cam_id}.csv')
        rows    = []
        for frame_idx in self.frame_keys:
            dets = self.frames_dets.get(frame_idx, [])
            ann  = self.annotations[frame_idx]
            for d in dets:
                gid     = d['global_id']
                true_id = ann.assignments.get(gid, 0)
                x1, y1, x2, y2 = d['bbox']
                rows.append({
                    'frame'    : frame_idx,
                    'cam'      : self.cam_id,
                    'true_id'  : true_id,
                    'global_id': gid,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                })

        with open(gt_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'frame', 'cam', 'true_id', 'global_id', 'x1', 'y1', 'x2', 'y2'])
            writer.writeheader()
            writer.writerows(rows)

        labeled = sum(1 for r in rows if r['true_id'] != 0)
        print(f"[Labeler] Saved {len(rows)} rows ({labeled} labeled) → {gt_path}")
        return gt_path


# ================================================================
# Convert labeled session → Detection lists for metrics
# ================================================================

def load_labeled_session(session_dir: str) -> Tuple[List, List, object]:
    """
    Load a labeled session and return (gt_detections, pr_detections, result).

    Combines all gt_cam*.csv files into a unified GT list, and loads the
    predicted detections from detections.csv.

    Returns
    -------
    (gt_detections, pr_detections, TrackingResult)
    """
    from metrics import Detection
    from recorder import SessionRecorder

    # Load predictions + pipeline events
    pr_dets, result, metadata = SessionRecorder.load_session(session_dir)

    # Load GT from all camera label files
    gt_dets = []
    for fname in sorted(os.listdir(session_dir)):
        if not fname.startswith('gt_cam') or not fname.endswith('.csv'):
            continue
        with open(os.path.join(session_dir, fname), newline='') as f:
            for row in csv.DictReader(f):
                true_id = int(row['true_id'])
                if true_id == 0:
                    continue   # unlabeled
                gt_dets.append(Detection(
                    frame   = int(row['frame']),
                    cam     = int(row['cam']),
                    true_id = true_id,
                    pred_id = int(row['global_id']),
                    bbox    = (int(row['x1']), int(row['y1']),
                               int(row['x2']), int(row['y2'])),
                ))

    return gt_dets, pr_dets, result


# ================================================================
# CLI
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(description="FoundYou ground-truth labeler")
    p.add_argument('--session',  required=True,
                   help='Path to a recorded session directory')
    p.add_argument('--cam',      type=int, default=0,
                   help='Camera ID to label (default: 0)')
    p.add_argument('--video',    type=str, default=None,
                   help='Optional video file for background frames')
    p.add_argument('--no-video', action='store_true',
                   help='Draw bboxes on a blank frame (no video needed)')
    return p.parse_args()


if __name__ == '__main__':
    args  = parse_args()
    video = None if args.no_video else args.video
    lb    = Labeler(args.session, args.cam, video_path=video)
    path  = lb.run()
    if path:
        print(f"Ground truth saved to: {path}")
        print(f"\nNow run the benchmark:")
        print(f"  python -m evaluate --session {args.session}")
