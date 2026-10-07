"""
FoundYou — Live Session Recorder
==================================
Hooks into GlobalReIDManager at runtime to record every detection,
pipeline resolve event, and merge — then saves it all to disk so the
offline benchmarks can consume it.

How it works
------------
  1. Wrap your GlobalReIDManager with SessionRecorder before the main loop.
  2. Call recorder.record_frame(cam, frame_idx, dets) after each mgr.update().
  3. Call recorder.save() on shutdown.
  4. Run the benchmarks against the saved file.

Integration into main.py
------------------------
  Replace the relevant section in run() with:

    from recorder import SessionRecorder

    mgr      = GlobalReIDManager()
    recorder = SessionRecorder(mgr, out_dir="recordings")

    # inside the main loop, after mgr.update():
    recorder.record_frame(cid, frame_counter[cid], dets_snapshot[cid])

    # in the finally block, before mgr.db.save():
    recorder.save()

Output files
------------
  recordings/
    session_<timestamp>/
      detections.csv      — one row per detection per frame
      events.csv          — resolve and merge events
      config_snapshot.json — Config values at recording time
      metadata.json        — camera count, frame count, duration

CSV formats
-----------
  detections.csv:
    frame, cam, local_id, global_id, x1, y1, x2, y2, ts

  events.csv:
    ts, event_type, gid, cam, local_id, detail
    event_type ∈ {step_a_cross, step_a_reentry, step_b_reacq, step_c_new, merge}
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from threading import Lock
from typing import Dict, List

from metrics import Detection, TrackingResult


class SessionRecorder:
    """
    Wraps GlobalReIDManager to intercept and record all runtime events.

    Monkey-patches mgr._resolve and mgr._merge_active_tracks to capture
    step-level decisions without modifying reid.py.
    """

    def __init__(self, mgr, out_dir: str = "recordings"):
        self.mgr       = mgr
        self.out_dir   = out_dir
        self._lock     = Lock()
        self._start_ts = time.time()

        # Session directory
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(out_dir, f"session_{stamp}")
        os.makedirs(self.session_dir, exist_ok=True)

        # Accumulators
        self._det_rows    : List[tuple] = []
        self._event_rows  : List[tuple] = []
        self._frame_count : Dict[int, int] = defaultdict(int)
        self.result       = TrackingResult()

        # Patch the manager
        self._patch_manager()

        # Save config snapshot immediately
        self._save_config_snapshot()

        print(f"[Recorder] Session: {self.session_dir}")

    # ------------------------------------------------------------------ #
    # Monkey-patching                                                      #
    # ------------------------------------------------------------------ #

    def _patch_manager(self):
        """
        Replace mgr._resolve and mgr._merge_active_tracks with wrappers
        that record events before delegating to the originals.
        """
        original_resolve = self.mgr._resolve
        original_merge   = self.mgr._merge_active_tracks
        recorder         = self

        def patched_resolve(cam: int, local_id: int, feat, ts: float) -> int:
            gid = original_resolve(cam, local_id, feat, ts)
            recorder._on_resolve(cam, local_id, gid, ts)
            return gid

        def patched_merge(ts: float):
            merges_before = self.mgr.db.stats.get('merged', 0)
            original_merge(ts)
            merges_after  = self.mgr.db.stats.get('merged', 0)
            new_merges    = merges_after - merges_before
            for _ in range(new_merges):
                recorder._on_merge(ts)

        self.mgr._resolve             = patched_resolve
        self.mgr._merge_active_tracks = patched_merge

    def _on_resolve(self, cam: int, local_id: int, gid: int, ts: float):
        """Called after every _resolve(). Determine which step fired."""
        with self._lock:
            # Heuristic: check DB stats delta to infer which step ran.
            # A new link → step_a; reacquired → step_b; created → step_c.
            # We check by comparing gid against what existed before resolve.
            stats = self.mgr.db.stats

            # step_c: gid is brand new (highest gid seen = this one)
            from tracks import GlobalTrack
            is_new = (gid == GlobalTrack._nxt - 1)

            # step_b: track was archived and is now active
            gt = self.mgr.active.get(gid)
            was_archived = (gt is not None and
                            not gt.archived and
                            gt.det_count == 1 and
                            len(gt._buf.get(cam, [])) == 1)

            if is_new:
                self.result.step_c += 1
                event = 'step_c_new'
                detail = f'gid={gid}'
            elif was_archived:
                self.result.step_b += 1
                event = 'step_b_reacq'
                detail = f'gid={gid}'
            else:
                self.result.step_a += 1
                same_cam = (cam in (self.mgr.active.get(gid).cams_seen
                                    if self.mgr.active.get(gid) else set()))
                event  = 'step_a_reentry' if same_cam else 'step_a_cross'
                detail = f'gid={gid}'

            self._event_rows.append((
                round(ts, 3), event, gid, cam, local_id, detail
            ))

    def _on_merge(self, ts: float):
        with self._lock:
            self.result.merges += 1
            self._event_rows.append((
                round(ts, 3), 'merge', -1, -1, -1, ''
            ))

    # ------------------------------------------------------------------ #
    # Per-frame recording                                                  #
    # ------------------------------------------------------------------ #

    def record_frame(self, cam: int, frame_idx: int, dets: List[dict]):
        """
        Call this after mgr.update() for each camera each frame.

        Parameters
        ----------
        cam       : camera ID
        frame_idx : frame counter for this camera (monotonically increasing)
        dets      : the detection list from dets_snapshot[cam] — each dict
                    must have keys: local_id, gid, bbox
        """
        ts = time.time() - self._start_ts
        with self._lock:
            self._frame_count[cam] = max(self._frame_count[cam], frame_idx)
            for d in dets:
                gid = d.get('gid')
                if gid is None:
                    continue
                x1, y1, x2, y2 = d['bbox']
                self._det_rows.append((
                    frame_idx, cam, d['local_id'], gid,
                    x1, y1, x2, y2, round(ts, 3),
                ))

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #

    def save(self):
        """Write all buffered data to the session directory."""
        self.result.total_time = time.time() - self._start_ts

        # detections.csv
        det_path = os.path.join(self.session_dir, 'detections.csv')
        with open(det_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['frame', 'cam', 'local_id', 'global_id',
                        'x1', 'y1', 'x2', 'y2', 'ts'])
            w.writerows(self._det_rows)

        # events.csv
        ev_path = os.path.join(self.session_dir, 'events.csv')
        with open(ev_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['ts', 'event_type', 'gid', 'cam', 'local_id', 'detail'])
            w.writerows(self._event_rows)

        # metadata.json
        meta = {
            'duration_s'   : round(self.result.total_time, 2),
            'cameras'      : sorted(self._frame_count.keys()),
            'frame_counts' : dict(self._frame_count),
            'total_dets'   : len(self._det_rows),
            'step_a'       : self.result.step_a,
            'step_b'       : self.result.step_b,
            'step_c'       : self.result.step_c,
            'merges'       : self.result.merges,
        }
        with open(os.path.join(self.session_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        total_events = len(self._event_rows)
        print(f"[Recorder] Saved {len(self._det_rows)} detections, "
              f"{total_events} events → {self.session_dir}")

    # ------------------------------------------------------------------ #
    # Load back for benchmarking                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def load_session(session_dir: str):
        """
        Load a saved session back into Detection lists and a TrackingResult.

        Returns
        -------
        (pr_detections, result, metadata)
        """
        det_path = os.path.join(session_dir, 'detections.csv')
        ev_path  = os.path.join(session_dir, 'events.csv')
        meta_path = os.path.join(session_dir, 'metadata.json')

        pr_dets = []
        with open(det_path, newline='') as f:
            for row in csv.DictReader(f):
                pr_dets.append(Detection(
                    frame   = int(row['frame']),
                    cam     = int(row['cam']),
                    true_id = 0,   # filled in by align_with_gt()
                    pred_id = int(row['global_id']),
                    bbox    = (int(row['x1']), int(row['y1']),
                               int(row['x2']), int(row['y2'])),
                ))

        result = TrackingResult()
        with open(ev_path, newline='') as f:
            for row in csv.DictReader(f):
                et = row['event_type']
                if et.startswith('step_a'):
                    result.step_a += 1
                elif et == 'step_b_reacq':
                    result.step_b += 1
                elif et == 'step_c_new':
                    result.step_c += 1
                elif et == 'merge':
                    result.merges += 1

        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)
            result.total_time = metadata.get('duration_s', 0.0)

        return pr_dets, result, metadata

    def _save_config_snapshot(self):
        """Save current Config values to JSON for reproducibility."""
        try:
            from config import Config
            snapshot = {k: v for k, v in vars(Config).items()
                        if not k.startswith('_') and not callable(v)}
            # Convert non-serialisable types
            for k, v in snapshot.items():
                if not isinstance(v, (int, float, str, bool, type(None))):
                    snapshot[k] = str(v)
            with open(os.path.join(self.session_dir, 'config_snapshot.json'), 'w') as f:
                json.dump(snapshot, f, indent=2)
        except Exception as e:
            print(f"[Recorder] Config snapshot failed: {e}")
