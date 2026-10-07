"""
FoundYou — Entry Point
=======================
Wires all modules together. Owns the main loop, keyboard handling,
camera discovery, and graceful shutdown.

Usage:
    python main.py [options]

Options:
    --cross-cam-dist FLOAT
    --same-cam-reentry-dist FLOAT
    --reacq-dist FLOAT
    --min-probes-to-match INT
    --merge-dist FLOAT
    --merge-interval FLOAT
    --inactive-ttl FLOAT
    --yolo-model STR
    --device {cpu,cuda,mps}
    --debug
"""

import os
import time
import platform
import warnings
import argparse
from queue import Queue, Empty
from threading import Lock

import cv2
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings("ignore")

from logger import log
from config import Config
from models import FeatureExtractor
from reid import GlobalReIDManager
from camera import CameraWorker
from id_validator import IDValidator

from ui import (
    _sidebar, _title, _footer,
    assemble_video, draw_sidebar, draw_detections,
)


# ======================== CLI =========================

def parse_args():
    parser = argparse.ArgumentParser(description="FoundYou Multi-Camera ReID")
    parser.add_argument('--cross-cam-dist',        type=float, dest='CROSS_CAM_DIST')
    parser.add_argument('--same-cam-reentry-dist', type=float, dest='SAME_CAM_REENTRY_DIST')
    parser.add_argument('--reacq-dist',            type=float, dest='REACQ_DIST')
    parser.add_argument('--min-probes-to-match',   type=int,   dest='MIN_PROBES_TO_MATCH')
    parser.add_argument('--merge-dist',            type=float, dest='MERGE_DIST')
    parser.add_argument('--merge-interval',        type=float, dest='MERGE_INTERVAL')
    parser.add_argument('--inactive-ttl',          type=float, dest='INACTIVE_TTL')
    parser.add_argument('--yolo-model',            type=str,   dest='YOLO_MODEL')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], dest='DEVICE')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--flip-cameras', type=int, nargs='+', default=None,
                        dest='FLIP_CAMERAS',
                        help='Camera IDs whose frames should be horizontally '
                             'flipped before processing. e.g. --flip-cameras 1 2')
    parser.add_argument('--max-cameras', type=int, default=3,
                        dest='MAX_CAMERAS',
                        help='Number of camera indices to probe (default: 3)')
    parser.add_argument('--record', action='store_true',
                        help='Record session to disk for offline benchmarking')
    parser.add_argument('--record-dir', type=str, default='recordings',
                        dest='RECORD_DIR',
                        help='Output directory for recordings (default: recordings)')
    return parser.parse_args()


# ======================== Camera discovery =========================

def _discover_cameras(max_index: int = 10) -> dict:
    """
    Probe camera indices 0..max_index-1 and return {cam_id: VideoCapture}.

    Tries every backend for every index so virtual cameras (OBS, Snap,
    Camo, etc.) are not missed when physical cameras claim the first backend.

    On macOS, OBS Virtual Camera requires AVFoundation and OBS must be
    running with Virtual Camera started before launch.
    """
    sys_ = platform.system()

    # Try all backends per index — don't stop after first backend hit.
    if sys_ == 'Darwin':
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    elif sys_ == 'Windows':
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    BACKEND_NAMES = {
        cv2.CAP_AVFOUNDATION : "AVFoundation",
        cv2.CAP_DSHOW        : "DirectShow",
        cv2.CAP_MSMF         : "MSMF",
        cv2.CAP_V4L2         : "V4L2",
        cv2.CAP_ANY          : "Auto",
    }

    caps  : dict = {}
    found : set  = set()

    for i in range(max_index):
        for backend in backends:
            if i in found:
                break   # already claimed this index with a better backend
            try:
                c = cv2.VideoCapture(i, backend)
                if not c.isOpened():
                    c.release()
                    continue
                ret, fr = c.read()
                if not ret or fr is None or fr.size == 0:
                    c.release()
                    continue
                c.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                caps[i] = c
                found.add(i)
                bn = BACKEND_NAMES.get(backend, str(backend))
                log.info(f"Camera {i} ({bn})")
            except Exception:
                continue

    if not found:
        log.warning("No cameras found on indices 0-%d. "
                    "If using OBS Virtual Camera, make sure OBS is open "
                    "and Virtual Camera is started.", max_index - 1)
    return caps


# ======================== Main loop =========================

def run():
    args = parse_args()
    Config.override_from_args(args)
    if args.debug:
        Config.DEBUG = True

    feat_ext = FeatureExtractor()
    mgr      = GlobalReIDManager()

    # ID consistency validator
    validator = IDValidator(mgr)

    # Optional session recorder for offline benchmarking
    recorder = None
    if args.record:
        from recorder import SessionRecorder
        recorder = SessionRecorder(mgr, out_dir=args.RECORD_DIR)
        log.info(f"Recording enabled -> {recorder.session_dir}")

    log.info("Scanning cameras...")
    caps = _discover_cameras(max_index=args.MAX_CAMERAS)
    if not caps:
        log.error("No cameras found")
        return

    log.info(f"FoundYou started with {len(caps)} camera(s)")
    log.info(f"Device:              {Config.DEVICE}")
    log.info(f"Cross-cam dist:      {Config.CROSS_CAM_DIST}")
    log.info(f"Same-cam re-entry:   {Config.SAME_CAM_REENTRY_DIST}")
    log.info(f"Reacq dist:          {Config.REACQ_DIST}")
    log.info(f"Min probes to match: {Config.MIN_PROBES_TO_MATCH}")
    log.info(f"Merge dist:          {Config.MERGE_DIST}")
    log.info(f"Inactive TTL:        {Config.INACTIVE_TTL}s")

    result_q = Queue(maxsize=60)
    workers  = []
    for cid, cap in caps.items():
        w = CameraWorker(cid, cap, Config.YOLO_MODEL, feat_ext, result_q)
        w.start()
        workers.append(w)

    t0             = time.time()
    frame_counters : dict = {cid: 0 for cid in caps}
    ftimes         = []
    last_ft        = t0
    frames_data    = {cid: None for cid in caps}
    last_dets      : dict = {cid: [] for cid in caps}
    last_dets_lock : Lock = Lock()
    dead_workers   : set  = set()

    try:
        while True:
            ts = time.time() - t0
            nf = 0

            while not result_q.empty() and nf < len(caps) * 2:
                try:
                    cid, frame_or_sentinel, dets = result_q.get_nowait()
                    if frame_or_sentinel is CameraWorker.DEAD_SENTINEL:
                        if cid not in dead_workers:
                            dead_workers.add(cid)
                            log.error(f"Camera {cid} worker died - "
                                      f"check model path / ultralytics install.")
                        continue
                    with last_dets_lock:
                        frames_data[cid] = frame_or_sentinel.copy()
                        last_dets[cid]   = dets
                    nf += 1
                except Empty:
                    break

            if nf > 0:
                with last_dets_lock:
                    snapshot = {k: list(v) for k, v in last_dets.items()}
                all_dets = [d for ds in snapshot.values() for d in ds]
                if all_dets:
                    with mgr.lock:
                        mgr.update(all_dets, ts)
                    if recorder:
                        for cid, cid_dets in snapshot.items():
                            recorder.record_frame(cid, frame_counters[cid], cid_dets)
                            frame_counters[cid] += 1

            with last_dets_lock:
                dets_snapshot = {k: list(v) for k, v in last_dets.items()}

            # Run ID validation check
            with mgr.lock:
                val_issues = validator.check(ts)

            # Draw detections onto each camera frame
            frames_ready = {}
            for cid, frame in frames_data.items():
                if frame is None:
                    frames_ready[cid] = np.zeros((480, 640, 3), np.uint8)
                    continue
                frames_ready[cid] = draw_detections(
                    frame, dets_snapshot.get(cid, []), mgr, ts, val_issues)

            now = time.time()
            ftimes.append(now - last_ft)
            if len(ftimes) > 30:
                ftimes.pop(0)
            fps     = 1.0 / (sum(ftimes) / len(ftimes) + 1e-9)
            last_ft = now

            sorted_cids = sorted(frames_ready.keys())
            if not sorted_cids:
                time.sleep(0.001)
                continue

            video = assemble_video(frames_ready, sorted_cids,
                                   dets_snapshot, mgr, ts, dead_workers)

            sb   = _sidebar(420, video.shape[0])
            sb   = draw_sidebar(sb, mgr, ts, fps, val_issues)
            body = np.hstack([video, sb])
            disp = np.vstack([_title(body.shape[1]), body, _footer(body.shape[1])])
            if disp.shape[1] > 2400:
                s    = 2400 / disp.shape[1]
                disp = cv2.resize(disp, (0, 0), fx=s, fy=s)

            cv2.imshow("FoundYou", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                Config.DEBUG = not Config.DEBUG
                log.info(f"Debug {'ON' if Config.DEBUG else 'OFF'}")
            elif key == ord('s'):
                with mgr.lock:
                    log.info(f"\n{'=' * 55}\n t={ts:.0f}s  fps={fps:.1f}")
                    log.info(f"  active IDs : {sorted(mgr.active.keys())}")
                    log.info(f"  l2g        : {dict(mgr.l2g)}")
                    for k, v in mgr.db.stats.items():
                        log.info(f"  {k:<14}: {v}")
                    log.info('=' * 55)
            elif key == ord('v'):
                log.info(f"\n{validator.report()}")
            elif key == ord('x'):
                fname = time.strftime("Screenshots/foundyou_%Y%m%d_%H%M%S.png")
                cv2.imwrite(fname, disp)
                log.info(f"Screenshot saved: {fname}")

            time.sleep(0.001)

    finally:
        log.info("Shutting down...")
        for w in workers:
            w.stop()
        for w in workers:
            w.join(timeout=2.0)
        if recorder:
            recorder.save()
        with mgr.lock:
            from tracks import GlobalTrack
            with GlobalTrack._nxt_lock:
                next_id = GlobalTrack._nxt
            mgr.db.save(next_id)
        for c in caps.values():
            c.release()
        cv2.destroyAllWindows()
        log.info("Done.")


if __name__ == "__main__":
    run()