"""
FoundYou — Track Data Model & Persistence
==========================================
GlobalTrack  — one identity's feature gallery and lifecycle state.
DB           — FAISS index + pickle persistence for all tracks.
"""

import os
import pickle
from collections import defaultdict
from threading import Lock

import numpy as np
import faiss

from config import Config
from features import _norm

from logger import log

# ======================== GLOBAL TRACK =========================
class GlobalTrack:
    _nxt      = 1
    _nxt_lock = Lock()

    def __init__(self, ts: float):
        with GlobalTrack._nxt_lock:
            self.gid        = GlobalTrack._nxt
            GlobalTrack._nxt += 1

        self.first_seen    = ts
        self.last_seen     = ts
        self.det_count     = 0
        self.cams_seen     : set  = set()
        self._buf          : dict = defaultdict(list)
        self.feature       : np.ndarray | None = None
        self.locals        : dict = {}
        self.archived      = False
        self.archived_time = None

        np.random.seed(self.gid * 91823)
        self.color = tuple(int(x) for x in np.random.randint(60, 230, 3))

    def observe(self, cam: int, local_id: int, feat, ts: float):
        key = (cam, local_id)
        self.locals[key] = ts
        self.last_seen   = ts
        self.det_count  += 1
        self.cams_seen.add(cam)
        normed = _norm(feat)
        if normed is not None:
            buf = self._buf[cam]
            buf.append(normed)
            if len(buf) > Config.CAM_FEAT_BUF:
                buf.pop(0)
            self._rebuild()
        if self.archived:
            self.archived      = False
            self.archived_time = None

    def _rebuild(self):
        """Rebuild merged gallery feature (mean-of-per-camera-means)."""
        cam_means = []
        for buf in self._buf.values():
            if not buf:
                continue
            m = np.stack(buf).mean(0)
            n = np.linalg.norm(m)
            if n > 1e-6:
                cam_means.append(m / n)
        if not cam_means:
            return
        merged = np.stack(cam_means).mean(0)
        n = np.linalg.norm(merged)
        if n > 1e-6:
            self.feature = merged / n

    def cam_is_occupied(self, cam: int, ts: float) -> bool:
        """True if a live local tracklet on cam already maps to this track."""
        for (c, _), t in self.locals.items():
            if c == cam and ts - t < Config.EXPIRE_LOCAL_TTL:
                return True
        return False

    def expire(self, cam: int, local_id: int):
        self.locals.pop((cam, local_id), None)

    def clear_locals(self):
        self.locals.clear()

    def total_probes(self) -> int:
        """Total feature vectors stored across all cameras."""
        return sum(len(b) for b in self._buf.values())

    def is_matchable(self) -> bool:
        """
        FP-3 fix: gallery must have >= MIN_PROBES_TO_MATCH vectors
        before it can claim new tracklets.
        """
        return self.total_probes() >= Config.MIN_PROBES_TO_MATCH

# ======================== DATABASE =========================
class DB:
    def __init__(self):
        self.tracks      : list  = []
        self.pos         : dict  = {}
        self.index       = faiss.IndexFlatIP(Config.FEATURE_DIM)
        self.stats       = defaultdict(int)
        self.last_clean  = 0.0
        self._dirty      = False
        self._load()

    def _ok(self, v):
        return (v is not None and v.shape == (Config.FEATURE_DIM,)
                and np.isfinite(v).all() and np.linalg.norm(v) > 1e-6)

    def upsert(self, gt: GlobalTrack):
        """Add (new) or update (existing). FAISS is NOT written here."""
        if gt.gid in self.pos:
            self.tracks[self.pos[gt.gid]] = gt
        else:
            self.pos[gt.gid] = len(self.tracks)
            self.tracks.append(gt)
            self.stats['created'] += 1
            self._dirty = True

    def get(self, gid: int) -> 'GlobalTrack | None':
        i = self.pos.get(gid)
        return self.tracks[i] if i is not None and i < len(self.tracks) else None

    def active_on_load(self) -> list:
        """Tracks not archived at save time - promote back to active."""
        return [gt for gt in self.tracks if not gt.archived]

    def search_archived(self, feat, k: int = 20) -> list:
        """Return [(gid, sim, gt)] for archived tracks, best-first."""
        if not self.tracks or not self._ok(feat):
            return []
        archived_by_gid = {gt.gid: gt for gt in self.tracks if gt.archived}
        if not archived_by_gid:
            return []
        q        = feat.reshape(1,-1).astype('float32')
        k_search = min(k, self.index.ntotal)
        if k_search == 0:
            return []
        sims, idxs = self.index.search(q, k_search)
        out       = []
        seen_gids = set()
        for sim, idx in zip(sims[0], idxs[0]):
            if idx < 0 or idx >= len(self.tracks):
                continue
            candidate_gid = self.tracks[idx].gid
            if candidate_gid in seen_gids:
                continue
            gt = archived_by_gid.get(candidate_gid)
            if gt is None:
                continue
            if sim < Config.REACQ_MIN_SIM:
                continue
            seen_gids.add(candidate_gid)
            out.append((gt.gid, float(sim), gt))
        return out

    def rebuild_index(self):
        """Sole writer to self.index. Resets self.pos."""
        self.index = faiss.IndexFlatIP(Config.FEATURE_DIM)
        feats = np.zeros((len(self.tracks), Config.FEATURE_DIM), dtype='float32')
        for i, gt in enumerate(self.tracks):
            if self._ok(gt.feature):
                feats[i] = gt.feature
        self.index.add(feats)
        self.pos    = {gt.gid: i for i, gt in enumerate(self.tracks)}
        self._dirty = False
        if Config.DEBUG:
            log.debug(f"FAISS rebuild: {len(self.tracks)} tracks")

    def cleanup(self, ts: float):
        if ts - self.last_clean < Config.CLEANUP_INTERVAL:
            return
        cutoff = ts - Config.CLEANUP_AFTER
        to_del = [i for i, gt in enumerate(self.tracks)
                  if gt.archived and gt.archived_time
                  and gt.archived_time < cutoff]
        if to_del:
            for i in sorted(to_del, reverse=True):
                self.pos.pop(self.tracks[i].gid, None)
                del self.tracks[i]
                self.stats['cleaned'] += 1
            self.rebuild_index()
        self.last_clean = ts

    def save(self, next_id: int):
        try:
            with open(Config.DB_PATH, 'wb') as f:
                pickle.dump({'tracks': self.tracks, 'pos': self.pos,
                             'stats': dict(self.stats),
                             'next_id': next_id,
                             'fdim': Config.FEATURE_DIM}, f)
            faiss.write_index(self.index, Config.FAISS_PATH)
            log.info("DB saved")
        except Exception as e:
            log.warning(f"DB save failed: {e}")

    def _load(self):
        try:
            if not os.path.exists(Config.DB_PATH):
                return
            with open(Config.DB_PATH, 'rb') as f:
                d = pickle.load(f)
            if d.get('fdim') != Config.FEATURE_DIM:
                log.warning("DB feature_dim mismatch - starting fresh")
                return
            self.tracks = d.get('tracks', [])
            self.pos    = d.get('pos', {})
            self.stats.update(d.get('stats', {}))
            with GlobalTrack._nxt_lock:
                GlobalTrack._nxt = d.get('next_id', 1)

            for gt in self.tracks:
                if not hasattr(gt, '_buf'):
                    gt._buf = defaultdict(list)
                if not hasattr(gt, 'archived'):
                    gt.archived = False
                if not hasattr(gt, 'archived_time'):
                    gt.archived_time = None
                if not hasattr(gt, 'locals'):
                    gt.locals = {}
                else:
                    gt.locals.clear()

            self.rebuild_index()
            log.info(f"DB loaded: {len(self.tracks)} tracks  "
                     f"({sum(1 for g in self.tracks if not g.archived)} restore active, "
                     f"{sum(1 for g in self.tracks if g.archived)} archived)")
        except Exception as e:
            log.warning(f"DB load failed: {e} - starting fresh")

