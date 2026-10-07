"""
FoundYou — Multi-Camera Person Re-Identification
====================================================

Architecture
------------
  Per-camera tracking : ByteTrack (ultralytics built-in)
  Global ReID layer   : OSNet-512 cosine similarity

  _resolve() pipeline for every NEW local tracklet:

    Step A  Match ACTIVE global track
    Step B  Reacquire ARCHIVED track
    Step C  Create new global track

Dependencies
------------
  pip install ultralytics torch torchvision faiss-cpu
  pip install gdown      # optional - OSNet weight download
  pip install torchreid  # optional - better OSNet weight loading
"""

import os
import time
import pickle
import platform
import warnings
import argparse
import logging
from threading import Thread, Lock
from queue import Queue, Empty
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import faiss

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("FoundYou")


# ======================== CONFIGURATION =========================
class Config:
    # ----- Matching thresholds (cosine DISTANCE = 1 - similarity) -----

    # FP-2 fix: cross-camera threshold.
    CROSS_CAM_DIST        = 0.40

    # FP-2 fix: same-camera re-entry threshold (tight — appearance barely changes).
    SAME_CAM_REENTRY_DIST = 0.30

    # FP-4 fix: archive reacquisition threshold, tightened from 0.50 to 0.40.
    REACQ_DIST            = 0.40
    # FAISS inner-product pre-filter floor (1 - REACQ_DIST).
    REACQ_MIN_SIM         = 0.60

    # FP-3 fix: minimum probes before a gallery is allowed to claim new tracklets.
    MIN_PROBES_TO_MATCH   = 3

    # Post-hoc merge: two active tracks whose robust distance falls below this
    # threshold are collapsed into one (the older/higher-det-count track wins).
    # This is the fix for early fragmentation — two IDs assigned to the same
    # person before either gallery was matchable are corrected once both
    # galleries have enough probes to make a reliable comparison.
    # Set slightly tighter than CROSS_CAM_DIST so we only merge when genuinely
    # confident, not just within the cross-camera tolerance band.
    MERGE_DIST            = 0.35

    # How often (seconds) to run the active-track merge pass.
    MERGE_INTERVAL        = 2.0

    # ----- Feature gallery ---------------------------------------------
    CAM_FEAT_BUF = 24

    # ----- Track lifecycle ---------------------------------------------
    INACTIVE_TTL     = 15.0
    EXPIRE_LOCAL_TTL = 4.0
    MERGE_COOLDOWN   = 0.5
    CLEANUP_AFTER    = 300.0
    CLEANUP_INTERVAL = 120.0

    # ----- FAISS -------------------------------------------------------
    FAISS_REBUILD_INTERVAL = 30.0

    # ----- ByteTrack ---------------------------------------------------
    BT_CONF    = 0.45
    BT_IOU     = 0.45
    YOLO_MODEL = 'yolo11s.pt'

    # ----- Camera handling ---------------------------------------------
    CAM_MAX_FAILURES = 30

    # ----- Misc --------------------------------------------------------
    FEATURE_DIM = 512
    DB_PATH     = "foundyou.pkl"
    FAISS_PATH  = "foundyou.faiss"
    DEVICE      = 'mps' if torch.backends.mps.is_available() else \
                  ('cuda' if torch.cuda.is_available() else 'cpu')
    DEBUG       = True

    @classmethod
    def override_from_args(cls, args):
        for key, value in vars(args).items():
            if value is not None and hasattr(cls, key.upper()):
                setattr(cls, key.upper(), value)
                log.info(f"Config override: {key.upper()} = {value}")


# ======================== OSNet Backbone =========================
class _ConvBnRelu(nn.Module):
    def __init__(self, ic, oc, k, s=1, p=0, g=1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(ic, oc, k, stride=s, padding=p, groups=g, bias=False),
            nn.BatchNorm2d(oc), nn.ReLU(inplace=True))
    def forward(self, x): return self.f(x)

class _LiteConv(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.f = nn.Sequential(_ConvBnRelu(ic, ic, 3, p=1, g=ic),
                                _ConvBnRelu(ic, oc, 1))
    def forward(self, x): return self.f(x)

class _OSBlock(nn.Module):
    def __init__(self, ic, oc, r=4):
        super().__init__()
        mid = max(oc // r, 16)
        self.c1 = _ConvBnRelu(ic, mid, 1)
        self.streams = nn.ModuleList([
            _LiteConv(mid, mid),
            nn.Sequential(_LiteConv(mid, mid), _LiteConv(mid, mid)),
            nn.Sequential(_LiteConv(mid, mid), _LiteConv(mid, mid), _LiteConv(mid, mid)),
            nn.Sequential(_LiteConv(mid, mid), _LiteConv(mid, mid),
                          _LiteConv(mid, mid), _LiteConv(mid, mid)),
        ])
        gm = max(mid // 4, 1)
        self.gate = nn.Sequential(nn.Linear(mid, gm), nn.ReLU(inplace=True),
                                   nn.Linear(gm, 4), nn.Sigmoid())
        self.c2   = _ConvBnRelu(mid, oc, 1)
        self.skip = nn.Sequential(_ConvBnRelu(ic, oc, 1)) if ic != oc else nn.Identity()

    def forward(self, x):
        br = [s(self.c1(x)) for s in self.streams]
        w  = self.gate(sum(b.mean([2,3]) for b in br)/4).unsqueeze(-1).unsqueeze(-1)
        return F.relu(self.skip(x) + self.c2(sum(w[:,i]*br[i] for i in range(4))),
                      inplace=True)

class OSNet(nn.Module):
    CH = [64, 256, 384, 512]
    def __init__(self, fdim=512):
        super().__init__()
        c = self.CH
        self.stem   = nn.Sequential(_ConvBnRelu(3, c[0], 7, s=2, p=3),
                                     nn.MaxPool2d(3, stride=2, padding=1))
        self.layer2 = self._layer(c[0], c[1], 2, True)
        self.layer3 = self._layer(c[1], c[2], 2, True)
        self.layer4 = self._layer(c[2], c[3], 2, False)
        self.head   = nn.Sequential(_ConvBnRelu(c[3], c[3], 1),
                                     nn.AdaptiveAvgPool2d(1))
        self.fc     = nn.Linear(c[3], fdim)

    @staticmethod
    def _layer(ic, oc, n, ds):
        layers = [_OSBlock(ic, oc)] + [_OSBlock(oc, oc) for _ in range(n-1)]
        if ds:
            layers += [nn.Sequential(nn.Conv2d(oc,oc,2,stride=2,bias=False),
                                      nn.BatchNorm2d(oc), nn.ReLU(inplace=True))]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.fc(self.head(x).flatten(1))


def _build_backbone():
    """Returns (net, feature_dim). Does NOT mutate Config."""
    try:
        import torchreid
        src = torchreid.models.build_model(
            name='osnet_x1_0', num_classes=1, pretrained=True)
        src.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 128)
            out   = src(dummy)
        fdim = out.shape[1]
        log.info(f"OSNet via torchreid - fdim={fdim}")
        return src, fdim
    except Exception as e:
        log.warning(f"torchreid not usable ({e}), falling back to custom OSNet")

    fdim  = Config.FEATURE_DIM
    model = OSNet(fdim)
    cache = os.path.expanduser("~/.cache/osnet/osnet_x1_0_imagenet.pth")
    os.makedirs(os.path.dirname(cache), exist_ok=True)

    if not os.path.exists(cache):
        try:
            import gdown
            log.info("Downloading OSNet weights...")
            gdown.download(
                "https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY",
                cache, quiet=False)
        except Exception as e:
            log.warning(f"Download failed: {e} - random weights (poor ReID)")
            return model, fdim

    try:
        raw = torch.load(cache, map_location='cpu')
        sd  = raw.get('state_dict', raw) if isinstance(raw, dict) else raw
        dst = model.state_dict()
        ok  = {k: v for k, v in sd.items()
               if k in dst and dst[k].shape == v.shape}
        if len(ok) == 0:
            log.warning("0 keys matched - random weights. Install torchreid.")
        else:
            model.load_state_dict(ok, strict=False)
            log.info(f"OSNet from cache ({len(ok)}/{len(dst)} keys)")
    except Exception as e:
        log.warning(f"Cache load failed: {e} - random weights")

    return model, fdim


# ======================== FEATURE EXTRACTOR =========================
class FeatureExtractor:
    def __init__(self):
        log.info(f"Building OSNet on {Config.DEVICE}...")
        net, fdim = _build_backbone()
        if fdim != Config.FEATURE_DIM:
            log.info(f"Updating FEATURE_DIM: {Config.FEATURE_DIM} -> {fdim}")
            Config.FEATURE_DIM = fdim
        self.fdim = fdim
        self.net  = net.to(Config.DEVICE).eval()
        self.tf   = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        log.info(f"FeatureExtractor ready (dim={self.fdim})")

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(batch), p=2, dim=1)

    @torch.no_grad()
    def __call__(self, crops: list) -> list:
        out     = [None] * len(crops)
        tensors = []
        valid   = []
        for i, c in enumerate(crops):
            if c is None or c.size == 0 or c.shape[0] <= 10 or c.shape[1] <= 5:
                continue
            try:
                t = self.tf(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)).unsqueeze(0)
                tensors.append(t); valid.append(i)
            except Exception:
                continue

        if not tensors:
            return out

        try:
            B  = torch.cat(tensors).to(Config.DEVICE)
            f  = self._forward(B)
            ff = self._forward(torch.flip(B, [3]))
            fs = F.normalize((f + ff) / 2, p=2, dim=1).cpu().numpy()
            for ai, ci in enumerate(valid):
                v = fs[ai].astype(np.float32)
                n = np.linalg.norm(v)
                if n > 1e-6 and not np.isnan(v).any():
                    out[ci] = v / n
        except Exception as e:
            if Config.DEBUG:
                log.debug(f"Embed batch failed: {e}, per-crop fallback")
            for ai, ci in enumerate(valid):
                try:
                    t = tensors[ai].to(Config.DEVICE)
                    v = self._forward(t).cpu().numpy()[0]
                    n = np.linalg.norm(v)
                    if n > 1e-6 and not np.isnan(v).any():
                        out[ci] = (v/n).astype(np.float32)
                except Exception:
                    pass
        return out


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


# ======================== GLOBAL REID MANAGER =========================
class GlobalReIDManager:
    """
    _resolve() pipeline:

      Step A  Match ACTIVE global track
              - skip if cam occupied
              - skip if gallery not matchable (FP-3 fix)
              - use _robust_dist_to_gallery() (FP-1 fix)
              - threshold: SAME_CAM_REENTRY_DIST or CROSS_CAM_DIST (FP-2 fix)
              - pick single best match (lowest distance)

      Step B  Reacquire ARCHIVED track
              - FAISS coarse filter + _robust_dist_to_gallery() verify
              - skip if gallery not matchable (FP-3 fix)
              - threshold: REACQ_DIST (FP-4 fix)

      Step C  Create new global track
    """

    def __init__(self):
        self.db           = DB()
        self.lock         = Lock()
        self.active       : dict = {}
        self.l2g          : dict = {}
        self._last_rebuild = 0.0
        self._last_merge   = 0.0

        for gt in self.db.active_on_load():
            self.active[gt.gid] = gt
            if Config.DEBUG:
                log.debug(f"Restored G{gt.gid} to active "
                          f"(probes={gt.total_probes()})")
        if self.active:
            log.info(f"Restored {len(self.active)} active track(s) from DB")

    def update(self, detections: list, ts: float):
        self.db.cleanup(ts)
        self._expire_old(ts)

        by_key : dict = defaultdict(list)
        for d in detections:
            by_key[(d['cam'], d['local_id'])].append(d)

        for key, dets in by_key.items():
            cam, local_id = key
            feats = [d['feat'] for d in dets if d['feat'] is not None]
            feat  = _norm(np.stack(feats).mean(0)) if feats else None

            if key in self.l2g:
                gid = self.l2g[key]
                gt  = self.active.get(gid)
                if gt is None:
                    del self.l2g[key]
                    gid = self._resolve(cam, local_id, feat, ts)
                    self.l2g[key] = gid
                else:
                    gt.observe(cam, local_id, feat, ts)
                    self.db.upsert(gt)
            else:
                gid = self._resolve(cam, local_id, feat, ts)
                self.l2g[key] = gid

            for d in dets:
                d['gid'] = gid

        # Archive idle tracks
        idle = [gid for gid, gt in self.active.items()
                if ts - gt.last_seen > Config.INACTIVE_TTL]
        for gid in idle:
            gt = self.active.pop(gid)
            gt.archived      = True
            gt.archived_time = ts
            gt.clear_locals()
            self.db.upsert(gt)
            self.db.stats['archived'] += 1
            for k in [k for k, v in self.l2g.items() if v == gid]:
                del self.l2g[k]
            if Config.DEBUG:
                log.debug(f"Archived G{gid}  age={ts-gt.first_seen:.1f}s  "
                          f"dets={gt.det_count}  cams={gt.cams_seen}")

        # Periodic post-hoc merge of active tracks that belong to the same person.
        if ts - self._last_merge >= Config.MERGE_INTERVAL:
            self._merge_active_tracks(ts)
            self._last_merge = ts

        if (ts - self._last_rebuild >= Config.FAISS_REBUILD_INTERVAL
                or self.db._dirty):
            self.db.rebuild_index()
            self._last_rebuild = ts

    def _merge_active_tracks(self, ts: float):
        """
        Post-hoc merge pass — fixes early fragmentation.

        Problem:
          When a person first appears, their gallery has < MIN_PROBES_TO_MATCH
          frames so _resolve() cannot link them to any existing track.  If
          ByteTrack momentarily drops and reassigns a local_id, or the person
          appears on a second camera in the same window, a second global track
          is created for the same person.  Both tracks then accumulate probes
          in parallel and are never reconciled.

        Solution:
          Every MERGE_INTERVAL seconds, compare every pair of ACTIVE global
          tracks whose galleries are both matchable.  If their robust distance
          is below MERGE_DIST AND they share no camera (or the overlapping
          camera's slot cleared), merge the younger/smaller track into the
          older/larger one:

          - All l2g mappings pointing at the loser are repointed to the winner.
          - The loser's _buf is folded into the winner's _buf (capped at
            CAM_FEAT_BUF per camera) and the winner's gallery is rebuilt.
          - The loser is archived immediately so it no longer appears in the
            active set and is eventually cleaned up.

        Merge direction:
          Winner = track with more det_count (better gallery), ties broken by
          lower gid (older track, which is what the user saw first).

        Safety guards:
          - Both tracks must be matchable (enough probes).
          - They must NOT both have an active local tracklet on the same camera
            at this moment (that would mean they are genuinely two people).
          - Distance must be < MERGE_DIST (tighter than CROSS_CAM_DIST).
        """
        gids     = list(self.active.keys())
        merged   : set = set()   # gids that have already been absorbed

        for i in range(len(gids)):
            if gids[i] in merged:
                continue
            gt_a = self.active.get(gids[i])
            if gt_a is None or not gt_a.is_matchable():
                continue

            for j in range(i + 1, len(gids)):
                if gids[j] in merged:
                    continue
                gt_b = self.active.get(gids[j])
                if gt_b is None or not gt_b.is_matchable():
                    continue

                # Safety: if both tracks are currently live on the same camera,
                # they MUST be different people — don't merge.
                cams_a = {c for (c, _), t in gt_a.locals.items()
                          if ts - t < Config.EXPIRE_LOCAL_TTL}
                cams_b = {c for (c, _), t in gt_b.locals.items()
                          if ts - t < Config.EXPIRE_LOCAL_TTL}
                if cams_a & cams_b:
                    continue

                d = _robust_dist_to_gallery(gt_a.feature, gt_b)
                if d > Config.MERGE_DIST:
                    continue

                # Pick winner (more detections wins; tie → lower gid).
                if (gt_a.det_count > gt_b.det_count or
                        (gt_a.det_count == gt_b.det_count and gt_a.gid < gt_b.gid)):
                    winner, loser = gt_a, gt_b
                else:
                    winner, loser = gt_b, gt_a

                loser_gid  = loser.gid
                winner_gid = winner.gid

                # Fold loser's feature buffer into winner's.
                for cam, buf in loser._buf.items():
                    dst = winner._buf[cam]
                    dst.extend(buf)
                    if len(dst) > Config.CAM_FEAT_BUF:
                        # Keep the most recent CAM_FEAT_BUF frames.
                        winner._buf[cam] = dst[-Config.CAM_FEAT_BUF:]
                winner.cams_seen.update(loser.cams_seen)
                winner.det_count += loser.det_count
                winner.first_seen = min(winner.first_seen, loser.first_seen)
                winner._rebuild()

                # Repoint all l2g entries that pointed at loser → winner.
                for key in list(self.l2g.keys()):
                    if self.l2g[key] == loser_gid:
                        self.l2g[key] = winner_gid
                        # Transfer the local tracklet into the winner.
                        cam_k, lid_k = key
                        winner.locals[key] = loser.locals.get(key, ts)

                # Archive the loser silently (no cooldown needed — it's a merge).
                loser.archived      = True
                loser.archived_time = ts
                loser.clear_locals()
                self.active.pop(loser_gid, None)
                self.db.upsert(loser)
                self.db.upsert(winner)
                self.db.stats['merged'] = self.db.stats.get('merged', 0) + 1
                merged.add(loser_gid)

                log.info(f"Merged G{loser_gid} -> G{winner_gid}  "
                         f"dist={d:.3f}  winner_probes={winner.total_probes()}")
                # gt_a may now be the winner — update the reference so the
                # outer loop can keep comparing gt_a against other tracks.
                if winner_gid == gids[i]:
                    gt_a = winner

    def _resolve(self, cam: int, local_id: int, feat, ts: float) -> int:
        """Assign a global ID to a brand-new local tracklet."""

        # ---- STEP A: Match against ACTIVE global tracks ----
        best_gid  = None
        best_dist = 2.0

        if feat is not None:
            for gid, gt in self.active.items():
                if gt.cam_is_occupied(cam, ts):
                    continue
                # FP-3: skip unripe galleries.
                if not gt.is_matchable():
                    continue
                # FP-1: robust distance.
                d = _robust_dist_to_gallery(feat, gt)
                # FP-2: per-scenario threshold.
                same_cam  = cam in gt.cams_seen
                threshold = (Config.SAME_CAM_REENTRY_DIST if same_cam
                             else Config.CROSS_CAM_DIST)
                if d < threshold and d < best_dist:
                    best_dist = d
                    best_gid  = gid

        if best_gid is not None:
            gt = self.active[best_gid]
            gt.observe(cam, local_id, feat, ts)
            self.db.upsert(gt)
            if Config.DEBUG:
                scenario = "same-cam re-entry" if cam in gt.cams_seen else "cross-cam"
                log.debug(f"Link [{scenario}] G{best_gid} <- cam{cam}/L{local_id}  "
                          f"dist={best_dist:.3f}  probes={gt.total_probes()}")
            self.db.stats['links'] += 1
            return best_gid

        # ---- STEP B: Reacquire ARCHIVED global track ----
        if feat is not None:
            for gid, sim, gt in self.db.search_archived(feat):
                if gid in self.active:
                    continue
                # FP-3: skip unripe galleries.
                if not gt.is_matchable():
                    continue
                if (gt.archived_time and
                        ts - gt.archived_time < Config.MERGE_COOLDOWN):
                    continue
                # FP-1 + FP-4: robust verify + tighter threshold.
                robust_d = _robust_dist_to_gallery(feat, gt)
                if robust_d > Config.REACQ_DIST:
                    continue
                gt.clear_locals()
                gt.archived      = False
                gt.archived_time = None
                gt.observe(cam, local_id, feat, ts)
                self.active[gid] = gt
                self.db.upsert(gt)
                self.db.stats['reacquired'] += 1
                if Config.DEBUG:
                    log.debug(f"Reacquired G{gid}  cam{cam}/L{local_id}  "
                              f"faiss_sim={sim:.3f}  robust_dist={robust_d:.3f}")
                return gid

        # ---- STEP C: Create new global track ----
        gt = GlobalTrack(ts)
        gt.observe(cam, local_id, feat, ts)
        self.active[gt.gid] = gt
        self.db.upsert(gt)
        if Config.DEBUG:
            log.debug(f"New G{gt.gid}  cam{cam}/L{local_id}")
        return gt.gid

    def _expire_old(self, ts: float):
        ttl   = Config.EXPIRE_LOCAL_TTL
        stale = [k for k, gid in self.l2g.items()
                 if gid in self.active and
                 ts - self.active[gid].locals.get(k, 0.0) > ttl]
        for k in stale:
            gid = self.l2g.pop(k)
            gt  = self.active.get(gid)
            if gt:
                gt.expire(k[0], k[1])
            if Config.DEBUG:
                log.debug(f"Expired cam{k[0]}/L{k[1]} -> G{gid}")

    def current_cams(self, gt: 'GlobalTrack', ts: float) -> set:
        """Cameras where gt currently has a live local tracklet."""
        return {c for (c, _), t in gt.locals.items()
                if ts - t < Config.EXPIRE_LOCAL_TTL}


# ======================== CAMERA WORKER =========================
class CameraWorker(Thread):
    DEAD_SENTINEL = object()

    def __init__(self, cam_id, cap, yolo_path, feat_ext, out_q):
        super().__init__(daemon=True)
        self.cam_id     = cam_id
        self.cap        = cap
        self.yolo_path  = yolo_path
        self.feat_ext   = feat_ext
        self.out_q      = out_q
        self.running    = True
        self.fail_count = 0
        self.max_fail   = Config.CAM_MAX_FAILURES

    def run(self):
        try:
            from ultralytics import YOLO
            yolo = YOLO(self.yolo_path)
            yolo.to(Config.DEVICE)
        except Exception as e:
            log.error(f"Camera {self.cam_id}: failed to load YOLO "
                      f"'{self.yolo_path}': {e}")
            try:
                self.out_q.put_nowait((self.cam_id, CameraWorker.DEAD_SENTINEL, []))
            except Exception:
                pass
            self.cap.release()
            return

        log.info(f"Camera {self.cam_id} worker started")

        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None or frame.size == 0:
                    self.fail_count += 1
                    if self.fail_count >= self.max_fail:
                        log.error(f"Camera {self.cam_id} failed repeatedly, stopping.")
                        break
                    time.sleep(0.05)
                    continue

                self.fail_count = 0
                frame = cv2.resize(frame, (640, 480))
                h, w  = frame.shape[:2]

                res   = yolo.track(frame, persist=True, classes=[0],
                                   conf=Config.BT_CONF, iou=Config.BT_IOU,
                                   tracker="bytetrack.yaml", verbose=False)
                boxes = res[0].boxes if res else []

                crops, metas = [], []
                for box in boxes:
                    try:
                        if box.id is None:
                            continue
                        lid = int(box.id[0])
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        x1,y1 = max(0,x1), max(0,y1)
                        x2,y2 = min(w,x2), min(h,y2)
                        if x2-x1 <= 20 or y2-y1 <= 40:
                            continue
                        crops.append(frame[y1:y2, x1:x2])
                        metas.append({'cam': self.cam_id, 'local_id': lid,
                                      'bbox': (x1,y1,x2,y2),
                                      'cx': (x1+x2)/(2*w),
                                      'cy': (y1+y2)/(2*h),
                                      'feat': None, 'gid': None})
                    except Exception:
                        continue

                if crops:
                    feats = self.feat_ext(crops)
                    for m, f in zip(metas, feats):
                        m['feat'] = f

                try:
                    self.out_q.put_nowait((self.cam_id, frame, metas))
                except Exception:
                    pass

            except Exception as e:
                log.warning(f"Worker cam{self.cam_id}: {e}")
                time.sleep(0.05)

        self.cap.release()
        log.info(f"Camera {self.cam_id} worker stopped")

    def stop(self):
        self.running = False


# ======================== UI =========================
def _sidebar(w, h):
    s = np.zeros((h,w,3), np.uint8)
    s[:] = (18,18,18)
    return s

_TITLE_CACHE  : np.ndarray | None = None
_FOOTER_CACHE : np.ndarray | None = None

def _title(width: int) -> np.ndarray:
    global _TITLE_CACHE
    if _TITLE_CACHE is not None and _TITLE_CACHE.shape[1] == width:
        return _TITLE_CACHE
    h  = 85
    tb = np.zeros((h, width, 3), np.uint8)
    tb[:] = (18,18,18)
    cv2.line(tb,(0,h-1),(width,h-1),(255,120,60),2)
    cv2.putText(tb,"FoundYou",(30,40),cv2.FONT_HERSHEY_SIMPLEX,1.6,(90,45,20),3)
    cv2.putText(tb,"FoundYou",(28,38),cv2.FONT_HERSHEY_SIMPLEX,1.6,(255,120,60),3)
    cv2.putText(tb,"Multi-Camera ReID",
                (28,68),cv2.FONT_HERSHEY_SIMPLEX,0.65,(150,150,165),2)
    _TITLE_CACHE = tb
    return tb

def _footer(width: int) -> np.ndarray:
    global _FOOTER_CACHE
    if _FOOTER_CACHE is not None and _FOOTER_CACHE.shape[1] == width:
        return _FOOTER_CACHE
    h = 55
    f = np.zeros((h, width, 3), np.uint8)
    f[:] = (18,18,18)
    cv2.line(f,(0,0),(width,0),(255,120,60),2)
    x = 20
    for ctrl in ["[Q] Quit", "[D] Debug", "[S] Stats", "[X] Screenshot"]:
        tw = cv2.getTextSize(ctrl,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)[0][0]
        cv2.rectangle(f,(x-6,12),(x+tw+6,46),(35,35,38),-1)
        cv2.rectangle(f,(x-6,12),(x+tw+6,46),(255,120,60),1)
        cv2.putText(f,ctrl,(x,36),cv2.FONT_HERSHEY_SIMPLEX,0.7,(235,235,245),2)
        x += tw + 45
    _FOOTER_CACHE = f
    return f

CAM_PAD   = 10   # pixels of dark border around each camera frame
ID_STRIP_H = 54  # height of the ID strip below each camera

def _wrap_camera(frame: np.ndarray, cid: int, dets: list,
                 mgr, ts: float, is_dead: bool) -> np.ndarray:
    """
    Surround a camera frame with:
      - CAM_PAD px dark padding on all four sides
      - A dark ID strip below the frame listing every active global ID
        visible on this camera, drawn as coloured pill badges.
    """
    BG   = (22, 22, 22)
    h, w = frame.shape[:2]

    # ---- top/left/right padding ----
    left_pad = np.full((h, CAM_PAD, 3), BG, np.uint8)
    right_pad = np.full((h, CAM_PAD, 3), BG, np.uint8)
    mid_row = np.hstack([left_pad, frame, right_pad])
    padded_w = mid_row.shape[1]
    top_pad = np.full((CAM_PAD, padded_w, 3), BG, np.uint8)  # ← was width w, now padded_w

    # ---- ID strip ----
    strip = np.full((ID_STRIP_H, padded_w, 3), BG, np.uint8)
    # Camera label on the left
    cv2.putText(strip, f"CAM {cid}", (CAM_PAD + 4, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 120, 60), 2)
    if is_dead:
        cv2.putText(strip, "DEAD", (CAM_PAD + 80, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 220), 2)

    # Collect unique GIDs visible on this camera
    gids_visible = []
    seen = set()
    for d in dets:
        gid = d.get('gid')
        if gid is not None and gid not in seen:
            seen.add(gid)
            gids_visible.append(gid)

    # Draw pill badges  "G3"  with the track colour
    bx = CAM_PAD + 115
    for gid in sorted(gids_visible):
        gt = mgr.active.get(gid) or mgr.db.get(gid)
        color = gt.color if gt else (140, 140, 140)
        lbl   = f"G{gid}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        pill_w = tw + 16
        if bx + pill_w > padded_w - CAM_PAD:
            break   # out of space
        # filled pill
        cv2.rectangle(strip, (bx, 10), (bx + pill_w, 10 + th + 12), color, -1)
        # slight dark border
        cv2.rectangle(strip, (bx, 10), (bx + pill_w, 10 + th + 12), (0,0,0), 1)
        cv2.putText(strip, lbl, (bx + 8, 10 + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        bx += pill_w + 8

    # ---- bottom padding ----
    bot_pad = np.full((CAM_PAD, padded_w, 3), BG, np.uint8)

    return np.vstack([top_pad, mid_row, strip, bot_pad])


def _assemble_video(frames_ready: dict, sorted_cids: list,
                    dets_snapshot: dict, mgr, ts: float,
                    dead_workers: set) -> np.ndarray:
    """
    Wrap each camera with padding+ID strip then tile them.
    Layout mirrors the old arrangement: 1, 2 (vertical), 3 (2+1), 4 (2×2).
    """
    wrapped = []
    for cid in sorted_cids:
        frame = frames_ready.get(cid)
        if frame is None:
            frame = np.zeros((480, 640, 3), np.uint8)
        dets    = dets_snapshot.get(cid, [])
        is_dead = cid in dead_workers
        wrapped.append(_wrap_camera(frame, cid, dets, mgr, ts, is_dead))

    # Make all wrapped frames the same size (pad to max dims)
    max_h = max(f.shape[0] for f in wrapped)
    max_w = max(f.shape[1] for f in wrapped)
    BG    = (22, 22, 22)

    def pad_to(f):
        dh = max_h - f.shape[0]
        dw = max_w - f.shape[1]
        if dh > 0:
            f = np.vstack([f, np.full((dh, f.shape[1], 3), BG, np.uint8)])
        if dw > 0:
            f = np.hstack([f, np.full((f.shape[0], dw, 3), BG, np.uint8)])
        return f

    wrapped = [pad_to(f) for f in wrapped]
    n = len(wrapped)

    if   n == 1:
        return wrapped[0]
    elif n == 2:
        return np.vstack(wrapped)
    elif n == 3:
        top = np.hstack(wrapped[:2])
        bot = pad_to(wrapped[2])
        # centre the lone bottom camera
        side = (top.shape[1] - bot.shape[1]) // 2
        lp   = np.full((bot.shape[0], side,                              3), BG, np.uint8)
        rp   = np.full((bot.shape[0], top.shape[1] - bot.shape[1] - side, 3), BG, np.uint8)
        return np.vstack([top, np.hstack([lp, bot, rp])])
    else:
        return np.vstack([np.hstack(wrapped[:2]), np.hstack(wrapped[2:4])])


def _draw_sidebar(sb, mgr, ts, fps):
    sb[:] = (18,18,18)
    y  = 26
    lh = 24

    def row(s, col=(220,220,232), sc=0.52):
        nonlocal y
        cv2.putText(sb,s,(16,y),cv2.FONT_HERSHEY_SIMPLEX,sc,col,1)
        y += lh

    def hdr(s, col=(255,120,60)):
        nonlocal y
        cv2.line(sb,(0,y-4),(sb.shape[1],y-4),(40,40,40),1)
        cv2.putText(sb,s,(12,y+14),cv2.FONT_HERSHEY_SIMPLEX,0.65,col,2)
        y += 30

    hdr("SYSTEM")
    for s in [f"Runtime : {int(ts)}s",
              f"FPS     : {fps:.1f}",
              f"Active  : {len(mgr.active)} global IDs",
              f"DB total: {len(mgr.db.tracks)}",
              f"Max GID : {GlobalTrack._nxt-1}"]:
        row(s)

    y += 4
    hdr("THRESHOLDS")
    for s in [f"Cross-cam:        dist < {Config.CROSS_CAM_DIST:.2f}",
              f"Same-cam reentry: dist < {Config.SAME_CAM_REENTRY_DIST:.2f}",
              f"Reacquisition:    dist < {Config.REACQ_DIST:.2f}",
              f"Active merge:     dist < {Config.MERGE_DIST:.2f}",
              f"Min probes:       {Config.MIN_PROBES_TO_MATCH}"]:
        row(s,(100,255,120),0.42)

    y += 4
    hdr("STATS")
    st = mgr.db.stats
    for k in ['created','links','reacquired','merged','archived','cleaned']:
        row(f"{k:<14}: {st.get(k,0)}")

    y += 4
    hdr("ACTIVE IDs")
    gts   = sorted(mgr.active.values(), key=lambda g: g.gid)
    avail = max(1, (sb.shape[0] - y - 12) // lh)
    for gt in gts[:avail]:
        if y > sb.shape[0]-12:
            break
        cv2.rectangle(sb,(16,y-12),(30,y-2),gt.color,-1)
        cv2.rectangle(sb,(16,y-12),(30,y-2),(255,120,60),1)
        cur_cams = mgr.current_cams(gt, ts)
        cams     = ','.join(str(c) for c in sorted(cur_cams)) if cur_cams else '-'
        age      = int(ts - gt.first_seen)
        flag     = '' if gt.is_matchable() else ' [warming]'
        cv2.putText(sb,
            f"G{gt.gid}  C[{cams}]  {age}s  p={gt.total_probes()}{flag}",
            (36,y),cv2.FONT_HERSHEY_SIMPLEX,0.40,(220,220,232),1)
        y += lh
    if len(gts) > avail:
        row(f"... +{len(gts)-avail} more",(110,110,125))
    return sb


# ======================== MAIN =========================
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
    parser.add_argument('--device', type=str, choices=['cpu','cuda','mps'], dest='DEVICE')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def run():
    args = parse_args()
    Config.override_from_args(args)
    if args.debug:
        Config.DEBUG = True

    feat_ext = FeatureExtractor()
    mgr      = GlobalReIDManager()

    caps = {}
    log.info("Scanning cameras...")
    sys_     = platform.system()
    backends = {'Darwin':  [cv2.CAP_AVFOUNDATION],
                'Windows': [cv2.CAP_DSHOW, cv2.CAP_ANY]
               }.get(sys_, [cv2.CAP_V4L2, cv2.CAP_ANY])

    found = set()
    for backend in backends:
        for i in range(3):
            if i in found:
                continue
            try:
                c = cv2.VideoCapture(i, backend)
                if c.isOpened():
                    ret, fr = c.read()
                    if ret and fr is not None and fr.size > 0:
                        c.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        caps[i] = c
                        found.add(i)
                        bn = {cv2.CAP_AVFOUNDATION:"AVFoundation",
                              cv2.CAP_DSHOW:"DirectShow",
                              cv2.CAP_V4L2:"V4L2",
                              cv2.CAP_ANY:"Auto"}.get(backend, str(backend))
                        log.info(f"Camera {i} ({bn})")
                    else:
                        c.release()
            except Exception:
                continue
        if found:
            break

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

    result_q       = Queue(maxsize=60)
    workers        = []
    for cid, cap in caps.items():
        w = CameraWorker(cid, cap, Config.YOLO_MODEL, feat_ext, result_q)
        w.start()
        workers.append(w)

    t0             = time.time()
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

            frames_ready = {}
            with last_dets_lock:
                dets_snapshot = {k: list(v) for k, v in last_dets.items()}

            # Draw detections onto each camera frame (no padding here — that
            # happens in _wrap_camera / _assemble_video below).
            for cid, frame in frames_data.items():
                if frame is None:
                    ph = np.zeros((480, 640, 3), np.uint8)
                    frames_ready[cid] = ph
                    continue

                disp = frame.copy()

                for d in dets_snapshot.get(cid, []):
                    x1,y1,x2,y2 = d['bbox']
                    gid          = d.get('gid')
                    if gid is None:
                        cv2.rectangle(disp,(x1,y1),(x2,y2),(80,80,80),2)
                        continue
                    gt = mgr.active.get(gid) or mgr.db.get(gid)
                    if not gt:
                        continue

                    cv2.rectangle(disp,(x1,y1),(x2,y2),gt.color,3)
                    cur_cams = mgr.current_cams(gt, ts)
                    lbl      = f"G{gid}"
                    if len(cur_cams) > 1:
                        lbl += f" [C{','.join(str(c) for c in sorted(cur_cams))}]"
                    if not gt.is_matchable():
                        lbl += f" [{gt.total_probes()}/{Config.MIN_PROBES_TO_MATCH}]"

                    (lw,lh_),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.65,2)
                    cv2.rectangle(disp,(x1,y1-lh_-10),(x1+lw+10,y1),gt.color,-1)
                    cv2.putText(disp,lbl,(x1+5,y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)

                frames_ready[cid] = disp

            now = time.time()
            ftimes.append(now - last_ft)
            if len(ftimes) > 30:
                ftimes.pop(0)
            fps     = 1.0 / (sum(ftimes)/len(ftimes) + 1e-9)
            last_ft = now

            sorted_cids = sorted(frames_ready.keys())
            if not sorted_cids:
                time.sleep(0.001)
                continue

            # Assemble camera tiles with padding + ID strips
            video = _assemble_video(frames_ready, sorted_cids,
                                    dets_snapshot, mgr, ts, dead_workers)

            sb   = _sidebar(420, video.shape[0])
            sb   = _draw_sidebar(sb, mgr, ts, fps)
            body = np.hstack([video, sb])
            disp = np.vstack([_title(body.shape[1]), body, _footer(body.shape[1])])
            if disp.shape[1] > 2400:
                s = 2400 / disp.shape[1]
                disp = cv2.resize(disp, (0,0), fx=s, fy=s)

            cv2.imshow("FoundYou", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                Config.DEBUG = not Config.DEBUG
                log.info(f"Debug {'ON' if Config.DEBUG else 'OFF'}")
            elif key == ord('s'):
                with mgr.lock:
                    log.info(f"\n{'='*55}\n t={ts:.0f}s  fps={fps:.1f}")
                    log.info(f"  active IDs : {sorted(mgr.active.keys())}")
                    log.info(f"  l2g        : {dict(mgr.l2g)}")
                    for k,v in mgr.db.stats.items():
                        log.info(f"  {k:<14}: {v}")
                    log.info('='*55)
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
        with mgr.lock:
            with GlobalTrack._nxt_lock:
                next_id = GlobalTrack._nxt
            mgr.db.save(next_id)
        for c in caps.values():
            c.release()
        cv2.destroyAllWindows()
        log.info("Done.")


if __name__ == "__main__":
    run()
