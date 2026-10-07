"""
FoundYou — Global ReID Manager
================================
Orchestrates the three-step resolve pipeline and periodic maintenance.

  _resolve() pipeline for every NEW local tracklet:

    Step A  Match ACTIVE global track
    Step B  Reacquire ARCHIVED track
    Step C  Create new global track

  _merge_active_tracks() — periodic post-hoc merge pass that collapses
  tracks fragmented before their galleries were matchable.
"""

from collections import defaultdict
from threading import Lock
import numpy as np

from config import Config
from features import _norm, _robust_dist_to_gallery
from tracks import GlobalTrack, DB
from face import face_gate, face_distance, FaceResult, NO_FACE
from logger import log

# ------------------------------------------------------------------ #
# Face gallery helper                                                 #
# ------------------------------------------------------------------ #

def _update_track_face(gt: 'GlobalTrack', face: 'FaceResult') -> None:
    """
    Persist the best (highest-quality) usable face result on a track.

    We keep a single 'best_face' rather than a gallery because:
      - Face detection is optional and coverage is low (person may not
        face the camera most of the time).
      - A single high-quality face embedding is more reliable than a
        mean of low-quality ones.

    Called by _resolve() and update() whenever a detection is linked
    to a track.
    """
    if face is None or not face.usable:
        return
    current = getattr(gt, 'best_face', None)
    if current is None or face.quality > current.quality:
        gt.best_face = face


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

            # Best face result for this tracklet key (highest quality wins).
            face_results = [d.get('face', NO_FACE) for d in dets]
            face = max(face_results, key=lambda r: r.quality if r.found else -1)

            if key in self.l2g:
                gid = self.l2g[key]
                gt  = self.active.get(gid)
                if gt is None:
                    del self.l2g[key]
                    gid = self._resolve_with_face(cam, local_id, feat, face, ts)
                    self.l2g[key] = gid
                else:
                    gt.observe(cam, local_id, feat, ts)
                    # Store best face result on the track for future comparisons.
                    _update_track_face(gt, face)
                    self.db.upsert(gt)
            else:
                gid = self._resolve_with_face(cam, local_id, feat, face, ts)
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
        """
        Legacy-compatible public entry point.

        SessionRecorder monkey-patches this method, so its signature must
        stay as (cam, local_id, feat, ts).  All face-aware logic lives in
        _resolve_with_face(); this shim calls it with NO_FACE so the
        recorder's patched version never sees the extra argument.
        """
        return self._resolve_with_face(cam, local_id, feat, NO_FACE, ts)

    def _resolve_with_face(self, cam: int, local_id: int, feat,
                           face: 'FaceResult', ts: float) -> int:
        """Assign a global ID to a brand-new local tracklet (face-aware)."""

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
                # FP-1: robust body distance.
                body_d = _robust_dist_to_gallery(feat, gt)
                # FP-2: per-scenario threshold.
                same_cam  = cam in gt.cams_seen
                threshold = (Config.SAME_CAM_REENTRY_DIST if same_cam
                             else Config.CROSS_CAM_DIST)
                if body_d >= threshold:
                    continue

                # --- Face veto ---
                # If both this detection and the track have a usable face
                # embedding and the faces are clearly different, skip this
                # match regardless of body similarity.
                gt_face = getattr(gt, 'best_face', NO_FACE)
                if not face_gate(face, gt_face):
                    if Config.DEBUG:
                        log.debug(f"Face VETO: G{gid} body_d={body_d:.3f} "
                                  f"face_d={face_distance(face, gt_face):.3f}")
                    continue

                # --- Combined score ---
                # When face embeddings are available, blend body + face distance.
                # This sharpens cross-camera linking without over-trusting faces.
                if (face.usable and getattr(gt, 'best_face', NO_FACE).usable):
                    face_d  = face_distance(face, gt_face)
                    w       = Config.FACE_WEIGHT
                    combined = (1 - w) * body_d + w * face_d
                else:
                    combined = body_d

                if combined < threshold and combined < best_dist:
                    best_dist = combined
                    best_gid  = gid

        if best_gid is not None:
            gt = self.active[best_gid]
            gt.observe(cam, local_id, feat, ts)
            _update_track_face(gt, face)
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
                # Face veto on reacquisition too.
                gt_face = getattr(gt, 'best_face', NO_FACE)
                if not face_gate(face, gt_face):
                    if Config.DEBUG:
                        log.debug(f"Face VETO reacq: G{gid} body_d={robust_d:.3f}")
                    continue
                gt.clear_locals()
                gt.archived      = False
                gt.archived_time = None
                gt.observe(cam, local_id, feat, ts)
                _update_track_face(gt, face)
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
        _update_track_face(gt, face)
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