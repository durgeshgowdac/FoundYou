"""
FoundYou — Runtime ID Validator
=================================
Detects and reports ID assignment problems while the system is running.

Problems detected
-----------------
  SWAP      Two active tracks on different cameras are probably the same
            person — their gallery embeddings are very close but they have
            different global IDs.  Root cause: flipped camera, low probe
            count at match time, or threshold too loose.

  DUPLICATE One camera has two active tracks that are too similar to each
            other — likely ByteTrack split one person into two local tracks
            and the ReID manager didn't merge them yet.

  UNSTABLE  A track's intra-gallery cosine variance is high — its stored
            embeddings are inconsistent, meaning it may have accumulated
            frames from more than one person.

Usage
-----
  Instantiate once and call check() on every MERGE_INTERVAL tick:

    validator = IDValidator(mgr)
    # in main loop, periodically:
    issues = validator.check(ts)
    for issue in issues:
        log.warning(issue.summary())

  Or wire it into the UI — ui.py calls draw_validation_overlay() to
  render issues directly onto the sidebar.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import Config
from features import _dist, _robust_dist_to_gallery
from logger import log


# ================================================================
# Issue types
# ================================================================

class IssueKind(Enum):
    SWAP      = "SWAP"       # two IDs are probably the same person
    DUPLICATE = "DUPLICATE"  # one camera has two IDs for one person
    UNSTABLE  = "UNSTABLE"   # one track has inconsistent appearance


@dataclass
class ValidationIssue:
    kind      : IssueKind
    gid_a     : int
    gid_b     : Optional[int]   # None for UNSTABLE
    dist      : float           # the distance that triggered the issue
    cam_a     : Optional[int]   # relevant camera(s)
    cam_b     : Optional[int]
    ts        : float
    # How many times this exact pair has been flagged consecutively.
    # Low counts = transient; high counts = persistent problem.
    streak    : int = 1

    def summary(self) -> str:
        if self.kind == IssueKind.SWAP:
            return (f"[{self.kind.value}] G{self.gid_a}(cam{self.cam_a}) ↔ "
                    f"G{self.gid_b}(cam{self.cam_b})  dist={self.dist:.3f}  "
                    f"streak={self.streak}")
        if self.kind == IssueKind.DUPLICATE:
            return (f"[{self.kind.value}] G{self.gid_a} and G{self.gid_b} "
                    f"both on cam{self.cam_a}  dist={self.dist:.3f}  "
                    f"streak={self.streak}")
        return (f"[{self.kind.value}] G{self.gid_a}  "
                f"intra_var={self.dist:.3f}  streak={self.streak}")

    @property
    def is_persistent(self) -> bool:
        """True if the issue has appeared in 3+ consecutive checks."""
        return self.streak >= 3

    @property
    def severity(self) -> str:
        if self.streak >= 5:
            return "HIGH"
        if self.streak >= 3:
            return "MEDIUM"
        return "LOW"


# ================================================================
# Validator
# ================================================================

class IDValidator:
    """
    Periodic ID consistency checker.

    Runs independently of the merge pass — it only reports, never mutates
    state.  The merge pass in reid.py is responsible for actually fixing
    swaps.  The validator's job is to surface issues that the merge pass
    missed (e.g. because MIN_PROBES_TO_MATCH hasn't been reached yet, or
    because MERGE_DIST is set too tight).
    """

    def __init__(self, mgr):
        self.mgr          = mgr
        self._last_check  = 0.0
        # streak tracking: issue_key → streak count
        self._streaks     : Dict[tuple, int] = defaultdict(int)
        # all issues from the last check
        self.last_issues  : List[ValidationIssue] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def check(self, ts: float) -> List[ValidationIssue]:
        """
        Run all checks. Returns a list of ValidationIssue objects.
        Only runs if at least ID_VALIDATION_INTERVAL seconds have elapsed.
        """
        if not Config.ID_VALIDATION:
            return []
        if ts - self._last_check < Config.ID_VALIDATION_INTERVAL:
            return self.last_issues
        self._last_check = ts

        issues = []
        issues.extend(self._check_swaps(ts))
        issues.extend(self._check_duplicates(ts))
        issues.extend(self._check_unstable(ts))

        # Update streaks
        current_keys = {self._issue_key(i) for i in issues}
        new_streaks  = defaultdict(int)
        for i in issues:
            k = self._issue_key(i)
            new_streaks[k] = self._streaks[k] + 1
            i.streak = new_streaks[k]
        self._streaks = new_streaks

        # Log persistent issues
        for i in issues:
            if i.is_persistent:
                log.warning(f"[IDValidator] {i.summary()}")
            elif Config.DEBUG:
                log.debug(f"[IDValidator] {i.summary()}")

        self.last_issues = issues
        return issues

    def force_check(self, ts: float) -> List[ValidationIssue]:
        """Run immediately regardless of interval."""
        self._last_check = 0.0
        return self.check(ts)

    # ------------------------------------------------------------------ #
    # Swap detection                                                       #
    # ------------------------------------------------------------------ #

    def _check_swaps(self, ts: float) -> List[ValidationIssue]:
        """
        SWAP: two tracks on different cameras are very close in embedding
        space — they should be the same global ID but aren't.

        This is the primary symptom of a flipped-camera problem: the
        embedding of the flipped camera diverges from the correct one,
        the ReID match fails, and a second ID is created.

        We catch it by comparing every pair of active tracks that:
          - Are on different cameras right now
          - Both have >= ID_SWAP_MIN_PROBES probes
          - Have robust gallery distance < ID_SWAP_DIST
        """
        issues   = []
        active   = list(self.mgr.active.values())
        min_prob = Config.ID_SWAP_MIN_PROBES

        for i in range(len(active)):
            gt_a = active[i]
            if gt_a.total_probes() < min_prob:
                continue
            cams_a = self.mgr.current_cams(gt_a, ts)

            for j in range(i + 1, len(active)):
                gt_b = active[j]
                if gt_b.total_probes() < min_prob:
                    continue
                cams_b = self.mgr.current_cams(gt_b, ts)

                # Must be on different cameras right now
                if not cams_a or not cams_b:
                    continue
                if cams_a & cams_b:
                    continue   # same cam → could be two different people

                d = _robust_dist_to_gallery(gt_a.feature, gt_b)
                if d < Config.ID_SWAP_DIST:
                    issues.append(ValidationIssue(
                        kind  = IssueKind.SWAP,
                        gid_a = gt_a.gid,
                        gid_b = gt_b.gid,
                        dist  = d,
                        cam_a = next(iter(cams_a)),
                        cam_b = next(iter(cams_b)),
                        ts    = ts,
                    ))

        return issues

    # ------------------------------------------------------------------ #
    # Duplicate detection                                                  #
    # ------------------------------------------------------------------ #

    def _check_duplicates(self, ts: float) -> List[ValidationIssue]:
        """
        DUPLICATE: two active tracks are both live on the same camera and
        their embeddings are very close — ByteTrack split one person into
        two local tracklets that the merge pass hasn't collapsed yet.

        Threshold is tighter than SWAP because same-camera tracks that are
        truly different people will appear at different positions, and
        ByteTrack usually keeps them separate.  We use MERGE_DIST here
        since the merge pass should have caught this already — if it didn't,
        that's the anomaly we want to surface.
        """
        issues = []
        active = list(self.mgr.active.values())

        for i in range(len(active)):
            gt_a  = active[i]
            if gt_a.total_probes() < Config.ID_SWAP_MIN_PROBES:
                continue
            cams_a = self.mgr.current_cams(gt_a, ts)

            for j in range(i + 1, len(active)):
                gt_b  = active[j]
                if gt_b.total_probes() < Config.ID_SWAP_MIN_PROBES:
                    continue
                cams_b = self.mgr.current_cams(gt_b, ts)

                shared = cams_a & cams_b
                if not shared:
                    continue

                d = _robust_dist_to_gallery(gt_a.feature, gt_b)
                if d < Config.MERGE_DIST:
                    for cam in shared:
                        issues.append(ValidationIssue(
                            kind  = IssueKind.DUPLICATE,
                            gid_a = gt_a.gid,
                            gid_b = gt_b.gid,
                            dist  = d,
                            cam_a = cam,
                            cam_b = cam,
                            ts    = ts,
                        ))

        return issues

    # ------------------------------------------------------------------ #
    # Unstable gallery detection                                           #
    # ------------------------------------------------------------------ #

    def _check_unstable(self, ts: float) -> List[ValidationIssue]:
        """
        UNSTABLE: a track's per-camera gallery has high intra-variance,
        suggesting its buffer has accumulated frames from more than one
        person — usually caused by an ID switch that the tracker didn't
        catch, or a very crowded scene where bboxes overlapped.

        Intra-variance = mean pairwise cosine distance within the buffer.
        A well-lit, consistent person should be < 0.15.
        Above 0.30 is a strong signal of contamination.
        """
        UNSTABLE_THRESHOLD = 0.30
        issues = []

        for gt in self.mgr.active.values():
            if gt.total_probes() < Config.ID_SWAP_MIN_PROBES:
                continue

            cam_vars = []
            for cam, buf in gt._buf.items():
                if len(buf) < 3:
                    continue
                dists = []
                for ii in range(len(buf)):
                    for jj in range(ii + 1, len(buf)):
                        dists.append(_dist(buf[ii], buf[jj]))
                if dists:
                    cam_vars.append(float(np.mean(dists)))

            if not cam_vars:
                continue

            mean_var = float(np.mean(cam_vars))
            if mean_var > UNSTABLE_THRESHOLD:
                issues.append(ValidationIssue(
                    kind  = IssueKind.UNSTABLE,
                    gid_a = gt.gid,
                    gid_b = None,
                    dist  = mean_var,
                    cam_a = None,
                    cam_b = None,
                    ts    = ts,
                ))

        return issues

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _issue_key(issue: ValidationIssue) -> tuple:
        """Stable key for streak tracking regardless of gid_a/gid_b order."""
        pair = tuple(sorted(filter(None, [issue.gid_a, issue.gid_b])))
        return (issue.kind, pair, issue.cam_a, issue.cam_b)

    # ------------------------------------------------------------------ #
    # Report                                                               #
    # ------------------------------------------------------------------ #

    def report(self) -> str:
        """Return a human-readable summary of the last check's issues."""
        if not self.last_issues:
            return "IDValidator: no issues detected."
        lines = [f"IDValidator: {len(self.last_issues)} issue(s)"]
        for i in self.last_issues:
            lines.append(f"  [{i.severity}] {i.summary()}")
        return "\n".join(lines)

    def swap_pairs(self) -> List[Tuple[int, int, float]]:
        """Return (gid_a, gid_b, dist) for all current SWAP issues."""
        return [(i.gid_a, i.gid_b, i.dist)
                for i in self.last_issues if i.kind == IssueKind.SWAP]

    def has_issues(self) -> bool:
        return bool(self.last_issues)

    def persistent_issues(self) -> List[ValidationIssue]:
        return [i for i in self.last_issues if i.is_persistent]