"""
FoundYou — Configuration
========================
Single source of truth for all thresholds and constants.
No project imports — safe to import from anywhere.
"""

import torch

from logger import log

# ======================== CONFIGURATION =========================
class Config:
    # ----- Matching thresholds (cosine DISTANCE = 1 - similarity) -----

    # cross-camera threshold.
    CROSS_CAM_DIST = 0.40

    # same-camera re-entry threshold (tight — appearance barely changes).
    SAME_CAM_REENTRY_DIST = 0.30

    # archive reacquisition threshold, tightened from 0.50 to 0.40.
    REACQ_DIST = 0.40
    # FAISS inner-product pre-filter floor (1 - REACQ_DIST).
    REACQ_MIN_SIM = 0.60

    # minimum probes before a gallery is allowed to claim new tracklets.
    MIN_PROBES_TO_MATCH = 3

    # Post-hoc merge: two active tracks whose robust distance falls below this
    # threshold are collapsed into one (the older/higher-det-count track wins).
    # This is the fix for early fragmentation — two IDs assigned to the same
    # person before either gallery was matchable are corrected once both
    # galleries have enough probes to make a reliable comparison.
    # Set slightly tighter than CROSS_CAM_DIST so we only merge when genuinely
    # confident, not just within the cross-camera tolerance band.
    MERGE_DIST = 0.35

    # How often (seconds) to run the active-track merge pass.
    MERGE_INTERVAL = 2.0

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
    YOLO_MODEL = 'yolo26s.pt'

    # ----- Camera handling ---------------------------------------------
    CAM_MAX_FAILURES = 30

    # Cameras whose raw frames should be horizontally flipped before any
    # processing. Set to a list of camera IDs, e.g. [1] or [0, 2].
    # This corrects the embedding so ReID matches across cameras correctly.
    FLIP_CAMERAS: list = []

    # ----- ID validation ---------------------------------------------------
    # Enable runtime consistency checks on global ID assignments.
    ID_VALIDATION = True

    # How often (seconds) to run the validation sweep.
    ID_VALIDATION_INTERVAL = 3.0

    # If two active tracks on DIFFERENT cameras have a robust gallery distance
    # below this threshold they are likely the same person with a swapped ID.
    ID_SWAP_DIST = 0.28

    # Minimum probes both tracks must have before swap detection runs.
    ID_SWAP_MIN_PROBES = 6

    # ----- Face detection (optional pre-ReID step) ---------------------
    # Master switch — set False to skip all face processing entirely.
    FACE_DETECTION       : bool  = True

    # Whether to extract a face embedding (requires insightface or dlib).
    # When False, face detection still runs but only gates on presence/quality.
    FACE_EMBEDDING       : bool  = True

    # Fraction of the top of the body crop to search for a face.
    # 0.50 = upper half (safe for most aspect ratios).
    FACE_SEARCH_FRACTION : float = 0.50

    # Minimum detector confidence to consider a detection real.
    FACE_MIN_CONF        : float = 0.55

    # Minimum quality score (sharpness × size × aspect) to use for matching.
    FACE_MIN_QUALITY     : float = 0.25

    # Cosine distance above which two face embeddings VETO a body-ReID match.
    # Only fires when both detections have usable face embeddings.
    # Set high (e.g. 0.80) to make the veto rare; lower for more aggression.
    FACE_VETO_DIST       : float = 0.55

    # Weight of face distance in the combined score (0 = ignore face entirely).
    # combined = (1 - FACE_WEIGHT) * body_dist + FACE_WEIGHT * face_dist
    # Only applied when both sides have usable face embeddings.
    FACE_WEIGHT          : float = 0.30

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
