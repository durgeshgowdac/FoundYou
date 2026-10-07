"""
FoundYou — Face Detection
==========================
Optional face detection step that runs BEFORE the ReID feature extractor.

Why:
  Body-crop ReID embeddings are robust but sometimes confused by clothing
  similarity or partial occlusions. When a face is visible and well-lit,
  adding a face-similarity signal can sharpen cross-camera linking and
  dramatically reduce ID swaps.

Design:
  - Uses OpenCV's DNN face detector (res10_300x300_ssd — ships with cv2)
    as the primary backend. Falls back to Haar cascades if unavailable.
  - FaceResult is attached to each detection's metadata dict as 'face'.
  - The ReID manager uses face distance as a *veto* and *tie-breaker*,
    NOT as the primary signal — body embedding still leads.

Attach / usage pattern in camera.py:
    face_det = FaceDetector()
    ...
    for m, crop in zip(metas, crops):
        m['face'] = face_det.detect(crop)

Then in reid.py / features.py:
    face_dist = face_distance(det['face'], gt.face_gallery)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from config import Config
from logger import log


# ------------------------------------------------------------------ #
# Data model                                                          #
# ------------------------------------------------------------------ #

@dataclass
class FaceResult:
    """Outcome of attempting to detect a face in a person crop."""

    found    : bool                     = False
    bbox     : Optional[Tuple[int,...]] = None  # (x1,y1,x2,y2) in crop coords
    conf     : float                    = 0.0   # detector confidence [0,1]
    quality  : float                    = 0.0   # estimated face quality  [0,1]
    embedding: Optional[np.ndarray]     = None  # unit-norm embedding if extracted

    @property
    def usable(self) -> bool:
        """True if this result is reliable enough to influence matching."""
        return (self.found
                and self.conf  >= Config.FACE_MIN_CONF
                and self.quality >= Config.FACE_MIN_QUALITY
                and self.embedding is not None)

    def to_dict(self) -> dict:
        return {
            'found'    : self.found,
            'bbox'     : self.bbox,
            'conf'     : round(self.conf, 4),
            'quality'  : round(self.quality, 4),
            'has_embed': self.embedding is not None,
        }


NO_FACE = FaceResult()   # shared sentinel — saves allocations


# ------------------------------------------------------------------ #
# Quality estimator                                                   #
# ------------------------------------------------------------------ #

def _estimate_quality(face_crop: np.ndarray) -> float:
    """
    Fast heuristic quality score in [0, 1].

    Combines:
      - Laplacian variance  (sharpness — low = blurry)
      - Face crop area      (small crops have noisy embeddings)
      - Aspect ratio        (extreme ratios suggest a bad bbox)

    Each component is clamped and normalised then averaged.
    No ML model needed — runs in microseconds.
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0

    h, w = face_crop.shape[:2]

    # --- sharpness ---
    gray    = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharp   = float(np.clip(lap_var / 300.0, 0.0, 1.0))   # 300 ≈ "sharp enough"

    # --- size ---
    area    = h * w
    size_s  = float(np.clip(area / (60 * 60), 0.0, 1.0))  # 60×60 px = baseline

    # --- aspect ratio (face is roughly 0.6–1.4) ---
    ratio   = h / max(w, 1)
    ar_s    = float(np.clip(1.0 - abs(ratio - 0.9) / 0.9, 0.0, 1.0))

    return round((sharp * 0.5 + size_s * 0.3 + ar_s * 0.2), 4)


# ------------------------------------------------------------------ #
# Detector backends                                                   #
# ------------------------------------------------------------------ #

class _DNNFaceDetector:
    """
    OpenCV DNN res10_300x300_ssd — very fast, reasonably accurate.

    The model files are pulled from opencv_extra / opencv_models if
    available in the system, or downloaded from GitHub on first use.
    """
    _PROTO  = "deploy.prototxt"
    _MODEL  = "res10_300x300_ssd_iter_140000.caffemodel"
    _CACHE  = os.path.expanduser("~/.cache/foundyou/face_dnn/")
    _PROTO_URL  = ("https://raw.githubusercontent.com/opencv/opencv/master/"
                   "samples/dnn/face_detector/deploy.prototxt")
    _MODEL_URL  = ("https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
                   "dnn_samples_face_detector_20170830/"
                   "res10_300x300_ssd_iter_140000.caffemodel")

    def __init__(self):
        self.net = None
        self._load()

    def _load(self):
        os.makedirs(self._CACHE, exist_ok=True)
        proto = os.path.join(self._CACHE, self._PROTO)
        model = os.path.join(self._CACHE, self._MODEL)

        if not os.path.exists(proto):
            self._download(self._PROTO_URL, proto)
        if not os.path.exists(model):
            self._download(self._MODEL_URL, model)

        if os.path.exists(proto) and os.path.exists(model):
            try:
                self.net = cv2.dnn.readNetFromCaffe(proto, model)
                log.info("Face DNN detector loaded (res10_300x300_ssd)")
            except Exception as e:
                log.warning(f"Face DNN load failed: {e}")
        else:
            log.warning("Face DNN model files missing — detector unavailable")

    @staticmethod
    def _download(url: str, dst: str):
        try:
            import urllib.request
            log.info(f"Downloading {os.path.basename(dst)}...")
            urllib.request.urlretrieve(url, dst)
        except Exception as e:
            log.warning(f"Download failed for {os.path.basename(dst)}: {e}")

    def detect(self, bgr: np.ndarray) -> list[tuple[float, tuple]]:
        """
        Returns list of (confidence, (x1,y1,x2,y2)) in image coords.
        Sorted descending by confidence.
        """
        if self.net is None:
            return []
        h, w = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(bgr, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        results = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < 0.3:
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 > 10 and y2 - y1 > 10:
                results.append((conf, (x1, y1, x2, y2)))

        return sorted(results, key=lambda r: r[0], reverse=True)

    @property
    def available(self) -> bool:
        return self.net is not None


class _HaarFaceDetector:
    """
    Fallback: OpenCV Haar cascade — always available, less accurate.
    Used only when DNN model files cannot be loaded/downloaded.
    """
    def __init__(self):
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cc = cv2.CascadeClassifier(xml)
        log.info("Face Haar cascade loaded (fallback)")

    def detect(self, bgr: np.ndarray) -> list[tuple[float, tuple]]:
        gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces  = self.cc.detectMultiScale(gray, 1.1, 5, minSize=(24, 24))
        result = []
        if len(faces):
            for (x, y, w, h) in faces:
                result.append((0.6, (x, y, x + w, y + h)))
        return result

    @property
    def available(self) -> bool:
        return not self.cc.empty()


# ------------------------------------------------------------------ #
# Face embedder (lightweight — optional)                              #
# ------------------------------------------------------------------ #

class _FaceEmbedder:
    """
    Wraps a compact face recognition model.

    Priority:
      1. insightface ArcFace  (best quality, pip install insightface)
      2. dlib face_recognition (good quality, requires dlib)
      3. None                  (no embedding, detection-only mode)

    When no embedder is available the FaceDetector still returns
    FaceResult.found=True with quality/conf — the body ReID embedding
    continues to carry all identity information and the face result is
    used only as a gating signal (e.g. "are they facing the camera?").
    """
    def __init__(self):
        self._embed_fn = None
        self._try_insightface()
        if self._embed_fn is None:
            self._try_dlib()
        if self._embed_fn is None:
            log.info("No face embedder available — face detection only (no face ReID)")

    def _try_insightface(self):
        try:
            import insightface
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name='buffalo_sc',
                               providers=['CUDAExecutionProvider',
                                          'CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(128, 128))

            def _embed(bgr_face: np.ndarray) -> np.ndarray | None:
                faces = app.get(bgr_face)
                if not faces:
                    return None
                e = faces[0].normed_embedding
                n = np.linalg.norm(e)
                return (e / n).astype(np.float32) if n > 1e-6 else None

            self._embed_fn = _embed
            log.info("Face embedder: insightface ArcFace")
        except Exception:
            pass

    def _try_dlib(self):
        try:
            import dlib
            import face_recognition

            def _embed(bgr_face: np.ndarray) -> np.ndarray | None:
                rgb = bgr_face[:, :, ::-1]
                encs = face_recognition.face_encodings(rgb)
                if not encs:
                    return None
                e = np.array(encs[0], dtype=np.float32)
                n = np.linalg.norm(e)
                return (e / n).astype(np.float32) if n > 1e-6 else None

            self._embed_fn = _embed
            log.info("Face embedder: dlib face_recognition")
        except Exception:
            pass

    def embed(self, face_crop: np.ndarray) -> np.ndarray | None:
        if self._embed_fn is None or face_crop is None or face_crop.size == 0:
            return None
        try:
            return self._embed_fn(face_crop)
        except Exception:
            return None

    @property
    def available(self) -> bool:
        return self._embed_fn is not None


# ------------------------------------------------------------------ #
# Public FaceDetector                                                 #
# ------------------------------------------------------------------ #

class FaceDetector:
    """
    Unified face detection + optional embedding interface.

    Call detect(person_crop_bgr) per detection.  Returns a FaceResult.

    Thread-safe: OpenCV DNN net.forward() is NOT thread-safe — instantiate
    one FaceDetector per CameraWorker thread (already the case since each
    worker owns its own feature pipeline).
    """

    def __init__(self):
        if not Config.FACE_DETECTION:
            self._det     = None
            self._embedder = None
            log.info("Face detection disabled (FACE_DETECTION=False)")
            return

        # Try DNN first, fall back to Haar
        dnn = _DNNFaceDetector()
        if dnn.available:
            self._det = dnn
        else:
            haar = _HaarFaceDetector()
            self._det = haar if haar.available else None

        if self._det is None:
            log.warning("No face detector backend available — face step skipped")

        # Embedder is optional
        self._embedder = _FaceEmbedder() if Config.FACE_EMBEDDING else None

        status = "DNN" if isinstance(self._det, _DNNFaceDetector) else \
                 "Haar" if isinstance(self._det, _HaarFaceDetector) else "none"
        embed_status = "yes" if (self._embedder and self._embedder.available) else "no"
        log.info(f"FaceDetector ready — backend={status}  embedding={embed_status}")

    @property
    def enabled(self) -> bool:
        return Config.FACE_DETECTION and self._det is not None

    def detect(self, person_crop: np.ndarray) -> FaceResult:
        """
        Run face detection on a person body crop.

        Args:
            person_crop: BGR numpy array (the body bounding-box crop from YOLO)

        Returns:
            FaceResult with found/bbox/conf/quality/embedding populated.
            Returns NO_FACE sentinel if disabled or no face found.
        """
        if not self.enabled or person_crop is None or person_crop.size == 0:
            return NO_FACE

        # Only look in the upper ~50% of the body crop — faces aren't in legs.
        h, w = person_crop.shape[:2]
        upper_h = max(1, int(h * Config.FACE_SEARCH_FRACTION))
        search_region = person_crop[:upper_h, :]

        candidates = self._det.detect(search_region)
        if not candidates:
            return FaceResult(found=False)

        # Take the most-confident detection.
        conf, (x1, y1, x2, y2) = candidates[0]

        face_crop = search_region[y1:y2, x1:x2]
        quality   = _estimate_quality(face_crop)

        embedding = None
        if (self._embedder and self._embedder.available
                and quality >= Config.FACE_MIN_QUALITY
                and conf    >= Config.FACE_MIN_CONF):
            embedding = self._embedder.embed(face_crop)

        return FaceResult(
            found    = True,
            bbox     = (x1, y1, x2, y2),   # in search_region coords
            conf     = float(conf),
            quality  = quality,
            embedding= embedding,
        )

    def batch_detect(self, crops: list) -> list[FaceResult]:
        """
        Detect faces in a batch of person crops.
        Returns a list of FaceResult aligned with the input list.
        """
        return [self.detect(c) for c in crops]


# ------------------------------------------------------------------ #
# Distance helper (used in reid.py)                                   #
# ------------------------------------------------------------------ #

def face_distance(result_a: FaceResult | None,
                  result_b: FaceResult | None) -> float:
    """
    Cosine distance between two face embeddings.

    Returns 2.0 (maximum distance) when either result is unusable,
    so callers can safely treat it as a fallback / no-signal value.
    """
    if result_a is None or result_b is None:
        return 2.0
    if not result_a.usable or not result_b.usable:
        return 2.0
    a = result_a.embedding
    b = result_b.embedding
    if a is None or b is None:
        return 2.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 2.0
    return float(1.0 - np.clip(np.dot(a / na, b / nb), -1, 1))


def face_gate(result: FaceResult | None,
              candidate_face: FaceResult | None,
              veto_threshold: float = None) -> bool:
    """
    Returns False (VETO) when two face embeddings are clearly different,
    True (PASS) otherwise — including when faces are not usable.

    Used in _resolve() to prevent a body-embedding match from linking
    two detections whose faces are clearly different people.

    veto_threshold defaults to Config.FACE_VETO_DIST.
    """
    if veto_threshold is None:
        veto_threshold = Config.FACE_VETO_DIST
    d = face_distance(result, candidate_face)
    if d >= 2.0:
        return True   # no face info → don't veto
    return d <= veto_threshold

