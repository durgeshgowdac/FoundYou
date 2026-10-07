"""
FoundYou — Camera Worker
=========================
One thread per camera. Runs YOLO+ByteTrack, crops detections,
extracts features, and pushes results to a shared output queue.
Completely decoupled from ReID logic.
"""

import time
from threading import Thread
import cv2
from config import Config
from logger import log
from face import FaceDetector, NO_FACE

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

        # One FaceDetector per worker thread (cv2 DNN is not thread-safe).
        face_det = FaceDetector()

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

                # Correct a physically-mirrored camera before any processing.
                # Flipping here means YOLO, ByteTrack, AND the ReID extractor
                # all see the corrected image — embeddings become comparable
                # with other cameras and ID switches are eliminated.
                if self.cam_id in Config.FLIP_CAMERAS:
                    frame = cv2.flip(frame, 1)  # 1 = horizontal flip

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
                    for m, f, crop in zip(metas, feats, crops):
                        m['feat'] = f
                        # --- Face detection step ---
                        # Runs per-crop on the upper body fraction.
                        # FaceResult is stored in meta; reid.py uses it as
                        # a veto / tie-breaker signal, never as the primary match.
                        if face_det.enabled:
                            try:
                                m['face'] = face_det.detect(crop)
                            except Exception:
                                m['face'] = NO_FACE
                        else:
                            m['face'] = NO_FACE

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
