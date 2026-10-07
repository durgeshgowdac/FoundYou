# FoundYou — Multi-Camera Person Re-Identification

Real-time person re-identification across multiple cameras. Each person receives a persistent global ID that follows them as they move between cameras, leave the frame, and return — surviving camera switches, temporary occlusions, and re-entries.

---

## Table of Contents

- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Benchmarking & Evaluation](#benchmarking--evaluation)
- [Project Structure](#project-structure)
- [Architecture Reference](#architecture-reference)
- [Metrics Reference](#metrics-reference)

---

## Quick Start

### Requirements

- Python 3.10+
- PyTorch (CPU / CUDA / MPS)
- OpenCV 4.x
- Ultralytics YOLO
- FAISS

### Install

```bash
git clone https://github.com/durgeshgowdac/foundyou.git
cd foundyou
pip install -r requirements.txt
```

For face embeddings (optional but recommended), install one of:

```bash
pip install insightface onnxruntime   # recommended
# or
pip install dlib face_recognition
```

> YOLO weights and the face DNN model (`res10_300x300_ssd`) are downloaded automatically on first run.

### Run

```bash
python main.py
```

### Device Support

Auto-detection order: **MPS → CUDA → CPU**

| Device | Notes |
|--------|-------|
| `cuda` | Recommended for multi-camera setups |
| `mps`  | Apple Silicon — auto-selected on macOS |
| `cpu`  | Works, slower on large batches |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Q` | Quit and save session |
| `D` | Toggle debug logging |
| `S` | Print stats to console |
| `V` | Print IDValidator report |
| `X` | Save screenshot to `Screenshots/` |

---

## How It Works

Each camera runs YOLO + ByteTrack in its own thread, producing per-camera local track IDs. A central `GlobalReIDManager` resolves those local IDs into persistent global ones using OSNet-512 appearance features and a three-step matching pipeline:

### Identity resolution (`reid.py`)

Every new `(cam, local_id)` tracklet is resolved through `_resolve_with_face()`, which runs three steps in order:

**Step A — match active track** - The incoming body embedding is compared against all currently active global tracks using robust gallery distance. A face veto blocks a match when both sides carry usable face embeddings and their face distance exceeds `FACE_VETO_DIST`. When faces are available on both sides, ranking is sharpened with a blended score: `(1 − FACE_WEIGHT) × body_dist + FACE_WEIGHT × face_dist`.

**Step B — reacquire archived track** - If `Step A` yields no match, FAISS runs a coarse inner-product search over archived tracks, pre-filtered by `REACQ_MIN_SIM`, followed by robust distance verification against `REACQ_DIST`. This is how the system recognises a person who left the scene and returned.

**Step C — create new track** - If both steps fail, a new `GlobalTrack` is created and assigned a fresh global ID.

### Robust gallery distance (`features.py`)

Instead of comparing a query against the nearest single stored frame (vulnerable to one blurry frame), the distance is:

```
robust_dist(feat, track) =
    (mean over cameras of median cosine distance to that camera's buffer
     + cosine distance to merged gallery mean) / 2
```

This makes thresholds meaningful — a single noisy frame cannot pull the distance below threshold.

### Post-hoc merge pass (`reid.py`)

When two people first appear, their galleries have fewer frames than `MIN_PROBES_TO_MATCH`, so `_resolve()` cannot link them. If ByteTrack drops and reassigns a local ID in that window, two global tracks are created for the same person. The merge pass runs every `MERGE_INTERVAL` seconds and collapses any two active tracks whose robust distance falls below `MERGE_DIST` and who are not simultaneously live on the same camera. The loser's feature buffer is folded into the winner's.

### ID validation (`id_validator.py`)

`IDValidator` runs periodically (read-only, never mutates state) and surfaces three issue types:

| Issue | Trigger |
|---|---|
| `SWAP` | Two active tracks on different cameras have gallery distance < `ID_SWAP_DIST` — likely the same person with two IDs |
| `DUPLICATE` | Two active tracks are both live on the same camera with gallery distance < `MERGE_DIST` — ByteTrack fragmentation the merge pass has not resolved yet |
| `UNSTABLE` | A track's intra-gallery cosine variance > 0.30 — embedding buffer contaminated by multiple people |

Issues are streaked across consecutive checks. Persistent issues (streak ≥ 3) are logged as warnings and highlighted on-screen with a red overlay.

### Face detection (`face.py`)

Each `CameraWorker` instantiates its own `FaceDetector` (one per thread — `cv2.dnn` is not thread-safe). Detection runs as a single batched `blobFromImages` → `net.forward()` call across all person crops in a frame, not a per-crop loop. The result is a `FaceResult` attached to each detection's metadata dict as `d['face']`.

When a face embedder (`insightface` or `dlib`) is available, the highest-quality face result for each track is stored as `track.best_face` and used in Step A matching.


### Session Persistence

On quit (`Q`), the full track database is saved to:

```
foundyou.pkl    — track objects (feature galleries, metadata)
foundyou.faiss  — FAISS index for fast archived-track search
```

These are loaded automatically on next launch. To start fresh:

```bash
rm foundyou.pkl foundyou.faiss
```

---

## Configuration

All thresholds live in `config.py` and can be overridden via CLI or programmatically.

### Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `CROSS_CAM_DIST` | `0.40` | Cosine distance threshold for cross-camera match |
| `SAME_CAM_REENTRY_DIST` | `0.30` | Tighter threshold for same-camera re-entry |
| `REACQ_DIST` | `0.40` | Archived track reacquisition threshold |
| `MERGE_DIST` | `0.35` | Post-hoc active track merge threshold |
| `MERGE_INTERVAL` | `2.0 s` | Seconds between merge passes |
| `MIN_PROBES_TO_MATCH` | `3` | Gallery frames required before a track can claim new tracklets |
| `INACTIVE_TTL` | `15.0 s` | Idle time before a track is archived |
| `CAM_FEAT_BUF` | `24` | Max frames stored per camera per track |
| `FACE_VETO_DIST` | `0.55` | Face cosine distance that blocks a body-embedding match |
| `FACE_WEIGHT` | `0.30` | Face contribution to blended matching score |

> All thresholds are **cosine distance** (0 = identical, 1 = orthogonal). Lower values are stricter.

#### Threshold Effects at a Glance

| Threshold | Tighter → | Looser → |
|-----------|-----------|----------|
| `CROSS_CAM_DIST` | More IDs, fewer false links | Fewer IDs, more false links |
| `SAME_CAM_REENTRY_DIST` | More IDs per camera | Fewer IDs, riskier merges |
| `REACQ_DIST` | Fewer reacquisitions | More reacquisitions, riskier |
| `MERGE_DIST` | Less merging | More aggressive merging |
| `MIN_PROBES_TO_MATCH` | Safer cold-start | More IDs during warmup |

### CLI

```bash
python main.py --cross-cam-dist 0.35 --device cuda --debug
```

| Flag | Default | Description |
|------|---------|-------------|
| `--yolo-model` | `yolo11s.pt` | YOLO weights file |
| `--device` | auto | `cpu`, `cuda`, or `mps` |
| `--max-cameras` | `3` | Camera indices to probe (0..N-1) |
| `--flip-cameras 1 2` | — | Horizontally flip specified cameras |
| `--cross-cam-dist` | `0.40` | Cross-camera cosine distance threshold |
| `--same-cam-reentry-dist` | `0.30` | Same-camera re-entry distance threshold |
| `--reacq-dist` | `0.40` | Archived track reacquisition threshold |
| `--min-probes-to-match` | `3` | Gallery frames required before matching |
| `--merge-dist` | `0.35` | Post-hoc active track merge threshold |
| `--merge-interval` | `2.0` | Seconds between merge pass runs |
| `--inactive-ttl` | `15.0` | Seconds before an unseen track is archived |
| `--record` | off | Record session to disk |
| `--record-dir` | `recordings/` | Output directory for recordings |
| `--debug` | off | Verbose debug logging |

### Programmatic

```python
from config import Config

Config.CROSS_CAM_DIST = 0.35
Config.DEVICE = 'cpu'
Config.DEBUG = True
```

---

## Benchmarking & Evaluation

### 1. Record a Session

```bash
python main.py --record --record-dir recordings/my_session
```

Each session saves:

```
recordings/session_<timestamp>/
  detections.csv        # frame, cam, local_id, global_id, x1, y1, x2, y2, ts
  events.csv            # ts, event_type, gid, cam, local_id, detail
  config_snapshot.json  # Config values at recording time
  metadata.json         # camera count, frame counts, duration, step histogram
```

### 2. Label Ground Truth

```bash
python labeler.py \
    --session recordings/session_<timestamp> \
    --cam 0 \
    --video path/to/video.mp4   # optional
```

Click a bounding box, type the real person ID, press Enter to confirm. Assignment propagates to all frames where that global ID appears. Press `S` to save `gt_cam0.csv`.

### 3. Evaluate a Session

```bash
python evaluate.py --session recordings/session_<timestamp>
```

Prints MOTA, MOTP, IDF1, FP/FN/IDS, gallery quality, pipeline step breakdown, merge effectiveness, and throughput.

```bash
# Save to CSV
python evaluate.py \
    --session recordings/session_<timestamp> \
    --out report.csv

# Compare multiple sessions (ablation)
python evaluate.py \
    --sessions recordings/session_baseline recordings/session_tight_cam \
    --names    baseline tight_cross_cam \
    --out      ablation.csv
```

### Synthetic Benchmark (no video or GPU required)

```bash
python run_benchmark.py --synthetic                   # single run
python run_benchmark.py --synthetic --ablation        # compare threshold variants
python run_benchmark.py --synthetic --ablation --out results.csv
```

### Unit Tests

```bash
python tests.py
# or
python -m pytest tests.py
```

All tests use synthetic data — no GPU, cameras, or annotation files required.

---

## Project Structure

```
foundyou/
├── main.py              # Entry point, camera discovery, main loop
├── config.py            # All thresholds and constants
├── camera.py            # CameraWorker thread — YOLO + ByteTrack + batched features
├── models.py            # OSNet backbone + FeatureExtractor
├── face.py              # FaceDetector (DNN/Haar) + optional ArcFace/dlib embedder
├── reid.py              # GlobalReIDManager — resolve, merge, archive
├── tracks.py            # GlobalTrack data model + DB (FAISS + pickle persistence)
├── features.py          # Distance helpers (_dist, _robust_dist_to_gallery)
├── id_validator.py      # Runtime SWAP / DUPLICATE / UNSTABLE detection
├── ui.py                # OpenCV rendering — boxes, sidebar, camera tiles
├── logger.py            # Shared logger instance
├── requirements.txt
├── metrics.py           # MOTA, MOTP, IDF1, CMC, mAP, ...
├── data.py              # GT loaders (CSV, MOT format) + synthetic data generator
├── recorder.py          # SessionRecorder — live CSV recording for offline eval
├── evaluate.py          # Session evaluator — full report from a labeled session
├── labeler.py           # OpenCV ground-truth labeling tool
├── ablation.py          # Ablation runner — compare Config variants side by side
├── run_benchmark.py     # CLI benchmark runner (real or synthetic data)
└── tests.py             # Unit tests for all metrics (no GPU required)
```

---

## Architecture Reference

### System Flowchart

```
┌──────────────────────────────────────────┐
│        Camera feed  (per thread)         │
│  cap.read() → resize → flip if needed    │
└───────────────────┬──────────────────────┘
                    │
┌───────────────────▼──────────────────────┐
│           YOLO + ByteTrack               │
│    Detect persons, assign local IDs      │
└───────────────────┬──────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼────────┐   ┌──────────▼───────────┐
│ OSNet batch    │   │ Face DNN batch detect │
│ 512d embeddings│   │ FaceResults, one fwd  │
│ (one fwd pass) │   │ pass per frame        │
└───────┬────────┘   └──────────┬────────────┘
        └───────────┬───────────┘
                    │
┌───────────────────▼──────────────────────┐
│      result_q  (cam_id, frame, metas)    │
└───────────────────┬──────────────────────┘
                    │
┌───────────────────▼──────────────────────┐
│    Main loop — drain queue, call update()│
│  Group by (cam, local_id), mean-pool     │
│  features, best face result kept         │
└───────────────────┬──────────────────────┘
                    │
┌───────────────────▼────────────────────────┐
│       GlobalReIDManager._resolve()         │
│  Robust gallery dist + face veto / blend   │
│                                            │
│  Step A        Step B          Step C      │
│ Match active   Reacquire       New         │
│  observe()     FAISS + verify  GlobalTrack │
└──────┬────────────┬───────────────┬────────┘
       └────────────┴───────────────┘
                    │
┌───────────────────▼──────────────────────┐
│       l2g[(cam, local_id)] = gid         │
└───────────────────┬──────────────────────┘
                    │
┌───────────────────▼──────────────────────┐
│           Periodic maintenance           │
│  Archive idle · Merge pass · FAISS       │
│  rebuild · IDValidator sweep             │
└───────────────────┬──────────────────────┘
                    │
┌───────────────────▼──────────────────────┐
│       UI render + cv2.imshow             │
│  Bounding boxes, sidebar, validation     │
│  overlay                                 │
└───────────────────┬──────────────────────┘
                    │
               Next frame  ↻
```


### Feature Highlights

- **Multi-camera ReID** — persistent global IDs across any number of cameras
- **ByteTrack integration** — robust local tracking within each camera
- **OSNet body embeddings** — 512-dimensional appearance features, flip-augmented at inference
- **Face detection pre-filter** — OpenCV DNN face detector gates and sharpens ReID matches; optional ArcFace / dlib embedding when available
- **Batched inference** — all crops processed in a single DNN forward pass per frame
- **Robust gallery matching** — per-camera median distance prevents single-frame false positives
- **Post-hoc merge pass** — collapses fragmented tracks once galleries are mature
- **ID validation** — runtime SWAP / DUPLICATE / UNSTABLE detection with streak tracking
- **FAISS archive search** — fast reacquisition of people who left and returned
- **Session persistence** — tracks saved to disk; IDs survive restarts
- **Session recorder** — optional CSV recording for offline benchmarking and ablation
- **Full benchmark suite** — MOTA, MOTP, IDF1, Rank-1/mAP, gallery quality, throughput


### Architecture notes

#### Why median distance instead of min?

`_robust_dist_to_gallery` computes the **median** cosine distance across all stored frames per camera, then averages the per-camera medians. The old `min()` approach was vulnerable to a single blurry or accidentally-similar frame pulling the distance down and causing false matches. Median is robust to those outliers.

#### Why a merge pass?

When a person first appears, their gallery has fewer than `MIN_PROBES_TO_MATCH` frames, so `_resolve()` skips matching entirely and creates a new track. If ByteTrack drops the ID in that window, a second track gets created for the same person. The merge pass runs every `MERGE_INTERVAL` seconds and collapses those duplicates once both galleries are large enough to compare reliably.

---

## Metrics Reference

| Metric | Function | Description |
|--------|----------|-------------|
| MOTA | `mota()` | Multi-Object Tracking Accuracy — accounts for FP, FN, ID switches |
| MOTP | `motp()` | Multi-Object Tracking Precision — mean IoU of matched pairs |
| IDF1 | `idf1()` | Identity F1 — precision/recall over identity-matched detections |
| Rank-k | `rank_k_accuracy()` | Fraction of queries with correct match in top-k |
| mAP | `mean_ap()` | Mean Average Precision over all queries |
| CMC | `compute_cmc()` | Full Cumulative Matching Characteristic curve |
| Fragmentation | `fragmentation()` | Track interruptions |
| Gallery quality | `gallery_dist_stats()` | Intra/inter-gallery distance + separation ratio |
| Probe convergence | `probe_convergence()` | How quickly gallery distance stabilises vs probe count |
| Step breakdown | `resolve_step_breakdown()` | Fraction of resolves via Step A / B / C |
| Merge effectiveness | `merge_effectiveness()` | Merges fired vs fragmentations corrected |
| Throughput | `throughput()` | Detections and resolves per second |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT