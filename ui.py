"""
FoundYou — UI / Rendering
==========================
Pure OpenCV/numpy. No camera or ReID logic here.
Draws bounding boxes, assembles camera tiles, renders the sidebar.
"""

import cv2
import numpy as np

from config import Config
from tracks import GlobalTrack
from id_validator import IssueKind

CAM_PAD = 10  # pixels of dark border around each camera frame
ID_STRIP_H = 54  # height of the ID strip below each camera

_TITLE_CACHE: np.ndarray | None = None
_FOOTER_CACHE: np.ndarray | None = None


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
    _FOOTER_CACHE = None  # regenerate if layout changed
    if _FOOTER_CACHE is not None and _FOOTER_CACHE.shape[1] == width:
        return _FOOTER_CACHE
    h = 55
    f = np.zeros((h, width, 3), np.uint8)
    f[:] = (18,18,18)
    cv2.line(f,(0,0),(width,0),(255,120,60),2)
    x = 20
    for ctrl in ["[Q] Quit", "[D] Debug", "[S] Stats", "[V] Validate", "[X] Screenshot"]:
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
        lbl   = f"ID {gid}"
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


def assemble_video(frames_ready: dict, sorted_cids: list,
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


def draw_sidebar(sb: np.ndarray, mgr, ts: float, fps: float,
                 val_issues: list = None) -> np.ndarray:
    sb[:] = (18, 18, 18)
    y      = 20
    lh     = 28
    indent = 16

    # ---- auto-compute value column x so it clears the longest label ----
    VAL_X = indent + max(
        cv2.getTextSize("Same-cam reentry", cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0][0],
        cv2.getTextSize("DB total ", cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)[0][0],
    ) + 14

    def row(s, col=(210, 210, 225), sc=0.58):
        nonlocal y
        cv2.putText(sb, s, (indent, y),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)
        y += lh

    def row2(label, value, col=(100, 245, 130), sc=0.50):
        """Two-column row with label left-aligned and value at fixed VAL_X."""
        nonlocal y
        cv2.putText(sb, label, (indent, y),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)
        cv2.putText(sb, value, (VAL_X, y),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)
        y += lh

    def hdr(s, col=(255, 120, 60)):
        nonlocal y
        y += 2                                          # breathing room above separator
        cv2.line(sb, (0, y), (sb.shape[1], y),
                 (55, 55, 55), 1)                        # separator line
        y += 30                                          # gap between line and heading text
        cv2.putText(sb, s, (indent - 4, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                    col, 2, cv2.LINE_AA)
        y += 32                                          # gap below heading before first row

    # ---- SYSTEM ----
    hdr("SYSTEM")
    row2(f"Runtime", f": {int(ts)}s", col=(210, 210, 225), sc=0.58)
    row2(f"FPS", f": {fps:.1f}", col=(210, 210, 225), sc=0.58)
    row2(f"Active", f": {len(mgr.active)} IDs", col=(210, 210, 225), sc=0.58)
    row2(f"DB total", f": {len(mgr.db.tracks)}", col=(210, 210, 225), sc=0.58)
    row2(f"Max GID", f": {GlobalTrack._nxt - 1}", col=(210, 210, 225), sc=0.58)

    # ---- THRESHOLDS ----
    hdr("THRESHOLDS")
    row2("Cross-cam",        f"< {Config.CROSS_CAM_DIST:.2f}")
    row2("Same-cam reentry", f"< {Config.SAME_CAM_REENTRY_DIST:.2f}")
    row2("Reacquisition",    f"< {Config.REACQ_DIST:.2f}")
    row2("Active merge",     f"< {Config.MERGE_DIST:.2f}")
    row2("Min probes",       f"  {Config.MIN_PROBES_TO_MATCH}")

    # ---- STATS ----
    hdr("STATS")
    st = mgr.db.stats
    for k in ['created', 'links', 'reacquired', 'merged', 'archived', 'cleaned']:
        row2(k, str(st.get(k, 0)), col=(210, 210, 225))

    y += 4
    hdr("VALIDATION")
    if not val_issues:
        row("No issues detected", (100, 255, 120), 0.42)
    else:
        from id_validator import IssueKind
        swap_n = sum(1 for i in val_issues if i.kind == IssueKind.SWAP)
        dup_n = sum(1 for i in val_issues if i.kind == IssueKind.DUPLICATE)
        ust_n = sum(1 for i in val_issues if i.kind == IssueKind.UNSTABLE)
        if swap_n:
            row(f"SWAP     x{swap_n}  (press V)", (60, 60, 255), 0.44)
        if dup_n:
            row(f"DUPLICATE x{dup_n}", (60, 180, 255), 0.44)
        if ust_n:
            row(f"UNSTABLE  x{ust_n}", (60, 200, 255), 0.44)
        for issue in val_issues[:3]:  # show up to 3 inline
            short = issue.summary()[:38]
            sev_color = (60, 60, 255) if issue.severity == "HIGH" else (60, 160, 255)
            row(short, sev_color, 0.38)

    y += 4

    # ---- ACTIVE IDs ----
    hdr("ACTIVE IDs")
    gts   = sorted(mgr.active.values(), key=lambda g: g.gid)
    avail = max(1, (sb.shape[0] - y - 16) // lh)

    for gt in gts[:avail]:
        if y + lh > sb.shape[0] - 8:
            break
        swatch_y1 = y - 16
        swatch_y2 = y - 3
        cv2.rectangle(sb, (indent, swatch_y1), (indent + 14, swatch_y2), gt.color, -1)
        cv2.rectangle(sb, (indent, swatch_y1), (indent + 14, swatch_y2), (200, 120, 60), 1)

        cur_cams = mgr.current_cams(gt, ts)
        cams     = ','.join(str(c) for c in sorted(cur_cams)) if cur_cams else '-'
        age      = int(ts - gt.first_seen)
        flag     = '' if gt.is_matchable() else ' [W]'
        cv2.putText(sb,
            f"ID {gt.gid}  C[{cams}]  {age}s  p={gt.total_probes()}{flag}",
            (indent + 20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.47, (215, 215, 230), 1, cv2.LINE_AA)
        y += lh

    if len(gts) > avail:
        row(f"  ... +{len(gts) - avail} more", col=(110, 110, 125), sc=0.48)

    return sb


# ======================== Per-frame detection overlay =========================

def draw_detections(frame: np.ndarray, dets: list, mgr, ts: float,
                    val_issues: list = None) -> np.ndarray:
    """Draw bounding boxes and global-ID labels onto a copy of frame."""
    # Build set of GIDs flagged as swapped for fast lookup
    swapped_gids = set()
    if val_issues:
        for issue in val_issues:
            if issue.kind == IssueKind.SWAP and issue.is_persistent:
                swapped_gids.add(issue.gid_a)
                if issue.gid_b:
                    swapped_gids.add(issue.gid_b)

    disp = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = d['bbox']
        gid = d.get('gid')
        if gid is None:
            cv2.rectangle(disp, (x1, y1), (x2, y2), (80, 80, 80), 2)
            continue
        gt = mgr.active.get(gid) or mgr.db.get(gid)
        if not gt:
            continue

        box_color = gt.color
        if gid in swapped_gids:
            # Pulsing red border overlay to flag the swap
            cv2.rectangle(disp, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                          (0, 0, 255), 4)
        cv2.rectangle(disp, (x1, y1), (x2, y2), box_color, 3)
        cur_cams = mgr.current_cams(gt, ts)
        lbl = f"ID {gid}"
        if gid in swapped_gids:
            lbl += " !"  # visual warning in the label

        if len(cur_cams) > 1:
            lbl += f" [C: {','.join(str(c) for c in sorted(cur_cams))}]"
        if not gt.is_matchable():
            lbl += f" [{gt.total_probes()}/{Config.MIN_PROBES_TO_MATCH}]"

        (lw, lh_), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(disp, (x1, y1 - lh_ - 10), (x1 + lw + 10, y1), gt.color, -1)
        cv2.putText(disp, lbl, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return disp
