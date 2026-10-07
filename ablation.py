"""
FoundYou — Ablation Runner
===========================
Run the same detection sequence through multiple Config variants and
compare metric outcomes side by side. This is the core tool for the
paper's ablation table.

Each variant overrides specific Config fields while keeping everything
else at the baseline. Results are collected into a comparison table
that can be printed, saved as CSV, or returned as a list of dicts.

Example
-------
    from ablation import run_ablation
    from data import load_csv, generate_synthetic_sequence

    gt, pr_baseline = generate_synthetic_sequence()

    table = run_ablation(
        gt_detections = gt,
        detection_sequences = [pr_baseline],   # one sequence per variant
        variants = [
            {'name': 'baseline',              'CROSS_CAM_DIST': 0.40, 'MERGE_DIST': 0.35},
            {'name': 'tight_cross_cam',       'CROSS_CAM_DIST': 0.30, 'MERGE_DIST': 0.35},
            {'name': 'no_merge',              'CROSS_CAM_DIST': 0.40, 'MERGE_DIST': 0.00},
            {'name': 'min_probes_1',          'CROSS_CAM_DIST': 0.40, 'MIN_PROBES_TO_MATCH': 1},
        ],
    )

    print_table(table)
    save_csv(table, 'ablation_results.csv')
"""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, List, Optional

from metrics import (
    Detection, TrackingResult,
    mota, motp, idf1, fragmentation,
    gallery_dist_stats, resolve_step_breakdown,
)


# ================================================================
# Core runner
# ================================================================

def run_ablation(
    gt_detections        : List[Detection],
    detection_sequences  : List[List[Detection]],
    variants             : List[Dict[str, Any]],
    tracking_results     : Optional[List[TrackingResult]] = None,
    tracks_list          : Optional[List[list]] = None,
    iou_thr              : float = 0.5,
) -> List[Dict]:
    """
    Evaluate each variant and return a list of result dicts.

    Parameters
    ----------
    gt_detections       : ground-truth Detection list (shared across variants)
    detection_sequences : one predicted Detection list per variant
    variants            : list of dicts, each with a 'name' key plus any
                          Config field overrides (used for documentation only
                          — the caller is responsible for running FoundYou
                          with those settings and passing the correct sequence)
    tracking_results    : optional list of TrackingResult per variant
                          (for step breakdown and merge metrics)
    tracks_list         : optional list of GlobalTrack lists per variant
                          (for gallery quality metrics)
    iou_thr             : IoU threshold for spatial matching

    Returns
    -------
    List of dicts, one per variant, each containing all metric values
    plus the variant's config overrides.
    """
    if len(detection_sequences) != len(variants):
        raise ValueError(
            f"Got {len(detection_sequences)} sequences but {len(variants)} variants")

    results = []

    for i, (variant, pr_dets) in enumerate(zip(variants, detection_sequences)):
        name   = variant.get('name', f'variant_{i}')
        result = (tracking_results[i]
                  if tracking_results and i < len(tracking_results)
                  else TrackingResult())
        tracks = (tracks_list[i]
                  if tracks_list and i < len(tracks_list)
                  else [])

        mot   = mota(gt_detections, pr_dets, iou_thr)
        motp_ = motp(gt_detections, pr_dets, iou_thr)
        idf   = idf1(gt_detections, pr_dets, iou_thr)
        frag  = fragmentation(gt_detections, pr_dets, iou_thr)
        step  = resolve_step_breakdown(result)
        gal   = gallery_dist_stats(tracks) if tracks else {}

        row = {
            'name'            : name,
            # key config overrides (for the table header)
            **{k: v for k, v in variant.items() if k != 'name'},
            # tracking metrics
            'mota'            : mot['mota'],
            'motp'            : motp_['motp'],
            'idf1'            : idf['idf1'],
            'idp'             : idf['idp'],
            'idr'             : idf['idr'],
            'fp'              : mot['fp'],
            'fn'              : mot['fn'],
            'id_switches'     : mot['id_sw'],
            'fragmentations'  : frag,
            # pipeline
            'step_a_frac'     : step['step_a'],
            'step_b_frac'     : step['step_b'],
            'step_c_frac'     : step['step_c'],
        }
        if gal:
            row['separation_ratio'] = gal.get('separation_ratio', '')
            row['intra_dist_mean']  = gal.get('intra_mean', '')

        results.append(row)

    return results


# ================================================================
# Output helpers
# ================================================================

# Columns shown in the printed table (in order)
_TABLE_COLS = [
    ('name',          'Variant',      20),
    ('mota',          'MOTA',          8),
    ('motp',          'MOTP',          8),
    ('idf1',          'IDF1',          8),
    ('id_switches',   'IDS',           6),
    ('fragmentations','Frags',         6),
    ('fp',            'FP',            6),
    ('fn',            'FN',            6),
    ('step_a_frac',   'StepA',         7),
    ('step_b_frac',   'StepB',         7),
    ('step_c_frac',   'StepC',         7),
]


def print_table(results: List[Dict], file=None) -> None:
    """
    Print ablation results as a fixed-width ASCII table.

    Parameters
    ----------
    results : output of run_ablation()
    file    : file-like object (default: stdout)
    """
    import sys
    out = file or sys.stdout

    # Header
    header = '  '.join(f"{col:{w}}" for _, col, w in _TABLE_COLS)
    sep    = '-' * len(header)
    print(sep, file=out)
    print(header, file=out)
    print(sep, file=out)

    for row in results:
        line = '  '.join(
            f"{str(row.get(key, '')):{w}}"
            for key, _, w in _TABLE_COLS
        )
        print(line, file=out)

    print(sep, file=out)


def table_to_string(results: List[Dict]) -> str:
    """Return print_table output as a string."""
    buf = io.StringIO()
    print_table(results, file=buf)
    return buf.getvalue()


def save_csv(results: List[Dict], path: str) -> None:
    """
    Save ablation results to a CSV file.

    Parameters
    ----------
    results : output of run_ablation()
    path    : output file path
    """
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def delta_table(baseline : Dict,
                variants  : List[Dict]) -> List[Dict]:
    """
    Compute per-metric delta relative to a baseline row.

    Positive delta = improvement (higher MOTA/IDF1, lower IDS/Frags).

    Parameters
    ----------
    baseline : one row from run_ablation() results
    variants : remaining rows

    Returns
    -------
    List of dicts with same keys, values replaced by (absolute, delta) tuples
    formatted as strings: "0.812 (+0.034)"
    """
    # Higher = better for these
    # higher_better = {'mota', 'motp', 'idf1', 'idp', 'idr', 'separation_ratio'}
    # Lower = better for these
    lower_better  = {'id_switches', 'fragmentations', 'fp', 'fn', 'intra_dist_mean'}

    out = []
    for row in variants:
        delta_row = {'name': row['name']}
        for key, col, _ in _TABLE_COLS:
            if key == 'name':
                continue
            try:
                val  = float(row.get(key, 0))
                base = float(baseline.get(key, 0))
                diff = val - base
                if key in lower_better:
                    diff = -diff   # flip sign so positive always = better
                sign = '+' if diff >= 0 else '-'
                delta_row[key] = f"{val:.4f} ({sign}{abs(diff):.4f})"
            except (TypeError, ValueError):
                delta_row[key] = str(row.get(key, ''))
        out.append(delta_row)
    return out
