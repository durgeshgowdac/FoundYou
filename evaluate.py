"""
FoundYou — Session Evaluator
==============================
Runs all benchmarks against a recorded and labeled session directory.
This is the single command you run after recording a live session and
labeling the ground truth.

Usage
-----
  python -m evaluate --session recordings/session_20240101_120000

  # With ablation across saved sessions from different Config runs:
  python -m evaluate \
      --sessions recordings/session_baseline recordings/session_tight_cam \
      --names    baseline tight_cross_cam \
      --ablation

  # Save report:
  python -m evaluate --session recordings/session_X --out report.csv
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

from labeler import load_labeled_session
from metrics import (
    mota, motp, idf1, fragmentation,
    gallery_dist_stats, resolve_step_breakdown, merge_effectiveness,
    throughput
)
from ablation import print_table, save_csv


# ================================================================
# Single session report
# ================================================================

def evaluate_session(session_dir: str,
                     iou_thr    : float = 0.5,
                     verbose    : bool  = True) -> dict:
    """
    Load a labeled session and compute the full metric suite.

    Returns
    -------
    dict of all metric values (same format as full_report())
    """
    if verbose:
        print(f"\nEvaluating: {session_dir}")

    gt_dets, pr_dets, result = load_labeled_session(session_dir)

    if not gt_dets:
        print("  WARNING: No labeled ground truth found.")
        print("  Run:  python -m labeler --session", session_dir)
        return {}

    if verbose:
        print(f"  GT detections  : {len(gt_dets)}")
        print(f"  PR detections  : {len(pr_dets)}")

    # Load config snapshot if available
    cfg_path = os.path.join(session_dir, 'config_snapshot.json')
    if verbose and os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        print(f"\n  Config (recorded):")
        for k in ['CROSS_CAM_DIST', 'SAME_CAM_REENTRY_DIST', 'REACQ_DIST',
                  'MERGE_DIST', 'MIN_PROBES_TO_MATCH', 'INACTIVE_TTL']:
            if k in cfg:
                print(f"    {k:<28}: {cfg[k]}")

    # Load track objects for gallery stats (best-effort)
    tracks = []
    try:
        import pickle
        db_path = os.path.join(session_dir, '..', '..', 'foundyou.pkl')
        if os.path.exists(db_path):
            with open(db_path, 'rb') as f:
                d = pickle.load(f)
            tracks = d.get('tracks', [])
    except Exception:
        pass

    # Meta for throughput
    meta_path = os.path.join(session_dir, 'metadata.json')
    total_dets = len(pr_dets)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        result.total_time = meta.get('duration_s', result.total_time)
        total_dets        = meta.get('total_dets',  total_dets)

    # Compute all metrics
    mot   = mota(gt_dets, pr_dets, iou_thr)
    motp_ = motp(gt_dets, pr_dets, iou_thr)
    idf   = idf1(gt_dets, pr_dets, iou_thr)
    frag  = fragmentation(gt_dets, pr_dets, iou_thr)
    step  = resolve_step_breakdown(result)
    gal   = gallery_dist_stats(tracks) if tracks else {}
    mrg   = merge_effectiveness(result, frag)
    tput  = throughput(result, total_dets)

    if verbose:
        _section("Tracking accuracy")
        _row("MOTA",         mot['mota'])
        _row("MOTP",         motp_['motp'])
        _row("IDF1",         idf['idf1'])
        _row("IDP",          idf['idp'])
        _row("IDR",          idf['idr'])

        _section("Error breakdown")
        _row("False positives",  mot['fp'])
        _row("False negatives",  mot['fn'])
        _row("ID switches",      mot['id_sw'])
        _row("Fragmentations",   frag)

        if gal:
            _section("Gallery quality")
            _row("Intra-gallery dist (mean)", gal['intra_mean'])
            _row("Inter-gallery dist (mean)", gal['inter_mean'])
            _row("Separation ratio",          gal['separation_ratio'])
            _row("Tracks measured",           gal['n_tracks'])

        _section("Pipeline breakdown")
        _row("Step A (active match)",    f"{step['step_a']:.1%}")
        _row("Step B (reacquisition)",   f"{step['step_b']:.1%}")
        _row("Step C (new track)",       f"{step['step_c']:.1%}")
        _row("Merge correction rate",    f"{mrg['correction_rate']:.1%}")
        _row("Over-merge rate",          f"{mrg['over_merge_rate']:.1%}")

        _section("Throughput")
        _row("Detections / sec",   tput['dets_per_sec'])
        _row("Resolves / sec",     tput['resolves_per_sec'])
        _row("Wall time (s)",      tput['wall_seconds'])

    report = {
        'session'         : os.path.basename(session_dir),
        'mota'            : mot['mota'],
        'motp'            : motp_['motp'],
        'idf1'            : idf['idf1'],
        'idp'             : idf['idp'],
        'idr'             : idf['idr'],
        'fp'              : mot['fp'],
        'fn'              : mot['fn'],
        'id_switches'     : mot['id_sw'],
        'fragmentations'  : frag,
        'step_a_frac'     : step['step_a'],
        'step_b_frac'     : step['step_b'],
        'step_c_frac'     : step['step_c'],
        'correction_rate' : mrg['correction_rate'],
        'over_merge_rate' : mrg['over_merge_rate'],
        'dets_per_sec'    : tput['dets_per_sec'],
        'wall_seconds'    : tput['wall_seconds'],
    }
    if gal:
        report['intra_dist_mean']  = gal['intra_mean']
        report['inter_dist_mean']  = gal['inter_mean']
        report['separation_ratio'] = gal['separation_ratio']

    return report


# ================================================================
# Multi-session ablation
# ================================================================

def evaluate_ablation(session_dirs : List[str],
                      names        : List[str],
                      iou_thr      : float = 0.5,
                      out_path     : Optional[str] = None):
    """
    Evaluate multiple sessions and print a side-by-side ablation table.
    """
    results = []
    for sd, name in zip(session_dirs, names):
        r = evaluate_session(sd, iou_thr=iou_thr, verbose=False)
        if r:
            r['name'] = name
            results.append(r)

    if not results:
        print("No results to compare.")
        return

    _section("Ablation comparison")
    print_table(results)

    if out_path:
        save_csv(results, out_path)
        print(f"\nSaved: {out_path}")


# ================================================================
# Helpers
# ================================================================

def _section(title: str):
    print(f"\n  {'─'*42}")
    print(f"  {title}")
    print(f"  {'─'*42}")


def _row(label: str, value, width: int = 30):
    print(f"    {label:<{width}}: {value}")


# ================================================================
# CLI
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(description="FoundYou session evaluator")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--session',  type=str,
                   help='Single session directory to evaluate')
    g.add_argument('--sessions', type=str, nargs='+',
                   help='Multiple session directories for ablation comparison')
    p.add_argument('--names',    type=str, nargs='*', default=None,
                   help='Names for each session in ablation mode')
    p.add_argument('--iou-thr',  type=float, default=0.5)
    p.add_argument('--out',      type=str,   default=None,
                   help='Output CSV path for results')
    p.add_argument('--ablation', action='store_true',
                   help='Compare multiple sessions side by side')
    return p.parse_args()


def main():
    args = parse_args()

    if args.session:
        report = evaluate_session(args.session, iou_thr=args.iou_thr)
        if report and args.out:
            save_csv([report], args.out)
            print(f"\nSaved: {args.out}")
        return

    if args.sessions:
        names = args.names or [os.path.basename(s) for s in args.sessions]
        evaluate_ablation(args.sessions, names,
                          iou_thr=args.iou_thr, out_path=args.out)


if __name__ == '__main__':
    main()
