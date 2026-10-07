"""
FoundYou — Benchmark Runner
=============================
End-to-end benchmark script. Run this to get a full metrics report
against a ground-truth annotation file, or against synthetic data
if no annotation file is supplied.

Usage
-----
  # Against a real annotation CSV:
  python -m run_benchmark --gt path/to/gt.csv --pred path/to/pred.csv

  # Against synthetic data (for CI / smoke test):
  python -m run_benchmark --synthetic

  # Ablation across threshold variants using synthetic data:
  python -m run_benchmark --synthetic --ablation

  # Save results:
  python -m run_benchmark --synthetic --ablation --out results.csv
"""

import argparse
import sys

from data import (
    load_csv, generate_synthetic_sequence,
)
from metrics import (
    mota, motp, idf1, fragmentation
)
from ablation import run_ablation, print_table, save_csv


# ================================================================
# Helpers
# ================================================================

def _fmt(label: str, value, width: int = 28) -> str:
    return f"  {label:<{width}}: {value}"


def _print_section(title: str):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print('─' * 50)


# ================================================================
# Single-run report
# ================================================================

def run_single(gt_path: str, pred_path: str, iou_thr: float = 0.5):
    print(f"\nLoading ground truth : {gt_path}")
    gt   = load_csv(gt_path)
    print(f"Loading predictions  : {pred_path}")
    pred = load_csv(pred_path)

    print(f"  GT detections  : {len(gt)}")
    print(f"  Pred detections: {len(pred)}")

    _print_section("Multi-Object Tracking")
    mot = mota(gt, pred, iou_thr)
    for k, v in mot.items():
        print(_fmt(k, v))

    mpt = motp(gt, pred, iou_thr)
    print(_fmt('motp', mpt['motp']))
    print(_fmt('matched_count', mpt['matched_count']))

    _print_section("Identity F1 (IDF1)")
    idf = idf1(gt, pred, iou_thr)
    for k, v in idf.items():
        print(_fmt(k, v))

    _print_section("Fragmentation")
    frag = fragmentation(gt, pred, iou_thr)
    print(_fmt('fragmentations', frag))
    print(_fmt('id_switches', mot['id_sw']))


# ================================================================
# Synthetic single run
# ================================================================

def run_synthetic(n_people=4, n_cameras=2, n_frames=200,
                  n_id_switches=3, n_fragmentations=4):
    print("\nGenerating synthetic sequence...")
    print(f"  people={n_people}  cameras={n_cameras}  frames={n_frames}")
    print(f"  injected id_switches={n_id_switches}  fragmentations={n_fragmentations}")

    gt, pred = generate_synthetic_sequence(
        n_people        = n_people,
        n_cameras       = n_cameras,
        n_frames        = n_frames,
        n_id_switches   = n_id_switches,
        n_fragmentations= n_fragmentations,
    )

    _print_section("Multi-Object Tracking")
    mot = mota(gt, pred)
    for k, v in mot.items():
        print(_fmt(k, v))

    mpt = motp(gt, pred)
    print(_fmt('motp', mpt['motp']))

    _print_section("Identity F1 (IDF1)")
    idf = idf1(gt, pred)
    for k, v in idf.items():
        print(_fmt(k, v))

    _print_section("Fragmentation")
    frag = fragmentation(gt, pred)
    print(_fmt('fragmentations (injected)', f"{frag}  (expected ~{n_fragmentations})"))
    print(_fmt('id_switches    (injected)', f"{mot['id_sw']}  (expected ~{n_id_switches})"))

    return gt, pred


# ================================================================
# Ablation run
# ================================================================

def run_ablation_demo(out_path: str | None = None):
    """
    Demonstrate the ablation runner with four synthetic variants.

    Each variant simulates a different Config setting by injecting a
    different number of errors into the predicted sequence.
    """
    print("\nRunning ablation demo (synthetic data)...")

    gt_base, _ = generate_synthetic_sequence(
        n_people=4, n_cameras=2, n_frames=300, seed=0)

    # Simulate four system variants by varying injected error counts
    variant_params = [
        dict(name='baseline',          n_id_switches=2, n_fragmentations=3),
        dict(name='tight_cross_cam',   n_id_switches=1, n_fragmentations=3),
        dict(name='no_merge_pass',     n_id_switches=2, n_fragmentations=8),
        dict(name='min_probes_1',      n_id_switches=5, n_fragmentations=3),
    ]

    sequences = []
    for vp in variant_params:
        _, pr = generate_synthetic_sequence(
            n_people=4, n_cameras=2, n_frames=300,
            n_id_switches   = vp['n_id_switches'],
            n_fragmentations= vp['n_fragmentations'],
            seed=vp['n_id_switches'] * 7 + vp['n_fragmentations'],
        )
        sequences.append(pr)

    config_variants = [
        {'name': 'baseline',         'CROSS_CAM_DIST': 0.40, 'MERGE_DIST': 0.35, 'MIN_PROBES': 3},
        {'name': 'tight_cross_cam',  'CROSS_CAM_DIST': 0.30, 'MERGE_DIST': 0.35, 'MIN_PROBES': 3},
        {'name': 'no_merge_pass',    'CROSS_CAM_DIST': 0.40, 'MERGE_DIST': 0.00, 'MIN_PROBES': 3},
        {'name': 'min_probes_1',     'CROSS_CAM_DIST': 0.40, 'MERGE_DIST': 0.35, 'MIN_PROBES': 1},
    ]

    results = run_ablation(
        gt_detections       = gt_base,
        detection_sequences = sequences,
        variants            = config_variants,
    )

    _print_section("Ablation Results")
    print_table(results)

    if out_path:
        save_csv(results, out_path)
        print(f"\n  Saved to: {out_path}")

    return results


# ================================================================
# CLI
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="FoundYou benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--gt',        type=str,   help='Path to GT CSV annotation file')
    p.add_argument('--pred',      type=str,   help='Path to predictions CSV file')
    p.add_argument('--iou-thr',   type=float, default=0.5,
                   help='IoU threshold for spatial matching (default: 0.5)')
    p.add_argument('--synthetic', action='store_true',
                   help='Run on synthetic data instead of real annotations')
    p.add_argument('--ablation',  action='store_true',
                   help='Run ablation comparison across Config variants')
    p.add_argument('--out',       type=str,   default=None,
                   help='Output CSV path for ablation results')
    p.add_argument('--people',    type=int,   default=4)
    p.add_argument('--cameras',   type=int,   default=2)
    p.add_argument('--frames',    type=int,   default=200)
    return p.parse_args()


def main():
    args = parse_args()

    if args.ablation:
        run_ablation_demo(out_path=args.out)
        return

    if args.synthetic:
        run_synthetic(
            n_people  = args.people,
            n_cameras = args.cameras,
            n_frames  = args.frames,
        )
        return

    if args.gt and args.pred:
        run_single(args.gt, args.pred, iou_thr=args.iou_thr)
        return

    print("Specify --synthetic, --ablation, or both --gt and --pred.")
    print("Run with --help for usage.")
    sys.exit(1)


if __name__ == '__main__':
    main()
