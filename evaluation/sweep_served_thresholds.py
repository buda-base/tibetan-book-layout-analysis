#!/usr/bin/env python3
"""Task A — per-class confidence sweep for the SERVED model (RT-DETR-l tam2col
seed 0) on the leak-free v4 833-page test, to re-establish the served threshold
recipe (h/f ~0.60, text-area ~0.55, footnote ~0.25) leak-free.

Canonical space (unified scorer): per class {header-footer, text-area, footnote}
report P/R/F1 across a confidence grid, the best-F1 conf per class, and the
0.25-vs-served operating points. For text-area we sweep BOTH the merged
canonical envelope AND the native (pre-merge) per-box scheme, and label which is
which (the paper's "0.955 -> 0.98 at conf 0.55" was the NATIVE sweep). We also
report the single global best-mean-F1 conf and the mean-F1 lost by using one
global conf vs per-class tuning.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from literature_metrics import (  # noqa: E402
    _pr_class, apply_schema, best_f1_sweep, iter_pages, load_sizes, to_px,
)

CANON = ["header-footer", "text-area", "footnote"]
COARSE = [round(0.05 * i, 2) for i in range(1, 19)]   # 0.05..0.90
FINE = [round(0.01 * i, 2) for i in range(1, 96)]     # 0.01..0.95
SERVED = {"header-footer": 0.60, "text-area": 0.55, "footnote": 0.25}


def canonical_pr(pages, conf):
    """Per-class P/R/F1 in the canonical space at operating `conf` (predictions
    gated + text-area re-enveloped from surviving boxes)."""
    gt = {n: {} for n in CANON}
    pred = {n: {} for n in CANON}
    for stem, W, H, gb, pb in pages:
        g = apply_schema(gb, "canonical", W, H)
        p = apply_schema(pb, "canonical", W, H, conf_floor=conf)
        for n in CANON:
            if g[n]:
                gt[n][stem] = [xy for xy, _ in g[n]]
            if p[n]:
                pred[n][stem] = p[n]
    return {n: _pr_class(gt[n], pred[n], conf=0.0) for n in CANON}


def native_textarea_pr(pages, conf):
    """P/R/F1 for text-area scored NATIVELY (per predicted box, no envelope
    merge) against the native text-area GT (class 1)."""
    gt, pred = {}, {}
    for stem, W, H, gb, pb in pages:
        gts = [to_px(b, W, H) for b in gb if b["cls"] == 1]
        prs = [(to_px(b, W, H), b["conf"]) for b in pb
               if b["cls"] == 1 and b["conf"] >= conf]
        if gts:
            gt[stem] = gts
        if prs:
            pred[stem] = prs
    return _pr_class(gt, pred, conf=0.0)


def curve(pages, grid):
    rows = {n: [] for n in CANON + ["text-area-native"]}
    for c in grid:
        per = canonical_pr(pages, c)
        for n in CANON:
            r = per[n]
            rows[n].append({"conf": c, "P": r["P"], "R": r["R"], "F1": r["F1"],
                            "tp": r["tp"], "fp": r["fp"], "fn": r["fn"]})
        nt = native_textarea_pr(pages, c)
        rows["text-area-native"].append(
            {"conf": c, "P": nt["P"], "R": nt["R"], "F1": nt["F1"],
             "tp": nt["tp"], "fp": nt["fp"], "fn": nt["fn"]})
    return rows


def best_of(curve_rows):
    b = max(curve_rows, key=lambda r: r["F1"])
    return b


def point_at(curve_rows, conf):
    return min(curve_rows, key=lambda r: abs(r["conf"] - conf))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path,
                    default=Path("/home/eroux/tmp/dataset_tdlav4_tam2col/labels/test"))
    ap.add_argument("--img-dir", type=Path,
                    default=Path("/home/eroux/tmp/dataset_tdlav4_tam2col/images/test"))
    ap.add_argument("--pred-dir", type=Path,
                    default=Path("/home/eroux/tmp/seed_variance/seed0/labels"))
    ap.add_argument("--out", type=Path,
                    default=Path("/home/eroux/BUDA/softs/tibetan-book-layout-analysis/"
                                 "evaluation/eval_results/tdla-v4/served_thresholds.json"))
    args = ap.parse_args()

    sizes = load_sizes(args.img_dir,
                       cache=args.out.parent / "literature" / "test_image_sizes.json")
    pages = list(iter_pages(args.gt_dir, args.pred_dir, sizes, conf_floor=0.0))
    print(f"{len(pages)} pages", flush=True)

    fine = curve(pages, FINE)
    coarse = curve(pages, COARSE)

    # per-class best-F1 (fine grid)
    best = {n: best_of(fine[n]) for n in CANON + ["text-area-native"]}
    # global best-mean-F1 conf (canonical), + mean-F1 loss vs per-class tuning
    sweep = best_f1_sweep(pages, "canonical")
    gconf = sweep["best_mean_conf"]
    at_g = canonical_pr(pages, gconf)
    mean_global = float(np.mean([at_g[n]["F1"] for n in CANON]))
    mean_perclass = float(np.mean([best[n]["F1"] for n in CANON]))

    # 0.25 (default) vs served operating points
    served_pts = {}
    for n in CANON:
        src = fine["text-area-native"] if n == "text-area" else fine[n]
        served_pts[n] = {
            "space": "native" if n == "text-area" else "canonical",
            "served_conf": SERVED[n],
            "at_0.25": point_at(src, 0.25),
            "at_served": point_at(src, SERVED[n]),
        }

    out = {
        "model": "RT-DETR-l tam2col seed0 (served)", "n_pages": len(pages),
        "grid": "fine 0.01..0.95 (curves), coarse 0.05..0.90 (readable)",
        "global_best_mean_F1_conf": gconf,
        "mean_F1_at_global_conf": mean_global,
        "mean_F1_perclass_tuned": mean_perclass,
        "mean_F1_loss_global_vs_perclass": mean_perclass - mean_global,
        "best_F1_per_class": {n: best[n] for n in best},
        "served_vs_default": served_pts,
        "curve_coarse": coarse,
        "curve_fine": fine,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}\n")

    print(f"global best-mean-F1 conf = {gconf:.2f}  mean-F1 there = {mean_global:.4f}")
    print(f"per-class-tuned mean-F1  = {mean_perclass:.4f}  "
          f"(loss from one global conf = {mean_perclass - mean_global:.4f})\n")
    print("best-F1 per class (fine grid):")
    for n in CANON + ["text-area-native"]:
        b = best[n]
        print(f"  {n:18s} F1={b['F1']:.4f} @conf {b['conf']:.2f}  "
              f"P={b['P']:.4f} R={b['R']:.4f}")
    print("\n0.25 (default) vs served:")
    for n in CANON:
        d, s = served_pts[n]["at_0.25"], served_pts[n]["at_served"]
        print(f"  {n:14s} [{served_pts[n]['space']:9s}] "
              f"0.25: P={d['P']:.4f} R={d['R']:.4f} F1={d['F1']:.4f}  ->  "
              f"{SERVED[n]:.2f}: P={s['P']:.4f} R={s['R']:.4f} F1={s['F1']:.4f}  "
              f"(dP={s['P']-d['P']:+.4f} dR={s['R']-d['R']:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
