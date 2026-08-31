#!/usr/bin/env python3
"""Task B — re-score the 5 RT-DETR-l tam2col seed checkpoints with the UNIFIED
scorer (per-conf text-area envelope, 0.01 grid, best-mean-F1 operating point),
so the variance band is comparable to the headline 0.959 (unified) rather than
the old fixed-conf-0.50 band.

No retraining: re-scores the existing seed dumps
(s3://.../seed-variance-tdlav4/seed{0..4}/labels, pulled to --preds-root).

Per seed, on the v4 833-page test:
  * canonical mean F1 + per-class F1 (h/f, text-area, footnote) at that seed's
    own best-mean-F1 conf,
  * canonical mAP@[.5:.95] (conf-independent; matches the old json),
  * DocLayNet-aligned shared-class mAP@[.5:.95] (page-header/page-footer/
    footnote, text-area excluded) — same as run_v4_shared_map.py.
Then mean / sample-sd / min / max across the 5 seeds.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from literature_metrics import (  # noqa: E402
    SHARED, best_f1_sweep, coco_map, iter_pages, load_sizes, operating_points,
)

S3 = "s3://bec.bdrc.io/models/hff-detection/seed-variance-tdlav4"
SEEDS = ["seed0", "seed1", "seed2", "seed3", "seed4"]


def agg(values):
    a = np.array(values, dtype=float)
    return {
        "mean": float(a.mean()),
        "sample_std": float(a.std(ddof=1)),
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(a.size),
        "values": [float(v) for v in a],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path,
                    default=Path("/home/eroux/tmp/dataset_tdlav4_tam2col/labels/test"))
    ap.add_argument("--img-dir", type=Path,
                    default=Path("/home/eroux/tmp/dataset_tdlav4_tam2col/images/test"))
    ap.add_argument("--preds-root", type=Path,
                    default=Path("/home/eroux/tmp/seed_variance"))
    ap.add_argument("--out", type=Path,
                    default=Path("/home/eroux/BUDA/softs/tibetan-book-layout-analysis/"
                                 "evaluation/eval_results/tdla-v4/seed_variance_unified.json"))
    ap.add_argument("--pull", action="store_true", help="aws s3 cp the 5 dumps first")
    args = ap.parse_args()

    if args.pull:
        for s in SEEDS:
            dst = args.preds_root / s / "labels"
            dst.mkdir(parents=True, exist_ok=True)
            subprocess.run(["aws", "s3", "cp", f"{S3}/{s}/labels/", str(dst),
                            "--recursive", "--only-show-errors"], check=True)

    sizes = load_sizes(args.img_dir,
                       cache=args.out.parent / "literature" / "test_image_sizes.json")
    print(f"{len(sizes)} test image sizes", flush=True)

    per_seed = {}
    for s in SEEDS:
        pred = args.preds_root / s / "labels"
        pages = list(iter_pages(args.gt_dir, pred, sizes, conf_floor=0.0))
        sweep = best_f1_sweep(pages, "canonical")
        conf = sweep["best_mean_conf"]
        op = operating_points(pages, "canonical", conf)["per_class"]
        canon = coco_map(pages, "canonical")
        pages_floor = list(iter_pages(args.gt_dir, pred, sizes, conf_floor=0.05))
        shared = coco_map(pages_floor, "doclaynet", mean_over=SHARED)
        per_seed[s] = {
            "n_pages": len(pages),
            "operating_conf": conf,
            "mean_F1": sweep["best_mean_F1"],
            "F1": {k: op[k]["F1"] for k in op},
            "canonical_mean_AP50_95": canon["mean_ap5095"],
            "canonical_mean_AP50": canon["mean_ap50"],
            "shared_mAP50_95": shared["mean_ap5095"],
            "shared_mAP50": shared["mean_ap50"],
        }
        print(f"[{s}] conf={conf:.2f} meanF1={sweep['best_mean_F1']:.4f} "
              f"hf={op['header-footer']['F1']:.4f} ta={op['text-area']['F1']:.4f} "
              f"fn={op['footnote']['F1']:.4f} canonAP={canon['mean_ap5095']:.4f} "
              f"shared={shared['mean_ap5095']:.4f}", flush=True)

    def col(path):
        return [per_seed[s][path[0]] if len(path) == 1
                else per_seed[s][path[0]][path[1]] for s in SEEDS]

    summary = {
        "canonical_mean_F1": agg(col(("mean_F1",))),
        "header_footer_F1": agg(col(("F1", "header-footer"))),
        "text_area_F1": agg(col(("F1", "text-area"))),
        "footnote_F1": agg(col(("F1", "footnote"))),
        "canonical_mean_AP50_95": agg(col(("canonical_mean_AP50_95",))),
        "shared_mAP50_95": agg(col(("shared_mAP50_95",))),
        "operating_conf": agg(col(("operating_conf",))),
    }
    out = {
        "method": ("training-seed variance: 5 independent RT-DETR-l tam2col runs "
                   "(seeds 0-4), re-scored on the leak-free v4 833-page test with "
                   "the UNIFIED scorer (per-conf text-area envelope, 0.01 grid, "
                   "each seed at its own best-mean-F1 operating point). Supersedes "
                   "the old fixed-conf-0.50 band."),
        "recipe": ("rtdetr-l.pt, imgsz 1024, epochs 100, patience 20, batch 8, "
                   "deterministic=True; only the seed differs"),
        "per_seed": per_seed,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    for k, v in summary.items():
        print(f"  {k:24s} {v['mean']:.4f} ± {v['sample_std']:.4f} "
              f"(min {v['min']:.4f} / max {v['max']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
