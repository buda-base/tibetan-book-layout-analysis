#!/usr/bin/env python3
"""Test-set bootstrap noise floor for the canonical mean-F1 architecture
comparison (paper Tables / Limitations "0.93-0.96 band").

All fine-tuned checkpoints are single training runs, so there is no training-
seed variance. This script instead quantifies the *evaluation* noise: how much
each system's canonical mean-F1 wobbles under resampling of the 860-page test
set, and -- crucially -- the distribution of the *paired* difference between
two systems (same resampled pages), i.e. whether a gap like RT-DETR-l 0.957 vs
RF-DETR-L 0.953 is distinguishable from zero on this test set.

Method: at each system's paper operating confidence, per-page (tp, fp, fn) are
precomputed for the three canonical classes (header-footer, text-area envelope,
footnote). Under a fixed threshold these counts are additive across pages, so a
bootstrap resample is an exact recomputation, not an approximation. We draw B
resamples of the 860 page indices with replacement; for each we recompute every
system's mean-F1 (mean over the 3 class F1s) and every pairwise difference,
sharing the same indices so differences are paired.

This is a test-set / evaluation noise floor. It does NOT capture training-seed
variance (that needs multi-seed retraining -- see the note printed at the end).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from literature_metrics import (  # noqa: E402
    _pr_class, apply_schema, iter_pages, load_sizes,
)

ROOT = Path("/home/eroux/azure_di_eval")
# id -> (name, pred_dir, paper operating conf)
SYSTEMS = {
    "rtdetr_tam2col": ("RT-DETR-l tam2col", ROOT / "tam2col_pred/labels", 0.50),
    "rfdetr_tam2col": ("RF-DETR-L tam2col", ROOT / "ft_preds/rfdetr/rfdetr_tam2col_pred/labels", 0.30),
    "doclayout_yolo_ft": ("DocLayout-YOLO tam2col", ROOT / "ft_preds/dl_yolo/dl_yolo_ft_pred/labels", 0.30),
    "pp_doclayout_ft": ("PP-DocLayout-L tam2col", ROOT / "ft_preds/pp_doclayout/pp_doclayout_ft_pred/labels", 0.75),
    "docling_heron_ft": ("Docling layout-heron tam2col", ROOT / "ft_preds/docling_heron/docling_heron_ft_pred/labels", 0.05),
}
CANON = ["header-footer", "text-area", "footnote"]


def per_page_counts(pages, conf):
    recs = {n: [] for n in CANON}
    for stem, W, H, g, p in pages:
        gg = apply_schema(g, "canonical", W, H)
        pp = apply_schema(p, "canonical", W, H)
        for n in CANON:
            gt_list = {stem: [xy for xy, _ in gg[n]]} if gg[n] else {}
            pred_list = {stem: pp[n]} if pp[n] else {}
            r = _pr_class(gt_list, pred_list, conf)
            recs[n].append((r["tp"], r["fp"], r["fn"]))
    return {n: np.array(v, dtype=np.int64) for n, v in recs.items()}  # (P,3)


def mean_f1(recs, idx):
    f1s = []
    for n in CANON:
        arr = recs[n][idx]
        tp, fp, fn = arr[:, 0].sum(), arr[:, 1].sum(), arr[:, 2].sum()
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * P * R / (P + R) if P + R else 0.0)
    return float(np.mean(f1s))


def ci(a, lo=2.5, hi=97.5):
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path,
                    default=Path("/home/eroux/azure_di_eval/testset/labels/test"))
    ap.add_argument("--img-dir", type=Path,
                    default=Path("/home/eroux/azure_di_eval/testset/images/test"))
    ap.add_argument("--out", type=Path, default=Path(
        "/home/eroux/BUDA/softs/tibetan-book-layout-analysis/evaluation/"
        "eval_results/literature/bootstrap_variance.json"))
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sizes = load_sizes(args.img_dir, cache=args.out.parent / "test_image_sizes.json")
    counts = {}
    full = {}
    for sid, (name, pdir, conf) in SYSTEMS.items():
        if not pdir.is_dir():
            print(f"SKIP {sid}: no dump at {pdir}", flush=True)
            continue
        pages = list(iter_pages(args.gt_dir, pdir, sizes, conf_floor=0.0))
        counts[sid] = per_page_counts(pages, conf)
        n = len(pages)
        full[sid] = mean_f1(counts[sid], np.arange(n))
        print(f"{sid}: full-set canonical mean-F1 = {full[sid]:.4f} "
              f"(conf {conf}, {n} pages)", flush=True)

    ids = list(counts.keys())
    n = counts[ids[0]][CANON[0]].shape[0]
    rng = np.random.default_rng(args.seed)
    boot = {sid: np.empty(args.B) for sid in ids}
    for b in range(args.B):
        idx = rng.integers(0, n, n)
        for sid in ids:
            boot[sid][b] = mean_f1(counts[sid], idx)

    per_system = {}
    for sid in ids:
        lo, hi = ci(boot[sid])
        per_system[sid] = {
            "name": SYSTEMS[sid][0], "conf": SYSTEMS[sid][2],
            "full_mean_F1": full[sid],
            "boot_mean": float(boot[sid].mean()),
            "boot_std": float(boot[sid].std(ddof=1)),
            "ci95": [lo, hi],
        }
        print(f"  {sid:18s} F1={full[sid]:.4f}  std={boot[sid].std(ddof=1):.4f}  "
              f"95% CI [{lo:.4f}, {hi:.4f}]", flush=True)

    # paired differences vs RT-DETR-l (the paper's headline system)
    ref = "rtdetr_tam2col"
    pairs = {}
    print(f"\nPaired differences vs {ref} (positive => {ref} better):", flush=True)
    for sid in ids:
        if sid == ref:
            continue
        d = boot[ref] - boot[sid]
        lo, hi = ci(d)
        p_ref_better = float((d > 0).mean())
        full_d = full[ref] - full[sid]
        pairs[f"{ref}_minus_{sid}"] = {
            "full_diff": full_d, "boot_mean_diff": float(d.mean()),
            "ci95": [lo, hi], "p_ref_better": p_ref_better,
            "crosses_zero": bool(lo < 0 < hi),
        }
        print(f"  {ref} - {sid:18s} = {full_d:+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"P({ref}>{sid})={p_ref_better:.2f}  "
              f"{'OVERLAPS 0' if lo < 0 < hi else 'separated'}", flush=True)

    out = {
        "method": "test-set bootstrap (paired), canonical mean-F1 at paper conf",
        "B": args.B, "seed": args.seed, "n_pages": n,
        "classes": CANON,
        "per_system": per_system,
        "paired_vs_" + ref: pairs,
        "caveat": ("Test-set/evaluation noise only. Training-seed variance is "
                   "NOT captured; each checkpoint is one run. A multi-seed study "
                   "would add training noise on top of this floor."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
