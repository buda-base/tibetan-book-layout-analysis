#!/usr/bin/env python3
"""Score the 5-curriculum RT-DETR-l ablation on the leak-free v4 833-page test
with the UNIFIED evaluator (literature_metrics, per-conf text-area envelope,
fine 0.01 grid).

Five label variants, all RT-DETR-l (rtdetr-l.pt, imgsz 1024, 100 ep / patience
20, batch 8, seed 0, deterministic):
  baseline  4 classes, text-area left as multiple boxes
  tam       text-area merged to one envelope
  tam2col   tam, but genuine two-column pages keep two boxes  (== production
            RT-DETR-l v4 fine-tune; reuses the seed0 dump)
  3cls      header+footer merged into one training class
  3cls_tam  both merges

For every variant we report, on the v4 test:
  * canonical mean F1 + per-class F1 at the best-mean-F1 operating point
  * canonical AP50 / AP50-95 (per class + mean)
  * NATIVE text-area AP (no eval-merge), two definitions:
      (a) vs each variant's OWN test labels (multi-box for baseline, envelope
          for tam, two-column-aware for tam2col) -- measures how well the model
          reproduces its own training target (envelope predictors trivially
          high);
      (b) vs a COMMON un-merged multi-box GT (baseline's raw labels) -- the
          cross-curriculum yardstick that exposes the merge measurement
          artifact; leak-free analog of the old 0.857 / 0.634 numbers.

Canonical GT is the shared v4 test (same GT dir as the 13-system table); native
GT is each variant's own curriculum labels.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from literature_metrics import (  # noqa: E402
    _coco_eval, best_f1_sweep, coco_map, iter_pages, load_sizes,
    operating_points, to_px,
)

# variant -> (display, S3 pred sub-path OR local dump, native GT dir)
S3 = "s3://bec.bdrc.io/models/hff-detection/tdlav4"


def native_ta_ap(gt_dir: Path, pred_dir: Path, sizes):
    """Single-class COCO AP for text-area (cls 1), in the native label scheme
    (no envelope merge applied at eval time)."""
    cats = [{"id": 1, "name": "text-area"}]
    images, gt_anns, dt_anns = [], [], []
    ann_id = 1
    for img_id, (stem, W, H, g, p) in enumerate(
            iter_pages(gt_dir, pred_dir, sizes, conf_floor=0.0), 1):
        images.append({"id": img_id, "file_name": f"{stem}.jpg",
                       "width": W, "height": H})
        for b in g:
            if b["cls"] != 1:
                continue
            x1, y1, x2, y2 = to_px(b, W, H)
            bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
            gt_anns.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                            "bbox": [x1, y1, bw, bh], "area": bw * bh,
                            "iscrowd": 0, "ignore": 0, "segmentation": []})
            ann_id += 1
        for b in p:
            if b["cls"] != 1:
                continue
            x1, y1, x2, y2 = to_px(b, W, H)
            bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
            dt_anns.append({"image_id": img_id, "category_id": 1,
                            "bbox": [x1, y1, bw, bh], "score": float(b["conf"])})
    per, _, _ = _coco_eval(images, cats, gt_anns, dt_anns)
    return per["text-area"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path,
                    default=Path("/home/eroux/tmp/dataset_tdlav4_tam2col/labels/test"))
    ap.add_argument("--img-dir", type=Path,
                    default=Path("/home/eroux/tmp/dataset_tdlav4_tam2col/images/test"))
    ap.add_argument("--preds-root", type=Path,
                    default=Path("/home/eroux/tmp/curricula_preds"))
    ap.add_argument("--native-gt-root", type=Path,
                    default=Path("/home/eroux/tmp/curricula"),
                    help="dir with <mode>/labels/test for tam/3cls/3cls_tam/tam2col")
    ap.add_argument("--baseline-native-gt", type=Path,
                    default=Path("/home/eroux/tmp/tdla-promote-v2/canonical-git/labels/test"))
    ap.add_argument("--tam2col-dump", type=Path,
                    default=Path("/home/eroux/tmp/tdlav4_lit/preds/rtdetr_tdlav4/labels"),
                    help="reused production seed0 tam2col dump")
    ap.add_argument("--out", type=Path,
                    default=Path("/home/eroux/BUDA/softs/tibetan-book-layout-analysis/"
                                 "evaluation/eval_results/tdla-v4/curriculum/metrics.json"))
    ap.add_argument("--pull", action="store_true", help="aws s3 cp the 4 new dumps first")
    args = ap.parse_args()

    args.preds_root.mkdir(parents=True, exist_ok=True)
    new_modes = ["baseline", "tam", "3cls", "3cls_tam"]
    if args.pull:
        for m in new_modes:
            dst = args.preds_root / m / "labels"
            dst.mkdir(parents=True, exist_ok=True)
            src = f"{S3}/eval/rtdetr_tdlav4_{m}/preds/labels/"
            print(f"pull {src} -> {dst}", flush=True)
            subprocess.run(["aws", "s3", "cp", src, str(dst), "--recursive",
                            "--only-show-errors"], check=True)

    sizes = load_sizes(args.img_dir, cache=args.out.parent / "test_image_sizes.json")
    print(f"{len(sizes)} test image sizes", flush=True)

    # variant -> pred dir (dump) + native GT dir
    variants = {}
    for m in new_modes:
        variants[m] = (args.preds_root / m / "labels",
                       (args.baseline_native_gt if m == "baseline"
                        else args.native_gt_root / m / "labels" / "test"))
    variants["tam2col"] = (args.tam2col_dump,
                           args.native_gt_root / "tam2col" / "labels" / "test")

    # common un-merged (multi-box) text-area GT: the old-note yardstick that
    # scores every variant's text-area against the SAME native multi-box GT,
    # so the merge artifact is visible cross-curriculum (analog of 0.857/0.634).
    common_native_gt = args.baseline_native_gt

    order = ["baseline", "tam", "tam2col", "3cls", "3cls_tam"]
    results = {}
    for m in order:
        pred_dir, native_gt = variants[m]
        if not pred_dir.is_dir() or not any(pred_dir.glob("*.txt")):
            print(f"!! {m}: no dump at {pred_dir}; skipping", file=sys.stderr)
            continue
        pages = list(iter_pages(args.gt_dir, pred_dir, sizes, conf_floor=0.0))
        sweep = best_f1_sweep(pages, "canonical")
        conf = sweep["best_mean_conf"]
        op = operating_points(pages, "canonical", conf)["per_class"]
        coco = coco_map(pages, "canonical")
        # (a) native vs each variant's OWN scheme (reproduces its training target)
        native_own = native_ta_ap(native_gt, pred_dir, sizes) if native_gt.is_dir() else \
            {"ap50": float("nan"), "ap5095": float("nan")}
        # (b) native vs COMMON multi-box GT (cross-curriculum artifact yardstick)
        native_common = native_ta_ap(common_native_gt, pred_dir, sizes) \
            if common_native_gt.is_dir() else {"ap50": float("nan"), "ap5095": float("nan")}
        results[m] = {
            "mode": m, "n_pages": len(pages), "operating_conf": conf,
            "mean_F1": sweep["best_mean_F1"],
            "F1": {k: op[k]["F1"] for k in op},
            "PR": {k: {"P": op[k]["P"], "R": op[k]["R"]} for k in op},
            "coco_canonical": {"per_class": coco["per_class"],
                               "mean_ap50": coco["mean_ap50"],
                               "mean_ap5095": coco["mean_ap5095"]},
            "native_text_area_AP_own_scheme": native_own,
            "native_text_area_AP_vs_common_multibox_gt": native_common,
        }
        print(f"[{m}] meanF1={sweep['best_mean_F1']:.3f} @conf {conf:.2f}  "
              f"hf={op['header-footer']['F1']:.3f} ta={op['text-area']['F1']:.3f} "
              f"fn={op['footnote']['F1']:.3f}  canonAP5095={coco['mean_ap5095']:.3f}  "
              f"nativeTA(own)={native_own['ap5095']:.3f} "
              f"nativeTA(commonGT)={native_common['ap5095']:.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"recipe": (
        "RT-DETR-l rtdetr-l.pt, imgsz 1024, epochs 100, patience 20, batch 8, "
        "seed 0, deterministic; ultralytics 8.4.135 (baseline/tam/3cls/3cls_tam); "
        "tam2col = production seed0 fleet dump"),
        "gt_dir": str(args.gt_dir), "variants": results}, indent=2))
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
