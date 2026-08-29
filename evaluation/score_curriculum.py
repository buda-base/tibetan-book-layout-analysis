#!/usr/bin/env python3
"""Score the v5 curriculum ablation under the COCO protocol on our 860-page
test set, to move paper Tables 3-4 off VOC-only AP.

Each variant is scored under schema (a) canonical (header+footer merged,
text-area envelope, footnote) because all five heads can be evaluated there.
The 4-class heads (baseline/tam/tam2col) are additionally scored under schema
(b) DocLayNet-aligned. The 3-class heads (3cls/3cls_tam) cannot separate
page-header from page-footer, so schema (b) is skipped for them.

For each variant we report COCO AP@0.50 and AP@0.50:0.95 (per class + mean),
plus a best-mean-F1 confidence sweep (VOC tables quoted a single AP; COCO AP
is threshold-independent, and the sweep gives a comparable operating F1).

Usage:
  python score_curriculum.py \\
      --gt-dir /path/testset/labels/test \\
      --img-dir /path/testset/images/test \\
      --pred-root /path/curriculum_preds \\
      --out /path/curriculum_coco.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from literature_metrics import (  # noqa: E402
    best_f1_sweep, coco_map, iter_pages, load_sizes,
)

# variant id -> (pred subdir, class head, schemas to score)
VARIANTS = [
    ("rtdetr_v5_baseline", "baseline_pred/labels", "4cls", ("canonical", "doclaynet")),
    ("rtdetr_v5_tam", "tam_pred/labels", "4cls", ("canonical", "doclaynet")),
    ("rtdetr_v5_tam2col", "tam2col_pred/labels", "4cls", ("canonical", "doclaynet")),
    ("rtdetr_v5_3cls", "3cls_pred/labels", "3cls", ("canonical",)),
    ("rtdetr_v5_3cls_tam", "3cls_tam_pred/labels", "3cls", ("canonical",)),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--img-dir", type=Path, required=True)
    ap.add_argument("--pred-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sizes = load_sizes(args.img_dir,
                       cache=args.out.parent / "test_image_sizes.json")
    print(f"{len(sizes)} image sizes", flush=True)

    t0 = time.time()
    out = {"n_images": len(sizes), "variants": {}}
    for vid, subdir, head, schemas in VARIANTS:
        pdir = args.pred_root / subdir
        if not pdir.is_dir():
            print(f"SKIP {vid}: no dump at {pdir}", flush=True)
            continue
        pages = list(iter_pages(args.gt_dir, pdir, sizes, conf_floor=0.0))
        rec = {"head": head, "n_pages": len(pages)}
        mnames = pdir.parent / "model_names.json"
        if mnames.is_file():
            rec["model_names"] = json.loads(mnames.read_text())
        for schema in schemas:
            rec[f"coco_{schema}"] = coco_map(pages, schema)
            rec[f"sweep_{schema}"] = best_f1_sweep(pages, schema)
        out["variants"][vid] = rec
        c = rec["coco_canonical"]
        print(f"{vid} [{head}]: canonical mean AP50={c['mean_ap50']:.3f} "
              f"AP50-95={c['mean_ap5095']:.3f} "
              f"bestF1={rec['sweep_canonical']['best_mean_F1']:.3f}"
              f"@{rec['sweep_canonical']['best_mean_conf']}", flush=True)
    out["wall_s"] = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} in {out['wall_s']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
