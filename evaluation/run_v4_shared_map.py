#!/usr/bin/env python3
"""G1 + G2 diagnostic — DocLayNet-aligned SHARED 3-class mAP on the v4 test.

Re-scores the archived v4 prediction dumps (nothing re-inferred) in the
DocLayNet-aligned shared class space:

    header -> page-header, footer -> page-footer, footnote -> footnote,
    TEXT-AREA EXCLUDED (our page/column envelope is not comparable to
    DocLayNet's paragraph-level `Text`).

header and page-footer stay SEPARATE here (unlike the canonical header-footer
class). Same pycocotools COCOeval protocol as everywhere else (bbox, IoU
0.50:0.05:0.95, 101-pt, maxDets 100, conf floor 0.05).

Per system we report shared mAP@0.50:0.95 and @0.50, per-class AP, and a
box-convention diagnostic: at IoU>=0.1 vs >=0.5, per shared class the recall
(coverage) and mean matched-IoU — this separates loose localisation from
genuine misses (used for the G2 domain-transfer paragraph).

GT is the RAW v4 test annotations (native 4-class; text-area kept native but
excluded from the shared mean).

Usage:
  .venv_eval/bin/python evaluation/run_v4_shared_map.py \\
      --gt-dir   /path/TDLA@v4/labels/test \\
      --img-dir  /home/eroux/tmp/dataset_tdlav4_tam2col/images/test \\
      --sizes-cache evaluation/eval_results/tdla-v4/literature/test_image_sizes.json \\
      --pred-root /home/eroux/tmp/tdlav4_lit/preds \\
      --out-dir evaluation/eval_results/tdla-v4/shared_map
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from literature_metrics import (  # noqa: E402
    coco_map, iou_recall_diagnostic, iter_pages, load_sizes,
)
from run_v4_lit_eval import SYSTEMS  # noqa: E402

SHARED = ("page-header", "page-footer", "footnote")
DIAG_IOUS = (0.1, 0.5)


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--img-dir", type=Path, required=True)
    ap.add_argument("--sizes-cache", type=Path, default=None)
    ap.add_argument("--pred-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--systems", default="")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    sizes = load_sizes(args.img_dir, cache=args.sizes_cache)
    print(f"{len(sizes)} image sizes", flush=True)

    want = {s.strip() for s in args.systems.split(",") if s.strip()}
    results = []
    for sid, name, group, role in SYSTEMS:
        if want and sid not in want:
            continue
        pred = args.pred_root / sid / "labels"
        if not pred.is_dir():
            print(f"skip {sid}: no {pred}", file=sys.stderr)
            continue
        pages = list(iter_pages(args.gt_dir, pred, sizes, conf_floor=0.05))
        shared = coco_map(pages, "doclaynet", mean_over=SHARED)
        full = coco_map(pages, "doclaynet")  # incl text-area (reference)
        diag = iou_recall_diagnostic(pages, "doclaynet", SHARED, DIAG_IOUS)
        pc = shared["per_class"]
        rec = {
            "id": sid, "name": name, "group": group, "role": role,
            "dataset_tag": "v4", "n_pages": len(pages),
            "shared_ap50": shared["mean_ap50"],
            "shared_ap5095": shared["mean_ap5095"],
            "shared_per_class": {n: pc.get(n) for n in SHARED},
            "textarea_ap5095": full["per_class"].get("text-area", {}).get("ap5095"),
            "diagnostic": diag,
        }
        results.append(rec)
        (out / f"{sid}.json").write_text(json.dumps(rec, indent=2))
        print(f"{sid:<26} shared mAP50-95={_fmt(shared['mean_ap5095'])} "
              f"mAP50={_fmt(shared['mean_ap50'])}  "
              f"(ph={_fmt(pc['page-header']['ap5095'])} "
              f"pf={_fmt(pc['page-footer']['ap5095'])} "
              f"fn={_fmt(pc['footnote']['ap5095'])})", flush=True)

    (out / "shared_map.json").write_text(json.dumps(
        {"schema": "doclaynet-shared (page-header,page-footer,footnote)",
         "gt_dir": str(args.gt_dir), "n_systems": len(results),
         "systems": results}, indent=2))
    print(f"\nwrote {out/'shared_map.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
