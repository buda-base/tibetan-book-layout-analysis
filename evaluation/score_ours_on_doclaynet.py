#!/usr/bin/env python3
"""Score our 4-class YOLO predictions against DocLayNet test labels, shared
classes only (page-header / page-footer / footnote). Text-area is reported
separately against DocLayNet `Text` (YOLO id 9) and flagged as a granularity
mismatch — it is NOT averaged into the shared mean.

Our native ids: 0 header, 1 text-area, 2 footnote, 3 footer.
DocLayNet YOLO ids: 5 Page-header, 4 Page-footer, 1 Footnote, 9 Text.

Usage:
  python score_ours_on_doclaynet.py \\
      --pred-dir <our_rtdetr_labels> \\
      --gt-dir <doclaynet_test/labels> \\
      --index <doclaynet_test/index.jsonl> \\
      --out metrics_ours_on_doclaynet.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make sibling import work when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from literature_metrics import (  # noqa: E402
    coco_map, contamination, cote_dataset, led_errors, operating_points,
    read_yolo,
)

# Map a DocLayNet-native YOLO file into our 4-class space for the shared
# classes, plus Text -> text-area (granularity mismatch, scored separately).
DL_TO_OURS = {
    5: 0,   # Page-header -> header
    4: 3,   # Page-footer -> footer
    1: 2,   # Footnote -> footnote
    9: 1,   # Text -> text-area (NOT averaged into shared mean)
}


def remap_dl(boxes):
    out = []
    for b in boxes:
        mapped = DL_TO_OURS.get(b["cls"])
        if mapped is None:
            continue
        out.append({**b, "cls": mapped})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.50,
                    help="operating point (tam2col paper conf)")
    args = ap.parse_args()

    index = [json.loads(ln) for ln in args.index.read_text().splitlines() if ln.strip()]
    pages = []
    n_miss_pred = 0
    for row in index:
        stem, W, H = row["stem"], int(row["width"]), int(row["height"])
        gt = remap_dl(read_yolo(args.gt_dir / f"{stem}.txt"))
        pp = args.pred_dir / f"{stem}.txt"
        if not pp.exists():
            n_miss_pred += 1
            pred = []
        else:
            pred = read_yolo(pp)
        pages.append((stem, W, H, gt, pred))
    print(f"{len(pages)} pages ({n_miss_pred} missing pred files)", flush=True)

    shared = ("page-header", "page-footer", "footnote")
    coco_all = coco_map(pages, "doclaynet")
    coco_shared = coco_map(pages, "doclaynet", mean_over=shared)
    op = operating_points(pages, "doclaynet", args.conf)
    blob = {
        "n_pages": len(pages),
        "n_missing_pred": n_miss_pred,
        "conf": args.conf,
        "coco_shared": coco_shared,
        "coco_all_four_including_textarea": coco_all,
        "textarea_granularity_mismatch": True,
        "textarea_note": (
            "Our text-area is a page/column envelope; DocLayNet Text is "
            "paragraph-level. Do not average text-area AP into the shared mean."
        ),
        "op_doclaynet": op,
        "contamination": contamination(pages, args.conf),
        "led": led_errors(pages, args.conf),
        "cote": cote_dataset(pages, args.conf, max_dim=1024),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blob, indent=2))
    s = coco_shared
    print(f"shared AP50={s['mean_ap50']:.4f}  AP50-95={s['mean_ap5095']:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
