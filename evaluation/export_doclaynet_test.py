#!/usr/bin/env python3
"""Export DocLayNet v1.2 *test* split to YOLO labels + a page index.

Streams `docling-project/DocLayNet-v1.2` (content-identical to v1.0; v1.2 only
changes packaging). Writes:

  <out>/images/*.png          (optional; --skip-images keeps only labels)
  <out>/labels/*.txt          YOLO: cls cx cy w h   (DocLayNet native ids 0-10)
  <out>/index.jsonl           stem, width, height, doc_category
  <out>/coco_shared.json      COCO GT for the 3 shared classes only
                                1 Page-header  2 Page-footer  3 Footnote

DocLayNet category_id in this dataset is 1-indexed:
  1 Caption, 2 Footnote, 3 Formula, 4 List-item, 5 Page-footer,
  6 Page-header, 7 Picture, 8 Section-header, 9 Table, 10 Text, 11 Title

YOLO class = category_id - 1.

Usage:
  python export_doclaynet_test.py --out /data/doclaynet_test [--skip-images]
                                  [--max-pages N]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# DocLayNet 1-indexed -> YOLO 0-indexed name
DOCLAYNET_YOLO = {
    0: "Caption", 1: "Footnote", 2: "Formula", 3: "List-item",
    4: "Page-footer", 5: "Page-header", 6: "Picture", 7: "Section-header",
    8: "Table", 9: "Text", 10: "Title",
}
# shared classes as YOLO ids
SHARED_YOLO = {1: "footnote", 4: "page-footer", 5: "page-header"}
# our native 4-class: header=0, text-area=1, footnote=2, footer=3
DOCLAYNET_TO_OURS = {5: 0, 4: 3, 1: 2}  # page-header, page-footer, footnote
# body-ish classes that can form a text-area envelope for contamination
BODY_YOLO = {9, 3, 7, 10, 0}  # Text, List-item, Section-header, Title, Caption


def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    cx = ((x1 + x2) / 2) / W
    cy = ((y1 + y2) / 2) / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return cx, cy, w, h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--max-pages", type=int, default=0)
    ap.add_argument("--dataset", default="docling-project/DocLayNet-v1.2")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    from datasets import load_dataset
    from PIL import Image

    out = args.out
    img_dir = out / "images"
    lbl_dir = out / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_images:
        img_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"streaming {args.dataset} split={args.split}", flush=True)
    ds = load_dataset(args.dataset, split=args.split, streaming=True, token=token)

    images_coco, anns, index = [], [], []
    ann_id = 1
    n = 0
    for row in ds:
        n += 1
        if args.max_pages and n > args.max_pages:
            break
        im = row["image"]
        if not isinstance(im, Image.Image):
            im = Image.fromarray(im)
        W, H = im.size
        stem = f"dl_{n:05d}"
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        if not args.skip_images:
            im.convert("RGB").save(img_dir / f"{stem}.png", optimize=True)
        bboxes = row["bboxes"]
        cats = row["category_id"]
        # bboxes are typically [x, y, w, h] in pixels (COCO)
        lines = []
        images_coco.append({"id": n, "file_name": f"{stem}.png",
                            "width": W, "height": H})
        for bb, cid in zip(bboxes, cats):
            cid = int(cid)
            yolo_c = cid - 1
            if len(bb) == 4:
                x, y, bw, bh = (float(v) for v in bb)
                x1, y1, x2, y2 = x, y, x + bw, y + bh
            else:
                continue
            cx, cy, w, h = xyxy_to_yolo(x1, y1, x2, y2, W, H)
            lines.append(f"{yolo_c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            if yolo_c in SHARED_YOLO:
                # COCO cats: 1 page-header, 2 page-footer, 3 footnote
                coco_cid = {5: 1, 4: 2, 1: 3}[yolo_c]
                anns.append({
                    "id": ann_id, "image_id": n, "category_id": coco_cid,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
                    "segmentation": [],
                })
                ann_id += 1
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
        index.append({"stem": stem, "width": W, "height": H,
                      "doc_category": meta.get("doc_category")})
        if n % 200 == 0:
            print(f"  {n} pages...", flush=True)

    coco = {
        "images": images_coco,
        "annotations": anns,
        "categories": [
            {"id": 1, "name": "page-header"},
            {"id": 2, "name": "page-footer"},
            {"id": 3, "name": "footnote"},
        ],
    }
    (out / "coco_shared.json").write_text(json.dumps(coco))
    (out / "index.jsonl").write_text("\n".join(json.dumps(r) for r in index) + "\n")
    print(f"wrote {n} pages -> {out}  shared GT boxes={len(anns)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
