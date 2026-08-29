#!/usr/bin/env python3
"""Stream DocLayNet v1.2 test, run our RT-DETR-l, write YOLO preds + GT labels.

Keeps almost no images on disk (one temp PNG at a time). Writes:

  <out>/pred_labels/<stem>.txt     our 4-class YOLO (cls cx cy w h conf)
  <out>/gt_labels/<stem>.txt       DocLayNet YOLO (0-10)
  <out>/index.jsonl                stem, width, height, doc_category

Our native ids: 0 header, 1 text-area, 2 footnote, 3 footer.

Usage:
  python rtdetr_predict_doclaynet_stream.py \\
      --weights tibetan_book_layout.pt --out /tmp/dl_transfer \\
      [--conf 0.05] [--imgsz 1024] [--device 0] [--max-pages N]
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from PIL import Image


def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    cx = ((x1 + x2) / 2) / W
    cy = ((y1 + y2) / 2) / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return cx, cy, w, h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-pages", type=int, default=0)
    ap.add_argument("--dataset", default="docling-project/DocLayNet-v1.2",
                    help="HF dataset id (v1.2 test split; PNG page + COCO boxes).")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    from datasets import load_dataset
    from ultralytics import RTDETR

    pred_dir = args.out / "pred_labels"
    gt_dir = args.out / "gt_labels"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.jsonl"

    print(f"loading RTDETR {args.weights}", flush=True)
    model = RTDETR(args.weights)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        print("Hugging Face auth: read token from env (download only)", flush=True)
    else:
        print("WARNING: no HF_TOKEN in env; Hub downloads will be unauthenticated",
              flush=True)
    print(f"streaming {args.dataset} split={args.split}", flush=True)
    ds = load_dataset(args.dataset, split=args.split, streaming=True, token=token)
    # Drop PDF blobs so we only pull page image + boxes (read-only).
    keep = [c for c in ("image", "bboxes", "category_id", "metadata")
            if c in (getattr(ds, "column_names", None) or
                     ["image", "bboxes", "category_id", "metadata"])]
    try:
        ds = ds.select_columns(keep)
    except Exception as exc:
        print(f"(select_columns skipped: {exc})", flush=True)

    n = 0
    already = {p.stem for p in pred_dir.glob("*.txt")}
    with index_path.open("a" if already else "w") as idxf:
        for row in ds:
            n += 1
            if args.max_pages and n > args.max_pages:
                break
            stem = f"dl_{n:05d}"
            if stem in already and (gt_dir / f"{stem}.txt").exists():
                continue
            im = row["image"]
            if not isinstance(im, Image.Image):
                im = Image.fromarray(im)
            im = im.convert("RGB")
            W, H = im.size
            meta = row.get("metadata") or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    meta = {}

            # GT in DocLayNet YOLO ids
            lines = []
            bboxes = row["bboxes"]
            cats = row["category_id"]
            for bb, cid in zip(bboxes, cats):
                cid = int(cid)
                yolo_c = cid - 1
                if len(bb) != 4:
                    continue
                x, y, bw, bh = (float(v) for v in bb)
                cx, cy, w, h = xyxy_to_yolo(x, y, x + bw, y + bh, W, H)
                lines.append(f"{yolo_c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            (gt_dir / f"{stem}.txt").write_text("\n".join(lines))

            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                im.save(tmp.name)
                results = model.predict(
                    source=tmp.name, conf=args.conf, imgsz=args.imgsz,
                    device=args.device, verbose=False,
                )
            r = results[0]
            Hm, Wm = r.orig_shape
            plines = []
            if r.boxes is not None:
                for b, cf, cl in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                     r.boxes.cls.tolist()):
                    x1, y1, x2, y2 = b
                    cx = ((x1 + x2) / 2) / Wm
                    cy = ((y1 + y2) / 2) / Hm
                    w = (x2 - x1) / Wm
                    h = (y2 - y1) / Hm
                    plines.append(f"{int(cl)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {cf:.4f}")
            (pred_dir / f"{stem}.txt").write_text("\n".join(plines))
            idxf.write(json.dumps({
                "stem": stem, "width": W, "height": H,
                "doc_category": meta.get("doc_category"),
            }) + "\n")
            idxf.flush()
            if n % 50 == 0:
                print(f"  {n} pages...", flush=True)
    print(f"done {n} pages -> {args.out}", flush=True)
    # Streaming IterableDataset teardown can hang on Hub HTTP; hard-exit
    # after files are flushed so nohup jobs actually finish.
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
