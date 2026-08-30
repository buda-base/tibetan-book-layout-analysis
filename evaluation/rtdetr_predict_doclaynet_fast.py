#!/usr/bin/env python3
"""Fast DocLayNet v1.2 test -> RT-DETR-l preds + GT (batched, local parquet).

Same outputs and label conventions as rtdetr_predict_doclaynet_stream.py, but:
  * downloads the test parquet ONCE (non-streaming) then reads locally — removes
    the per-row HTTP round-trip latency that pinned the streaming version at
    ~0% GPU;
  * feeds PIL images straight to model.predict in batches (no temp-PNG
    re-encode/reload), so the A10G is actually used.

Writes:
  <out>/pred_labels/<stem>.txt   our 4-class YOLO (cls cx cy w h conf)
  <out>/gt_labels/<stem>.txt     DocLayNet YOLO ids (category_id - 1)
  <out>/index.jsonl              stem, width, height, doc_category

Our native ids: 0 header, 1 text-area, 2 footnote, 3 footer.

Usage:
  python rtdetr_predict_doclaynet_fast.py \\
      --weights rtdetr_tdlav4_seed0.pt --out /root/dlt/out \\
      [--conf 0.05] [--imgsz 1024] [--device 0] [--batch 16] [--max-pages N]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image


def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    return ((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H, (x2 - x1) / W, (y2 - y1) / H


def gt_lines(bboxes, cats, W, H):
    lines = []
    for bb, cid in zip(bboxes, cats):
        if len(bb) != 4:
            continue
        cid = int(cid)
        x, y, bw, bh = (float(v) for v in bb)
        cx, cy, w, h = xyxy_to_yolo(x, y, x + bw, y + bh, W, H)
        lines.append(f"{cid - 1} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def pred_lines(result):
    Hm, Wm = result.orig_shape
    out = []
    if result.boxes is not None:
        for b, cf, cl in zip(result.boxes.xyxy.tolist(),
                             result.boxes.conf.tolist(),
                             result.boxes.cls.tolist()):
            x1, y1, x2, y2 = b
            cx = ((x1 + x2) / 2) / Wm
            cy = ((y1 + y2) / 2) / Hm
            w = (x2 - x1) / Wm
            h = (y2 - y1) / Hm
            out.append(f"{int(cl)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {cf:.4f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-pages", type=int, default=0)
    ap.add_argument("--dataset", default="docling-project/DocLayNet-v1.2")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    from datasets import load_dataset
    from ultralytics import RTDETR

    pred_dir = args.out / "pred_labels"
    gt_dir = args.out / "gt_labels"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.jsonl"

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"downloading {args.dataset} split={args.split} (non-streaming)",
          flush=True)
    ds = load_dataset(args.dataset, split=args.split, token=token)
    n_total = len(ds) if not args.max_pages else min(args.max_pages, len(ds))
    print(f"{n_total} pages cached locally; loading RTDETR {args.weights}",
          flush=True)
    model = RTDETR(args.weights)

    def flush(batch_imgs, batch_meta, idxf):
        results = model.predict(batch_imgs, conf=args.conf, imgsz=args.imgsz,
                                device=args.device, verbose=False)
        for r, (stem, W, H, doc_cat) in zip(results, batch_meta):
            (pred_dir / f"{stem}.txt").write_text("\n".join(pred_lines(r)))
            idxf.write(json.dumps({"stem": stem, "width": W, "height": H,
                                   "doc_category": doc_cat}) + "\n")

    n = 0
    batch_imgs, batch_meta = [], []
    with index_path.open("w") as idxf:
        for row in ds:
            n += 1
            if args.max_pages and n > args.max_pages:
                break
            stem = f"dl_{n:05d}"
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
            (gt_dir / f"{stem}.txt").write_text(
                "\n".join(gt_lines(row["bboxes"], row["category_id"], W, H)))
            batch_imgs.append(im)
            batch_meta.append((stem, W, H, meta.get("doc_category")))
            if len(batch_imgs) >= args.batch:
                flush(batch_imgs, batch_meta, idxf)
                batch_imgs, batch_meta = [], []
                if n % 500 == 0:
                    print(f"  {n}/{n_total} pages...", flush=True)
        if batch_imgs:
            flush(batch_imgs, batch_meta, idxf)
    print(f"done {n} pages -> {args.out}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
