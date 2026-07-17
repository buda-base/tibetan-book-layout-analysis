#!/usr/bin/env python3
"""Run a fine-tuned RF-DETR (Roboflow) checkpoint on images and write YOLO-format
predictions in our 4-class schema (0 header, 1 text-area, 2 footnote, 3 footer).

Resumable: skips images whose label file already exists.

Usage:
  python rfdetr_predict.py --checkpoint <pth> --source <img_dir> --out <out_dir>
                           [--conf 0.01] [--shape 1024] [--batch 1]
"""
from __future__ import annotations

import argparse
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--shape", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    src = Path(args.source)
    lbl_dir = Path(args.out) / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    todo = [p for p in imgs if not (lbl_dir / f"{p.stem}.txt").exists()]
    shape = (args.shape, args.shape)
    print(f"{len(imgs)} images ({len(todo)} to do) -> RF-DETR {args.checkpoint}, "
          f"conf {args.conf}, shape {shape}", flush=True)

    from rfdetr import RFDETRLarge

    model = RFDETRLarge.from_checkpoint(args.checkpoint)
    # class_names: ['none', 'header', 'text-area', 'footnote', 'footer']
    done = 0
    for i in range(0, len(todo), args.batch):
        chunk = todo[i : i + args.batch]
        paths = [str(p) for p in chunk]
        dets = model.predict(paths if len(paths) > 1 else paths[0],
                              threshold=args.conf, shape=shape)
        if not isinstance(dets, list):
            dets = [dets]
        for ip, det in zip(chunk, dets):
            from PIL import Image
            with Image.open(ip) as im:
                W, H = im.size
            lines = []
            if det is not None and len(det) > 0:
                for box, cls_id, score in zip(det.xyxy, det.class_id, det.confidence):
                    our_cls = int(cls_id) - 1
                    if our_cls < 0 or our_cls > 3:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    cx = ((x1 + x2) / 2) / W
                    cy = ((y1 + y2) / 2) / H
                    w = (x2 - x1) / W
                    h = (y2 - y1) / H
                    lines.append(f"{our_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {score:.4f}")
            (lbl_dir / f"{ip.stem}.txt").write_text("\n".join(lines))
            done += 1
        print(f"  {min(i + args.batch, len(todo))}/{len(todo)} ...", flush=True)

    print(f"done: {done} images -> {lbl_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
