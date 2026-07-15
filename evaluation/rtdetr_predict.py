#!/usr/bin/env python3
"""Run our RT-DETR-L layout model on a folder of images and write YOLO-format
predictions in our native 4-class schema (0 header, 1 text-area, 2 footnote,
3 footer) with a confidence column. Batched to bound GPU memory so it can run
alongside another job. Resumable.

Usage:
  python rtdetr_predict.py --weights <pt> --source <img_dir> --out <out_dir>
                           [--conf 0.05] [--imgsz 1024] [--device 0] [--batch 4]
"""
from __future__ import annotations

import argparse
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    src = Path(args.source)
    lbl_dir = Path(args.out) / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    todo = [p for p in imgs if not (lbl_dir / f"{p.stem}.txt").exists()]
    print(f"{len(imgs)} images ({len(todo)} to do) -> RT-DETR, conf {args.conf}",
          flush=True)

    from ultralytics import RTDETR
    model = RTDETR(args.weights)
    for i in range(0, len(todo), args.batch):
        chunk = [str(p) for p in todo[i : i + args.batch]]
        results = model.predict(source=chunk, conf=args.conf, imgsz=args.imgsz,
                                device=args.device, stream=False, verbose=False)
        for src_path, r in zip(todo[i : i + args.batch], results):
            H, W = r.orig_shape
            lines = []
            if r.boxes is not None:
                for b, cf, cl in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                     r.boxes.cls.tolist()):
                    x1, y1, x2, y2 = b
                    cx = ((x1 + x2) / 2) / W
                    cy = ((y1 + y2) / 2) / H
                    w = (x2 - x1) / W
                    h = (y2 - y1) / H
                    lines.append(f"{int(cl)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {cf:.4f}")
            (lbl_dir / f"{src_path.stem}.txt").write_text("\n".join(lines))
        print(f"  {min(i + args.batch, len(todo))}/{len(todo)} ...", flush=True)
    print(f"done -> {lbl_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
