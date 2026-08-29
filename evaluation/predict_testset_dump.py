#!/usr/bin/env python3
"""Run an Ultralytics RT-DETR checkpoint on a folder of test images and write
YOLO-format prediction dumps (cls cx cy w h conf), one .txt per image.

Used to re-score the v5 curriculum ablation (baseline / tam / 3cls / 3cls_tam)
under the COCO protocol. Prints `model.names` so the class-id -> name mapping
of each checkpoint is recorded next to the dump.

Usage:
  python predict_testset_dump.py --weights rtdetr_v5_baseline_best.pt \\
      --img-dir /path/to/testset/images/test \\
      --out /path/to/baseline_pred/labels \\
      [--conf 0.001] [--imgsz 1024] [--device 0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--img-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--conf", type=float, default=0.001,
                    help="low floor so COCO AP sees the full PR curve")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import RTDETR

    args.out.mkdir(parents=True, exist_ok=True)
    model = RTDETR(args.weights)
    print("model.names =", json.dumps(model.names), flush=True)
    (args.out.parent / "model_names.json").write_text(json.dumps(model.names))

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
    imgs = sorted(p for p in args.img_dir.iterdir() if p.suffix.lower() in exts)
    print(f"{len(imgs)} images -> {args.out}", flush=True)

    n = 0
    for p in imgs:
        r = model.predict(source=str(p), conf=args.conf, imgsz=args.imgsz,
                          device=args.device, verbose=False)[0]
        Hm, Wm = r.orig_shape
        lines = []
        if r.boxes is not None:
            for b, cf, cl in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                 r.boxes.cls.tolist()):
                x1, y1, x2, y2 = b
                cx = ((x1 + x2) / 2) / Wm
                cy = ((y1 + y2) / 2) / Hm
                w = (x2 - x1) / Wm
                h = (y2 - y1) / Hm
                lines.append(f"{int(cl)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {cf:.4f}")
        (args.out / f"{p.stem}.txt").write_text("\n".join(lines))
        n += 1
        if n % 100 == 0:
            print(f"  {n}/{len(imgs)}", flush=True)
    print(f"done {n} images", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
