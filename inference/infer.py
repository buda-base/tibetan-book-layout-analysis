#!/usr/bin/env python3
"""Run the Tibetan modern-book layout detector on one or more page images,
applying the recommended *per-class* confidence thresholds.

The model is a 4-class RT-DETR-l (header, text-area, footnote, footer). The
detector is deliberately recall-happy on the small marginal header/footer boxes,
so the single best operating point differs by class:

    header  (0)  conf >= 0.60      footnote (2)  conf >= 0.25
    text-area(1) conf >= 0.25      footer   (3)  conf >= 0.60

Raising the header/footer threshold to ~0.60 lifts their precision from ~0.83 to
~0.95 for only a ~0.02 recall cost (see the model card / blog post). Text-area
already comes out as one clean box per column (two on genuine two-column pages),
so no text-area post-processing is needed.

Usage:
    python infer.py --weights tibetan_book_layout.pt --source page.jpg
    python infer.py --weights tibetan_book_layout.pt --source pages/ --out preds
"""
from __future__ import annotations

import argparse
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Recommended per-class operating points (see model card). Use a single global
# 0.45 instead if you prefer one number for all classes.
CLASS_THRESHOLDS = {0: 0.60, 1: 0.25, 2: 0.25, 3: 0.60}
CONF_FLOOR = min(CLASS_THRESHOLDS.values())  # predict once at the lowest floor


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="path to the .pt weights")
    ap.add_argument("--source", required=True, help="image file or folder")
    ap.add_argument("--out", default=None,
                    help="optional folder to write YOLO-format .txt labels")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="0")
    ap.add_argument("--global-conf", type=float, default=None,
                    help="use ONE threshold for all classes instead of per-class")
    args = ap.parse_args()

    from ultralytics import RTDETR

    thresholds = ({c: args.global_conf for c in CLASS_THRESHOLDS}
                  if args.global_conf is not None else CLASS_THRESHOLDS)
    floor = min(thresholds.values())

    model = RTDETR(args.weights)
    names = model.names

    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(source=args.source, imgsz=args.imgsz, conf=floor,
                            device=args.device, stream=True, verbose=False)

    n_img = n_kept = 0
    for r in results:
        n_img += 1
        stem = Path(r.path).stem
        lines = []
        if r.boxes is not None:
            for cls, conf, xywhn in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                        r.boxes.xywhn.tolist()):
                cls = int(cls)
                if conf < thresholds.get(cls, floor):
                    continue
                x, y, w, h = xywhn
                lines.append((cls, conf, x, y, w, h))
        n_kept += len(lines)
        print(f"{stem}: {len(lines)} boxes")
        for cls, conf, x, y, w, h in lines:
            print(f"    {names[cls]:10} conf={conf:.3f}  "
                  f"cx={x:.3f} cy={y:.3f} w={w:.3f} h={h:.3f}")
        if out_dir:
            (out_dir / f"{stem}.txt").write_text(
                "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                        for c, _, x, y, w, h in lines))

    print(f"\n{n_img} images, {n_kept} boxes kept "
          f"(thresholds: {thresholds})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
