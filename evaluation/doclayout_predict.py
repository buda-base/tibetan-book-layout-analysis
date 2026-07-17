#!/usr/bin/env python3
"""Run DocLayout-YOLO (DocStructBench, off-the-shelf, no fine-tuning) on a folder
of images and write YOLO-format predictions in OUR 4-class schema:

    0 header    1 text-area    2 footnote    3 footer

DocStructBench classes -> our schema:
    0 title          -> 1 text-area
    1 plain text      -> 1 text-area
    2 abandon         -> 0 header  (headers/footers/page numbers to discard;
                                    canonically merged with footer anyway)
    7 table_footnote  -> 2 footnote (closest thing DocLayout has to a footnote)
    everything else (figure, caption, table, formula) -> dropped

Text-area boxes are left individual; the canonical evaluator merges them into one
envelope per page, so this stays apples-to-apples with our own models.

Output: one <stem>.txt per image with rows "cls cx cy w h conf" (normalized).
Resumable: images whose label file already exists are skipped.

Usage:
  python doclayout_predict.py --weights <pt> --source <img_dir> --out <out_dir>
                              [--conf 0.05] [--imgsz 1024] [--device 0] [--limit N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

DOCSTRUCT_MAP = {0: 1, 1: 1, 2: 0, 7: 2}  # else -> drop
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--native-classes", action="store_true",
                    help="Fine-tuned on our 4-class schema; pass class ids through")
    args = ap.parse_args()

    src = Path(args.source)
    lbl_dir = Path(args.out) / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    if args.limit:
        imgs = imgs[: args.limit]
    todo = [p for p in imgs if not (lbl_dir / f"{p.stem}.txt").exists()]
    print(f"{len(imgs)} images ({len(todo)} to do, {len(imgs) - len(todo)} cached) "
          f"-> DocLayout-YOLO (DocStructBench), conf {args.conf}, imgsz {args.imgsz}",
          flush=True)

    from doclayout_yolo import YOLOv10
    model = YOLOv10(args.weights)

    done = 0
    kept = {}
    # Process in small batches: passing the whole file list at once makes
    # ultralytics preload every image into RAM and OOMs small instances.
    for i in range(0, len(todo), args.batch):
        chunk = [str(p) for p in todo[i : i + args.batch]]
        results = model.predict(source=chunk, conf=args.conf, imgsz=args.imgsz,
                                device=args.device, stream=False, verbose=False)
        for r in results:
            H, W = r.orig_shape
            stem = Path(r.path).stem
            lines = []
            if r.boxes is not None:
                for b, cf, cl in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                     r.boxes.cls.tolist()):
                    cls = int(cl) if args.native_classes else DOCSTRUCT_MAP.get(int(cl))
                    if cls is None:
                        continue
                    x1, y1, x2, y2 = b
                    cx = ((x1 + x2) / 2) / W
                    cy = ((y1 + y2) / 2) / H
                    w = (x2 - x1) / W
                    h = (y2 - y1) / H
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {cf:.4f}")
                    kept[cls] = kept.get(cls, 0) + 1
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
            done += 1
        print(f"  {min(i + args.batch, len(todo))}/{len(todo)} ...", flush=True)

    print(f"done: {done} images -> {lbl_dir}", flush=True)
    print("kept per our-class:", dict(sorted(kept.items())), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
