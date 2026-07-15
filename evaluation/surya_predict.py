#!/usr/bin/env python3
"""Run Surya's FAST layout model (off-the-shelf, no fine-tuning) on a folder of
images and write YOLO-format predictions in OUR 4-class schema:

    0 header    1 text-area    2 footnote    3 footer

We use surya.fast_layout.FastLayoutPredictor -- the lightweight pure-torch RF-DETR
detector (checkpoint datalab-to/surya_layout2), NOT the heavyweight vLLM-served VLM
(surya-ocr-2). It emits the same DocLayNet-style labels; we map them to our schema
and drop everything that is not header/footer/footnote/text. Text-area is left as
the individual blocks Surya returns -- the canonical evaluator merges them into a
single envelope per page, so this stays apples-to-apples with our own models.

Output: one <stem>.txt per image with rows "cls cx cy w h conf" (normalized).
Resumable: images whose label file already exists are skipped.

Usage:
  python surya_predict.py --source <img_dir> --out <out_dir> [--batch 8]
                          [--limit N] [--threshold 0.05]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

# Surya label -> our class id (None = drop)
LABEL_MAP = {
    "PageHeader": 0,
    "PageFooter": 3,
    "Footnote": 2,
    "Text": 1,
    "SectionHeader": 1,
    "Caption": 1,
    "ListGroup": 1,
    "ListItem": 1,
    "Bibliography": 1,
    "Code": 1,
    "TableOfContents": 1,
    "Form": 1,
    "TextInlineMath": 1,
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def poly_to_bbox(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="low floor; confidence is written per box for later sweep")
    args = ap.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    lbl_dir = out / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    if args.limit:
        imgs = imgs[: args.limit]
    todo = [p for p in imgs if not (lbl_dir / f"{p.stem}.txt").exists()]
    print(f"{len(imgs)} images ({len(todo)} to do, {len(imgs) - len(todo)} cached) "
          f"-> Surya FAST layout (rf-detr), batch {args.batch}, thr {args.threshold}",
          flush=True)

    from surya.fast_layout import FastLayoutPredictor
    predictor = FastLayoutPredictor()

    done = 0
    unknown = {}
    for i in range(0, len(todo), args.batch):
        chunk = todo[i : i + args.batch]
        pil = []
        for p in chunk:
            im = Image.open(p).convert("RGB")
            pil.append(im)
        results = predictor(pil, threshold=args.threshold,
                             batch_size=args.batch, use_order=False)
        for p, im, res in zip(chunk, pil, results):
            W, H = im.size
            lines = []
            for b in (res.bboxes or []):
                cls = LABEL_MAP.get(b.label)
                if cls is None:
                    unknown[b.label] = unknown.get(b.label, 0) + 1
                    continue
                x1, y1, x2, y2 = poly_to_bbox(b.polygon)
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H
                w = (x2 - x1) / W
                h = (y2 - y1) / H
                conf = float(b.confidence) if b.confidence is not None else 1.0
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.4f}")
            (lbl_dir / f"{p.stem}.txt").write_text("\n".join(lines))
            done += 1
        print(f"  {min(i + args.batch, len(todo))}/{len(todo)} ...", flush=True)

    print(f"done: {done} images -> {lbl_dir}", flush=True)
    if unknown:
        print("dropped (not in our schema):", dict(sorted(
            unknown.items(), key=lambda kv: -kv[1])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
