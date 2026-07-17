#!/usr/bin/env python3
"""Run Docling layout-heron (RT-DETRv2, off-the-shelf) on a folder of images and
write YOLO-format predictions in OUR 4-class schema:

    0 header    1 text-area    2 footnote    3 footer

Heron id2label (docling-project/docling-layout-heron) -> our schema:
    page_header -> 0, page_footer -> 3, footnote -> 2, text/title/section_header/
    list_item/caption/... -> 1 (body); everything else dropped.

Output: one <stem>.txt per image with rows "cls cx cy w h conf" (normalized).
Resumable: images whose label file already exists are skipped.

Usage:
  python docling_heron_predict.py --source <img_dir> --out <out_dir>
                                  [--model docling-project/docling-layout-heron]
                                  [--conf 0.2] [--batch 4] [--limit N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

# Heron pretrained label -> our class id (None = drop)
LABEL_MAP = {
    "page_header": 0,
    "page_footer": 3,
    "footnote": 2,
    "text": 1,
    "title": 1,
    "section_header": 1,
    "list_item": 1,
    "caption": 1,
    "table": 1,
    "formula": 1,
    "code": 1,
    "document_index": 1,
    "form": 1,
    "key_value_region": 1,
}
# Fine-tuned tam2col checkpoint uses our schema directly
FINE_TUNED_LABEL_MAP = {
    "header": 0,
    "text-area": 1,
    "footnote": 2,
    "footer": 3,
}


def map_label(label: str) -> int | None:
    if label in FINE_TUNED_LABEL_MAP:
        return FINE_TUNED_LABEL_MAP[label]
    return LABEL_MAP.get(label)
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="docling-project/docling-layout-heron")
    ap.add_argument("--checkpoint", default="",
                    help="Fine-tuned checkpoint dir (default: use --model HF hub id)")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.source)
    lbl_dir = Path(args.out) / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    if args.limit:
        imgs = imgs[: args.limit]
    todo = [p for p in imgs if not (lbl_dir / f"{p.stem}.txt").exists()]
    ckpt = args.checkpoint or args.model
    print(f"{len(imgs)} images ({len(todo)} to do, {len(imgs) - len(todo)} cached) "
          f"-> Docling heron ({ckpt}), conf {args.conf}, batch {args.batch}",
          flush=True)

    import torch
    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = RTDetrImageProcessor.from_pretrained(ckpt)
    model = RTDetrV2ForObjectDetection.from_pretrained(ckpt).to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    dropped = {}
    done = 0
    for i in range(0, len(todo), args.batch):
        chunk = todo[i : i + args.batch]
        pil = [Image.open(p).convert("RGB") for p in chunk]
        inputs = processor(images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[im.size[1], im.size[0]] for im in pil], device=device)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=args.conf)
        for p, im, res in zip(chunk, pil, results):
            W, H = im.size
            lines = []
            for score, label_id, box in zip(
                    res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()):
                label = id2label.get(int(label_id), str(label_id))
                cls = map_label(label)
                if cls is None:
                    dropped[label] = dropped.get(label, 0) + 1
                    continue
                x1, y1, x2, y2 = box
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H
                w = (x2 - x1) / W
                h = (y2 - y1) / H
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {score:.4f}")
            (lbl_dir / f"{p.stem}.txt").write_text("\n".join(lines))
            done += 1
        print(f"  {min(i + args.batch, len(todo))}/{len(todo)} ...", flush=True)

    print(f"done: {done} images -> {lbl_dir}", flush=True)
    if dropped:
        print("dropped labels:", dict(sorted(dropped.items(), key=lambda kv: -kv[1])),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
