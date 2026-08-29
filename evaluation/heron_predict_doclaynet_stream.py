#!/usr/bin/env python3
"""Stream DocLayNet v1.2 test, run Docling layout-heron (RT-DETRv2, off-the-shelf),
write heron predictions in OUR 4-class YOLO schema plus DocLayNet GT + index.

This fills the 2x2 bottom-right cell with a *shared-3-class* number for heron
on DocLayNet test (page-header / page-footer / footnote), instead of the
published 11-class 0.699.

  <out>/pred_labels/<stem>.txt   heron -> our 4-class (0 header,1 body,2 fn,3 footer)
  <out>/gt_labels/<stem>.txt     DocLayNet YOLO (category_id-1), from same stream
  <out>/index.jsonl              stem, width, height, doc_category

Then score with score_ours_on_doclaynet.py (shared mean excludes text-area).

HF token: reads HF_TOKEN / HUGGING_FACE_HUB_TOKEN from env for Hub *download*
only (load_dataset + from_pretrained). Never printed, never uploaded.

Usage:
  python heron_predict_doclaynet_stream.py --out /path/heron_doclaynet \\
      [--model docling-project/docling-layout-heron] [--conf 0.05] \\
      [--device 0] [--max-pages N]
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from PIL import Image

# Heron label -> our class id (None = drop). Body-ish classes -> text-area (1).
LABEL_MAP = {
    "page_header": 0, "page_footer": 3, "footnote": 2,
    "text": 1, "title": 1, "section_header": 1, "list_item": 1,
    "caption": 1, "table": 1, "formula": 1, "code": 1,
    "document_index": 1, "form": 1, "key_value_region": 1,
}


def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    return ((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H, (x2 - x1) / W, (y2 - y1) / H


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="docling-project/docling-layout-heron")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-pages", type=int, default=0)
    ap.add_argument("--dataset", default="docling-project/DocLayNet-v1.2")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

    pred_dir = args.out / "pred_labels"
    gt_dir = args.out / "gt_labels"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.jsonl"

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print("Hugging Face auth: read token from env (download only)"
          if token else "WARNING: no HF_TOKEN in env", flush=True)

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(f"loading heron {args.model} on {device}", flush=True)
    processor = RTDetrImageProcessor.from_pretrained(args.model, token=token)
    model = RTDetrV2ForObjectDetection.from_pretrained(args.model, token=token).to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    print(f"streaming {args.dataset} split={args.split}", flush=True)
    ds = load_dataset(args.dataset, split=args.split, streaming=True, token=token)
    keep = [c for c in ("image", "bboxes", "category_id", "metadata")
            if c in (getattr(ds, "column_names", None) or
                     ["image", "bboxes", "category_id", "metadata"])]
    try:
        ds = ds.select_columns(keep)
    except Exception as exc:
        print(f"(select_columns skipped: {exc})", flush=True)

    n = 0
    dropped = {}
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

            glines = []
            for bb, cid in zip(row["bboxes"], row["category_id"]):
                if len(bb) != 4:
                    continue
                x, y, bw, bh = (float(v) for v in bb)
                cx, cy, w, h = xyxy_to_yolo(x, y, x + bw, y + bh, W, H)
                glines.append(f"{int(cid) - 1} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            (gt_dir / f"{stem}.txt").write_text("\n".join(glines))

            inputs = processor(images=[im], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            tgt = torch.tensor([[H, W]], device=device)
            res = processor.post_process_object_detection(
                outputs, target_sizes=tgt, threshold=args.conf)[0]
            plines = []
            for score, label_id, box in zip(res["scores"].tolist(),
                                            res["labels"].tolist(),
                                            res["boxes"].tolist()):
                label = id2label.get(int(label_id), str(label_id))
                cls = LABEL_MAP.get(label)
                if cls is None:
                    dropped[label] = dropped.get(label, 0) + 1
                    continue
                x1, y1, x2, y2 = box
                cx, cy, w, h = xyxy_to_yolo(x1, y1, x2, y2, W, H)
                plines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {score:.4f}")
            (pred_dir / f"{stem}.txt").write_text("\n".join(plines))
            idxf.write(json.dumps({"stem": stem, "width": W, "height": H,
                                   "doc_category": meta.get("doc_category")}) + "\n")
            idxf.flush()
            if n % 50 == 0:
                print(f"  {n} pages...", flush=True)
    print(f"done {n} pages -> {args.out}", flush=True)
    if dropped:
        print("dropped labels:", dict(sorted(dropped.items(), key=lambda kv: -kv[1])),
              flush=True)
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
