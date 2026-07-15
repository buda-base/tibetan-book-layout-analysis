#!/usr/bin/env python3
"""Convert a YOLO-format detection dataset to the COCO layout RF-DETR expects.

RF-DETR wants:  <out>/{train,valid,test}/_annotations.coco.json  with the images
alongside each json. Following the Roboflow convention, category id 0 is a dummy
("background") and the real classes start at id 1, so YOLO class c -> category
id c+1.

Images are symlinked (not copied) to save disk. The YOLO "val" split is written
as "valid" (RF-DETR's expected name).

Usage:
  python yolo2coco.py <yolo_dataset_dir> <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

SPLIT_MAP = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def load_names(ddir: Path):
    names, in_names = {}, False
    for ln in (ddir / "data.yaml").read_text().splitlines():
        if ln.strip().startswith("names:"):
            in_names = True
            continue
        if in_names:
            s = ln.strip()
            if not s or not s[0].isdigit():
                break
            k, v = s.split(":", 1)
            names[int(k)] = v.strip()
    return names


def convert_split(ddir: Path, split: str, out: Path, names: dict) -> tuple[int, int]:
    img_dir = ddir / "images" / split
    lbl_dir = ddir / "labels" / split
    if not img_dir.is_dir():
        return 0, 0
    out_split = out / SPLIT_MAP[split]
    out_split.mkdir(parents=True, exist_ok=True)

    categories = [{"id": 0, "name": "none", "supercategory": "none"}]
    categories += [{"id": c + 1, "name": names[c], "supercategory": "none"}
                   for c in sorted(names)]
    images, annotations = [], []
    ann_id = 1
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    for img_id, ip in enumerate(imgs, 1):
        with Image.open(ip) as im:
            W, H = im.size
        link = out_split / ip.name
        if not link.exists():
            link.symlink_to(ip.resolve())
        images.append({"id": img_id, "file_name": ip.name, "width": W, "height": H})
        lp = lbl_dir / f"{ip.stem}.txt"
        if lp.exists():
            for ln in lp.read_text().splitlines():
                p = ln.split()
                if len(p) < 5:
                    continue
                c = int(p[0])
                cx, cy, w, h = (float(x) for x in p[1:5])
                bw, bh = w * W, h * H
                bx, by = (cx * W) - bw / 2, (cy * H) - bh / 2
                annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": c + 1,
                    "bbox": [bx, by, bw, bh], "area": bw * bh, "iscrowd": 0,
                    "segmentation": [],
                })
                ann_id += 1
    coco = {"images": images, "annotations": annotations, "categories": categories}
    (out_split / "_annotations.coco.json").write_text(json.dumps(coco))
    return len(images), len(annotations)


def main() -> int:
    ddir = Path(sys.argv[1])
    out = Path(sys.argv[2])
    names = load_names(ddir)
    print(f"classes: {names}")
    for split in ("train", "val", "test"):
        ni, na = convert_split(ddir, split, out, names)
        print(f"  {split:5} -> {SPLIT_MAP.get(split, split):5}: {ni} images, {na} boxes")
    print(f"wrote COCO dataset to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
