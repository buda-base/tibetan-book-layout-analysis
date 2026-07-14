#!/usr/bin/env python3
"""Assemble the HuggingFace release folder for the v3 HFF dataset.

Hardlinks images (no extra disk), copies labels, writes data.yaml, the
train/val/test path lists, and a full README.md dataset card (with a fair-use /
no-warranty notice instead of an open license, plus a gating form).
"""
from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

SPLITS = ["train", "val", "test"]
NAMES = {0: "header", 1: "text-area", 2: "footnote", 3: "footer"}


def clamp_box(cx, cy, w, h):
    """Clip a YOLO box to [0,1] via its corners, recompute center/size, and
    round to 6 decimals so the reconstructed corners stay within [0,1] (no
    out-of-range coords / rounding overflow on import). Returns None if the box
    collapses to zero area after clipping."""
    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
    if x2 <= x1 or y2 <= y1:
        return None
    cx, cy, w, h = round((x1 + x2) / 2, 6), round((y1 + y2) / 2, 6), \
        round(x2 - x1, 6), round(y2 - y1, 6)
    # guard against rounding pushing a corner just outside [0,1]
    if cx - w / 2 < 0:
        w = round(2 * cx, 6)
    if cx + w / 2 > 1:
        w = round(2 * (1 - cx), 6)
    if cy - h / 2 < 0:
        h = round(2 * cy, 6)
    if cy + h / 2 > 1:
        h = round(2 * (1 - cy), 6)
    if w <= 0 or h <= 0:
        return None
    return cx, cy, w, h


def link_or_copy(src: Path, dst: Path) -> None:
    real = Path(os.path.realpath(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(real, dst)
    except OSError:
        shutil.copy2(real, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset_v3")
    ap.add_argument("--out", default="release_v3/TDLA-Training-Dataset-v2")
    ap.add_argument("--pretty", default="TDLA Training Dataset v2")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    imgcount, aug_per = {}, {}
    boxes = {s: Counter() for s in SPLITS}
    listing = {s: [] for s in SPLITS}
    for s in SPLITS:
        simg = src / "images" / s
        names = sorted(p.name for p in simg.iterdir())
        imgcount[s] = len(names)
        aug_per[s] = sum("__aug" in n for n in names)
        for n in names:
            link_or_copy(simg / n, out / "images" / s / n)
            lp = src / "labels" / s / f"{Path(n).stem}.txt"
            dst = out / "labels" / s / f"{Path(n).stem}.txt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            txt = lp.read_text() if lp.exists() else ""
            out_lines = []
            for ln in txt.splitlines():
                p = ln.split()
                if len(p) < 5:
                    continue
                c = int(p[0])
                r = clamp_box(*(float(v) for v in p[1:5]))
                if r is None:
                    continue
                cx, cy, w, h = r
                out_lines.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                boxes[s][c] += 1
            dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
            listing[s].append(f"./images/{s}/{n}")
        (out / f"{s}.txt").write_text("\n".join(listing[s]) + "\n")

    (out / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n" + "".join(f"  {i}: {NAMES[i]}\n" for i in sorted(NAMES))
    )

    tot = Counter()
    for s in SPLITS:
        tot.update(boxes[s])
    total_ann = sum(tot.values())
    total_img = sum(imgcount.values())

    def pct(c):
        return f"{100*c/total_ann:.1f}%"

    cls_rows = "\n".join(
        f"| {i} | {NAMES[i]} | {tot[i]} | {pct(tot[i])} |" for i in sorted(NAMES)
    )
    size_rows = "\n".join(
        f"| {s} | {imgcount[s]} |" for s in SPLITS
    )
    dist_rows = "\n".join(
        f"| {NAMES[i]} | {boxes['train'][i]} | {boxes['val'][i]} | {boxes['test'][i]} | {tot[i]} |"
        for i in sorted(NAMES)
    )

    readme = f"""---
license: other
license_name: fair-use-no-warranty
license_link: LICENSE
task_categories:
- object-detection
language:
- bo
tags:
- yolo
- tibetan
- document-layout-analysis
- bounding-box
size_categories:
- 1K<n<10K
pretty_name: {args.pretty}
extra_gated_prompt: >-
  The page images in this dataset are scans of Tibetan texts from the BDRC
  digital library and are provided on a FAIR-USE basis for research. No
  copyright license is granted. By requesting access you acknowledge that you
  are solely responsible for performing your own copyright / rights analysis
  before any use, and that the Buddhist Digital Resource Center (BDRC) accepts
  no liability for any misuse of this material.
extra_gated_fields:
  Full name: text
  Affiliation: text
  Intended use: text
  I have read the copyright notice and will perform my own copyright analysis before use: checkbox
  I understand BDRC is not liable for any misuse of this material: checkbox
---

# {args.pretty}

YOLO-format object-detection dataset for **Tibetan Document Layout Analysis (TDLA)**.
It contains bounding-box annotations for four layout classes on scanned Tibetan
document pages, split into training, validation, and test sets.

This is an expanded, re-reviewed successor to
[BDRC/TDLA-Training-Dataset](https://huggingface.co/datasets/BDRC/TDLA-Training-Dataset),
built from several annotation batches that were consolidated to a **single,
consistent annotation convention** and split to be **leakage-free**.

## Overview

| Property | Value |
| --- | --- |
| **Total annotations** | {total_ann} |
| **Total images** | {total_img} |
| **Number of classes** | 4 |
| **Image format** | JPEG (.jpg) |
| **Label format** | YOLO (.txt) |
| **Splits** | train / val / test |
| **Split unit** | volume-level (leakage-free) |

## Image Source

All images are sourced from the [Buddhist Digital Resource Center (BDRC)](https://bdrc.io) digital library.

## Classes

| ID | Name | Annotations | % of total |
| -- | --- | --- | --- |
{cls_rows}

## Annotation Process

Annotations were created on the Ultralytics HUB platform in a two-stage workflow:

1. **Annotation** — annotators drew bounding boxes for each of the four layout
   classes (header, text-area, footnote, footer) on every page image.
2. **Quality control** — a reviewer inspected every image, verifying label
   correctness, box tightness, and class assignment. Earlier annotation batches
   were re-reviewed so that all sources follow the same convention (in
   particular, marginal header/footer elements are boxed per element,
   consistently across the whole dataset).
3. **Automated consistency audit** — a final geometric/logical audit flagged
   likely mistakes (near-duplicate or conflicting-class boxes, impossible
   header/footer/footnote orderings, out-of-bounds boxes). Flagged pages were
   manually corrected and re-imported, removing conflicting duplicate boxes.

## Split Methodology

The train / val / test split is created by grouping pages at the **volume
(book) level** and assigning each volume as a whole to a single split. This
guarantees there is **no leakage** across splits — no page (or an augmented
copy of it) and no volume appears in more than one split. The split has been
audited for pixel-identical duplicates, shared page identities, and shared
volumes across splits (all clean).

- **Footnote stratification** — the footnote class is rare, so
  footnote-bearing volumes were distributed across all three splits to keep the
  class represented everywhere.
- **Augmented data** — a subset of the training images are augmented
  (geometric/photometric) copies. These are confined to the **training set
  only**; **validation and test contain exclusively original, non-augmented
  scans**, making them a clean benchmark. Augmented images can be recognised by
  an `__aug` marker in their filename.
- Approximate ratio: ~{100*imgcount['train']//total_img}% train /
  ~{round(100*imgcount['val']/total_img)}% val /
  ~{round(100*imgcount['test']/total_img)}% test by image count.

## Split Statistics

| Split | Images |
| --- | --- |
{size_rows}

(train includes {aug_per['train']} augmented images; val and test include {aug_per['val']} and {aug_per['test']}.)

## Annotation Distribution per Split

| Class | train | val | test | Total |
| --- | --- | --- | --- | --- |
{dist_rows}

> A single image can contain multiple annotations of the same class, so
> annotation counts may exceed image counts.

## Directory Structure

```
TDLA-Training-Dataset-v2/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── train.txt
├── val.txt
├── test.txt
├── data.yaml
└── README.md
```

## Usage

Point your YOLO training config at `data.yaml`:

```bash
yolo detect train data=data.yaml
```

The `train.txt`, `val.txt`, and `test.txt` files list relative image paths for each split.

## Label Format

Each `.txt` label file uses standard YOLO format — one row per bounding box:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to `[0, 1]` relative to image dimensions.

## Copyright & Usage Notice

This dataset does **not** come with an open-content license. The page images
are scans of Tibetan texts from the BDRC digital library and are distributed on
a **fair-use** basis for research and non-commercial layout-analysis work.

- **No copyright license is granted** over the underlying page images.
- **You are solely responsible** for performing your own copyright / rights
  analysis for your jurisdiction and intended use **before** using this
  material.
- **BDRC accepts no liability** for any misuse of this material.

By accessing the gated dataset you accept these terms.

## Acknowledgements

Developed by the [Buddhist Digital Resource Center (BDRC)](https://bdrc.io) for
the BDRC Etext Corpus. Thanks to the annotators and reviewers who produced and
consolidated the layout annotations.
"""
    (out / "README.md").write_text(readme)
    (out / "LICENSE").write_text(
        "Fair-use / no-warranty notice.\n\n"
        "The page images in this dataset are scans of Tibetan texts from the BDRC\n"
        "digital library, distributed on a fair-use basis for research. No copyright\n"
        "license is granted over the underlying images. Users are solely responsible\n"
        "for performing their own copyright/rights analysis before use. BDRC accepts\n"
        "no liability for any misuse of this material.\n"
    )
    print(f"release -> {out}")
    print(f"images: {total_img} (train {imgcount['train']}, val {imgcount['val']}, test {imgcount['test']})")
    print(f"annotations: {total_ann}  per-class {dict(tot)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
