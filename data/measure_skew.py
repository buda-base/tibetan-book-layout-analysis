#!/usr/bin/env python3
"""Estimate page skew across the dataset to decide AABB-vs-deskew empirically.

For each image we binarise the page and find the rotation angle (in a small
range) that maximises the variance of the horizontal projection profile — i.e.
the angle at which text rows are most sharply separated. That angle is the
estimated skew.

Reports median / p90 / p99 |skew| and the fraction of pages above 2deg and
5deg, and (if matplotlib is available) writes a histogram PNG plus a per-image
CSV. If the median is small (< ~2deg) axis-aligned boxes are fine; a fat tail
past ~5deg argues for a deskew preprocessing step.

Usage:
    python measure_skew.py --data dataset/data.yaml --sample 500
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List

import cv2
import numpy as np
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def gather_images(data_yaml: Path, splits: List[str]) -> List[Path]:
    doc = yaml.safe_load(data_yaml.read_text())
    root = Path(doc.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    imgs: List[Path] = []
    for split in splits:
        sub = doc.get(split)
        if not sub:
            continue
        d = root / sub
        if d.is_dir():
            imgs += [p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS]
    return sorted(imgs)


def estimate_skew(path: Path, work_width: int = 900) -> float | None:
    """Return estimated skew angle in degrees, or None if unreadable."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    if w > work_width:
        scale = work_width / w
        img = cv2.resize(img, (work_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    # Binarise: ink -> 1. Otsu, inverted (dark text on light page).
    _, binimg = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binimg = binimg.astype(np.float32)

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D((binimg.shape[1] / 2, binimg.shape[0] / 2), angle, 1.0)
        rot = cv2.warpAffine(
            binimg, m, (binimg.shape[1], binimg.shape[0]), flags=cv2.INTER_NEAREST
        )
        proj = rot.sum(axis=1)
        return float(np.var(proj))

    # Coarse search then refine.
    coarse = np.arange(-8.0, 8.01, 1.0)
    best = max(coarse, key=score)
    fine = np.arange(best - 1.0, best + 1.01, 0.2)
    best = max(fine, key=score)
    return round(float(best), 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="dataset data.yaml")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "test"], help="splits to sample"
    )
    parser.add_argument(
        "--sample", type=int, default=500, help="max images to sample (0 = all)"
    )
    parser.add_argument("--csv", type=Path, default=Path("skew_angles.csv"))
    parser.add_argument("--hist", type=Path, default=Path("skew_hist.png"))
    args = parser.parse_args()

    images = gather_images(args.data, args.splits)
    if not images:
        raise SystemExit(f"No images found for splits {args.splits} in {args.data}")
    random.seed(0)
    if args.sample and len(images) > args.sample:
        images = random.sample(images, args.sample)

    angles: List[float] = []
    rows = []
    for i, p in enumerate(images, 1):
        ang = estimate_skew(p)
        if ang is None:
            continue
        angles.append(ang)
        rows.append((str(p), ang))
        if i % 100 == 0:
            print(f"  {i}/{len(images)} processed...", flush=True)

    arr = np.abs(np.array(angles))
    with args.csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "skew_deg"])
        writer.writerows(rows)

    print("\n=== skew summary (|angle|, degrees) ===")
    print(f"  images measured : {len(angles)}")
    print(f"  median          : {np.median(arr):.2f}")
    print(f"  mean            : {arr.mean():.2f}")
    print(f"  p90             : {np.percentile(arr, 90):.2f}")
    print(f"  p99             : {np.percentile(arr, 99):.2f}")
    print(f"  max             : {arr.max():.2f}")
    print(f"  fraction > 2deg : {(arr > 2).mean():.1%}")
    print(f"  fraction > 5deg : {(arr > 5).mean():.1%}")
    print(f"  per-image CSV   : {args.csv}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.hist(angles, bins=np.arange(-8, 8.5, 0.5), edgecolor="black")
        plt.xlabel("estimated skew (degrees)")
        plt.ylabel("pages")
        plt.title(f"Page skew distribution (n={len(angles)})")
        plt.tight_layout()
        plt.savefig(args.hist, dpi=120)
        print(f"  histogram PNG   : {args.hist}")
    except Exception as exc:  # pragma: no cover
        print(f"  (histogram skipped: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
