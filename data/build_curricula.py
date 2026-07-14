#!/usr/bin/env python3
"""Derive a curriculum dataset from a base 4-class HFF dataset, keeping the exact
same split membership (images are hardlinked, only labels are transformed).

modes:
  tam   4 classes, but all text-area (class 1) boxes on a page are merged into a
        single envelope box (min/max corners). header/footnote/footer untouched.
        names: {0 header, 1 text-area, 2 footnote, 3 footer}
  3cls  header (0) and footer (3) merged into one class; text-area/footnote kept.
        remap {0->0, 1->1, 2->2, 3->0}; names {0 header-footer, 1 text-area, 2 footnote}
  3cls_tam  both of the above: header+footer merged into one class AND all
        text-area boxes merged into one envelope per page. names = 3cls names.
  tam2col  like tam (4 classes, text-area merged), EXCEPT pages detected as a
        two-column layout keep TWO text-area boxes (one per column). A page is
        two-column when its text-area boxes split into a left/right group that
        are horizontally disjoint and vertically co-extensive (share >= V_THRESH
        of the smaller column's height). names = 4-class names.

Usage: python build_curricula.py <base_dir> <mode:tam|3cls|3cls_tam|tam2col> <out_dir>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

NAMES4 = {0: "header", 1: "text-area", 2: "footnote", 3: "footer"}
NAMES3 = {0: "header-footer", 1: "text-area", 2: "footnote"}
REMAP3 = {0: 0, 1: 1, 2: 2, 3: 0}
SPLITS = ["train", "val", "test"]

# two-column detection thresholds
V_THRESH = 0.30      # columns must share >= 30% of the smaller column's height
X_OVERLAP_MAX = 0.20  # columns must overlap < 20% of the smaller column's width
STATS = {"pages_ta": 0, "pages_2col": 0}


def read_boxes(path: Path):
    boxes = []
    if not path.exists():
        return boxes
    for ln in path.read_text().splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        boxes.append((int(p[0]), *(float(v) for v in p[1:5])))
    return boxes


def envelope(group):
    """group: list of (cx,cy,w,h) -> single (cx,cy,w,h) covering all."""
    x1 = min(cx - w / 2 for cx, cy, w, h in group)
    y1 = min(cy - h / 2 for cx, cy, w, h in group)
    x2 = max(cx + w / 2 for cx, cy, w, h in group)
    y2 = max(cy + h / 2 for cx, cy, w, h in group)
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def merge_text_area(boxes):
    """Merge all class-1 (text-area) boxes into one envelope; keep the rest."""
    ta = [(cx, cy, w, h) for c, cx, cy, w, h in boxes if c == 1]
    out = [(c, cx, cy, w, h) for c, cx, cy, w, h in boxes if c != 1]
    if ta:
        cx, cy, w, h = envelope(ta)
        out.append((1, cx, cy, w, h))
    return out


def _corners(g):
    return (min(cx - w / 2 for cx, cy, w, h in g), min(cy - h / 2 for cx, cy, w, h in g),
            max(cx + w / 2 for cx, cy, w, h in g), max(cy + h / 2 for cx, cy, w, h in g))


def _to_cxcywh(c):
    return ((c[0] + c[2]) / 2, (c[1] + c[3]) / 2, c[2] - c[0], c[3] - c[1])


def two_column_split(ta):
    """Given text-area boxes (cx,cy,w,h), return [one envelope] normally, or
    [left, right] when the page is a genuine two-column layout."""
    if len(ta) <= 1:
        return [envelope(ta)] if ta else []
    # split at the largest gap between consecutive x-centers
    order = sorted(ta, key=lambda b: b[0])
    xc = [b[0] for b in order]
    split = max(range(len(xc) - 1), key=lambda k: xc[k + 1] - xc[k])
    left, right = order[:split + 1], order[split + 1:]
    if not left or not right:
        return [envelope(ta)]
    L, R = _corners(left), _corners(right)
    hov = max(0.0, min(L[2], R[2]) - max(L[0], R[0]))       # x overlap
    vov = max(0.0, min(L[3], R[3]) - max(L[1], R[1]))       # y overlap
    minw = min(L[2] - L[0], R[2] - R[0])
    minh = min(L[3] - L[1], R[3] - R[1])
    if minw > 0 and minh > 0 and hov / minw < X_OVERLAP_MAX and vov / minh >= V_THRESH:
        return [_to_cxcywh(L), _to_cxcywh(R)]
    return [envelope(ta)]


def merge_text_area_2col(boxes):
    """Like merge_text_area, but keep two boxes for two-column pages."""
    ta = [(cx, cy, w, h) for c, cx, cy, w, h in boxes if c == 1]
    out = [(c, cx, cy, w, h) for c, cx, cy, w, h in boxes if c != 1]
    if ta:
        STATS["pages_ta"] += 1
        merged = two_column_split(ta)
        if len(merged) == 2:
            STATS["pages_2col"] += 1
        for cx, cy, w, h in merged:
            out.append((1, cx, cy, w, h))
    return out


def transform(boxes, mode):
    if mode == "3cls":
        return [(REMAP3[c], cx, cy, w, h) for c, cx, cy, w, h in boxes]
    if mode == "tam":
        return merge_text_area(boxes)
    if mode == "3cls_tam":
        remapped = [(REMAP3[c], cx, cy, w, h) for c, cx, cy, w, h in boxes]
        return merge_text_area(remapped)
    if mode == "tam2col":
        return merge_text_area_2col(boxes)
    raise ValueError(mode)


def main() -> int:
    base = Path(sys.argv[1]).resolve()
    mode = sys.argv[2]
    out = Path(sys.argv[3]).resolve()
    assert mode in ("tam", "3cls", "3cls_tam", "tam2col"), mode
    names = NAMES3 if mode in ("3cls", "3cls_tam") else NAMES4

    n_img = n_box_in = n_box_out = 0
    for s in SPLITS:
        img_src = base / "images" / s
        lbl_src = base / "labels" / s
        img_dst = out / "images" / s
        lbl_dst = out / "labels" / s
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        if not img_src.is_dir():
            continue
        for ip in img_src.iterdir():
            if not ip.is_file():
                continue
            n_img += 1
            dst = img_dst / ip.name
            if not dst.exists():
                os.link(ip, dst)
            boxes = read_boxes(lbl_src / f"{ip.stem}.txt")
            n_box_in += len(boxes)
            tb = transform(boxes, mode)
            n_box_out += len(tb)
            lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in tb]
            (lbl_dst / f"{ip.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))

    yml = [f"path: {out}", "train: images/train", "val: images/val",
           "test: images/test", "names:"]
    yml += [f"  {i}: {names[i]}" for i in sorted(names)]
    (out / "data.yaml").write_text("\n".join(yml) + "\n")

    print(f"built {out} (mode={mode})")
    print(f"  images: {n_img}")
    print(f"  boxes:  {n_box_in} -> {n_box_out} (delta {n_box_out - n_box_in:+d})")
    print(f"  names:  {list(names.values())}")
    if mode == "tam2col":
        print(f"  two-column pages: {STATS['pages_2col']} / {STATS['pages_ta']} "
              f"pages with text-area (V_THRESH={V_THRESH}, X_OVERLAP_MAX={X_OVERLAP_MAX})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
