#!/usr/bin/env python3
"""Failure-mode analysis for the text-area use case.

We crop the predicted text-area and send it to OCR, so the damaging failure is
not "missed a header/footnote" but "folded the header/footnote INTO the text
area" — that silently contaminates the OCR text with running heads, folio
numbers, or footnotes.

For each model we ask, per canonical region type (header-footer, footnote):
  * detected      : a predicted box of that type matches the GT box (IoU>=0.5)
  * absorbed      : NOT detected, but >=50% of the GT box area lies inside the
                    predicted text-area envelope (the merged text block) -> the
                    region is inside what we would OCR as body text. BAD.
  * clean-miss    : NOT detected and NOT inside the text envelope -> the region
                    is simply dropped; the body text stays clean. Tolerable.

The predicted text-area envelope is the min/max box over all predicted text-area
boxes on the page (same merge the canonical evaluator uses).

Usage:
  python contamination.py <pred_dir> <gt_dir> [remap] [conf] [cover]
    remap default "0:0,1:1,2:2,3:0"  conf default 0.0  cover default 0.5
    (canonical ids: 0 header-footer, 1 text-area, 2 footnote)
"""
from __future__ import annotations

import sys
from pathlib import Path

HF, TA, FN = 0, 1, 2  # canonical ids after remap


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def covered_frac(box, env):
    """fraction of `box` area inside `env`."""
    if env is None:
        return 0.0
    ix1, iy1 = max(box[0], env[0]), max(box[1], env[1])
    ix2, iy2 = min(box[2], env[2]), min(box[3], env[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area = (box[2] - box[0]) * (box[3] - box[1])
    return inter / area if area > 0 else 0.0


def read(path, remap, conf_floor):
    out = {HF: [], TA: [], FN: []}
    if not path.exists():
        return out
    for ln in path.read_text().splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        cc = remap.get(int(p[0]))
        if cc is None:
            continue
        if len(p) >= 6 and float(p[5]) < conf_floor:
            continue
        cx, cy, w, h = (float(x) for x in p[1:5])
        out[cc].append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return out


def analyse(pred_dir, gt_dir, remap, conf, cover):
    stats = {HF: dict(gt=0, det=0, absorbed=0, clean=0),
             FN: dict(gt=0, det=0, absorbed=0, clean=0)}
    for gp in sorted(gt_dir.glob("*.txt")):
        g = read(gp, remap, 0.0)
        p = read(pred_dir / f"{gp.stem}.txt", remap, conf)
        env = None
        if p[TA]:
            env = [min(b[0] for b in p[TA]), min(b[1] for b in p[TA]),
                   max(b[2] for b in p[TA]), max(b[3] for b in p[TA])]
        for cls in (HF, FN):
            used = [False] * len(p[cls])
            for gb in g[cls]:
                stats[cls]["gt"] += 1
                best, bj = 0.0, -1
                for j, pb in enumerate(p[cls]):
                    if used[j]:
                        continue
                    v = iou(gb, pb)
                    if v > best:
                        best, bj = v, j
                if best >= 0.5 and bj >= 0:
                    used[bj] = True
                    stats[cls]["det"] += 1
                elif covered_frac(gb, env) >= cover:
                    stats[cls]["absorbed"] += 1
                else:
                    stats[cls]["clean"] += 1
    return stats


def main() -> int:
    pred_dir = Path(sys.argv[1])
    gt_dir = Path(sys.argv[2])
    remap = {int(k): int(v) for k, v in
             (x.split(":") for x in (sys.argv[3] if len(sys.argv) > 3
                                     else "0:0,1:1,2:2,3:0").split(","))}
    conf = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    cover = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
    st = analyse(pred_dir, gt_dir, remap, conf, cover)
    name = {HF: "header-footer", FN: "footnote"}
    print(f"failure analysis: {pred_dir}  (conf>={conf}, absorb if >={cover:.0%} "
          f"of missed region inside predicted text-area envelope)")
    print("=" * 78)
    print(f"{'region':14} {'GT':>5} {'detected':>9} {'MISSED':>7} "
          f"{'absorbed→TA':>12} {'clean-miss':>11}")
    for cls in (HF, FN):
        s = st[cls]
        miss = s["absorbed"] + s["clean"]
        det_p = s["det"] / s["gt"] * 100 if s["gt"] else 0
        ab_p = s["absorbed"] / miss * 100 if miss else 0
        print(f"{name[cls]:14} {s['gt']:5d} {s['det']:6d}({det_p:3.0f}%) "
              f"{miss:7d} {s['absorbed']:7d}({ab_p:3.0f}%) {s['clean']:11d}")
    print("-" * 78)
    print("absorbed→TA = missed region folded INTO the OCR text block (bad); "
          "clean-miss = dropped, text stays clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
