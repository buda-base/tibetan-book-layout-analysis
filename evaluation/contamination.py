#!/usr/bin/env python3
"""Failure-mode analysis for the text-area use case.

We crop the predicted text-area and send it to OCR, so the damaging failure is
not "missed a header/footnote" but "folded the header/footnote INTO the text
area" — that silently contaminates the OCR text with running heads, folio
numbers, or footnotes.

For each model we ask, per canonical region type (header-footer, footnote):
    * detected      : a predicted box of that type matches the GT box (IoU>=0.5)
    * recoverable   : detected AND >=50% of the GT area is inside the predicted
                    text-area envelope. Dual label; punch-out can fix it.
    * hidden        : NOT detected, but >=50% of the GT area is inside the
                    predicted text-area envelope (the OCR crop). No same-class
                    box to subtract. This is the paper's contamination_rate.
    * clean-miss    : NOT detected and NOT inside the text envelope -> dropped;
                    the body text stays clean. Tolerable.

The predicted text-area envelope is the min/max box over all predicted text-area
boxes on the page (same merge the canonical evaluator uses). A separate
as-textarea count uses IoU>=0.5 against native TA boxes (not the envelope).

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
    blank = dict(gt=0, det=0, absorbed=0, clean=0,
                 recoverable=0, as_ta=0, dual_as_ta=0, hidden_as_ta=0)
    stats = {HF: dict(blank), FN: dict(blank)}
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
                same = best >= 0.5 and bj >= 0
                if same:
                    used[bj] = True
                    stats[cls]["det"] += 1
                covered = covered_frac(gb, env) >= cover
                as_ta = any(iou(gb, tb) >= 0.5 for tb in p[TA])
                if covered:
                    if same:
                        stats[cls]["recoverable"] += 1
                    else:
                        stats[cls]["absorbed"] += 1
                elif not same:
                    stats[cls]["clean"] += 1
                if as_ta:
                    stats[cls]["as_ta"] += 1
                    if same:
                        stats[cls]["dual_as_ta"] += 1
                    else:
                        stats[cls]["hidden_as_ta"] += 1
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
          f"of region inside predicted text-area envelope)")
    print("=" * 98)
    print(f"{'region':14} {'GT':>5} {'detected':>9} {'recov.':>8} "
          f"{'hidden':>8} {'clean':>7} {'as-TA IoU':>10} {'hidden-IoU':>11}")
    for cls in (HF, FN):
        s = st[cls]
        n = s["gt"] or 1
        print(f"{name[cls]:14} {s['gt']:5d} {s['det']:6d}({100*s['det']/n:3.0f}%) "
              f"{s['recoverable']:4d}({100*s['recoverable']/n:3.0f}%) "
              f"{s['absorbed']:4d}({100*s['absorbed']/n:3.0f}%) "
              f"{s['clean']:7d} "
              f"{s['as_ta']:6d}({100*s['as_ta']/n:3.0f}%) "
              f"{s['hidden_as_ta']:6d}({100*s['hidden_as_ta']/n:3.0f}%)")
    print("-" * 98)
    print("recov. = detected AND inside TA envelope (punch-out can fix); "
          "hidden = missed AND inside TA envelope (paper contamination); "
          "as-TA IoU = IoU>=0.5 with a native text-area box")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
