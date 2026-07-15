#!/usr/bin/env python3
"""Score a folder of precomputed YOLO-format predictions (e.g. from Azure DI)
against a dataset's test GT, in OUR canonical 3-class space:

    0 header-footer  header + footer combined, matched INDIVIDUALLY
    1 text-area      all boxes merged into one envelope per page (gt and pred)
    2 footnote       unchanged

Predictions may optionally carry a 6th confidence column; if present, a
`--conf` floor is applied. Services like Azure DI emit no confidence, so the
default scores every predicted box (a single, un-thresholdable operating point)
and reports precision / recall / F1 / meanIoU per canonical class at IoU>=0.5.

Usage:
  python eval_pred_files.py <pred_labels_dir> <dataset_dir> [remap] [iou] [conf]
    remap default "0:0,1:1,2:2,3:0"   iou default 0.5   conf default 0.0
"""
from __future__ import annotations

import sys
from pathlib import Path

CANON = {0: "header-footer", 1: "text-area", 2: "footnote"}
MERGE = {1}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def envelope(bs):
    return [min(b[0] for b in bs), min(b[1] for b in bs),
            max(b[2] for b in bs), max(b[3] for b in bs)]


def read_yolo(path, remap, conf_floor=0.0):
    """-> dict class -> list of (x1,y1,x2,y2) in normalized coords.

    If a 6th column (confidence) is present, boxes below ``conf_floor`` are
    dropped. Predictions without a confidence column (e.g. Azure DI) are always
    kept -- there is only a single, un-thresholdable operating point.
    """
    out = {c: [] for c in CANON}
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


def score(gt_by_img, pred_by_img, thr):
    npos = sum(len(v) for v in gt_by_img.values())
    npred = sum(len(v) for v in pred_by_img.values())
    tp = 0
    ious = []
    for img, preds in pred_by_img.items():
        gts = gt_by_img.get(img, [])
        used = [False] * len(gts)
        for pb in sorted(preds, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                if used[j]:
                    continue
                v = iou(pb, g)
                if v > best:
                    best, bj = v, j
            if best >= thr and bj >= 0:
                tp += 1
                used[bj] = True
                ious.append(best)
    fp = npred - tp
    fn = npos - tp
    P = tp / (tp + fp) if tp + fp else 1.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    miou = sum(ious) / len(ious) if ious else 0.0
    return P, R, F1, miou, tp, fp, fn


def main() -> int:
    pred_dir = Path(sys.argv[1])
    ddir = Path(sys.argv[2])
    remap = {int(k): int(v) for k, v in
             (p.split(":") for p in (sys.argv[3] if len(sys.argv) > 3
                                     else "0:0,1:1,2:2,3:0").split(","))}
    thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    conf = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    lbl_dir = ddir / "labels/test"

    gt = {c: {} for c in CANON}
    pred = {c: {} for c in CANON}
    n = 0
    for gp in sorted(lbl_dir.glob("*.txt")):
        stem = gp.stem
        n += 1
        g = read_yolo(gp, remap)
        p = read_yolo(pred_dir / f"{stem}.txt", remap, conf_floor=conf)
        for c in CANON:
            gb = [envelope(g[c])] if (c in MERGE and g[c]) else g[c]
            pb = [envelope(p[c])] if (c in MERGE and p[c]) else p[c]
            if gb:
                gt[c][stem] = gb
            if pb:
                pred[c][stem] = pb

    print(f"canonical eval of predictions in {pred_dir} over {n} test images "
          f"(IoU>={thr}, conf>={conf}, text-area merged, header+footer combined)")
    print("=" * 74)
    print(f"{'class':13} {'P':>7} {'R':>7} {'F1':>7} {'meanIoU':>8} "
          f"{'TP':>5} {'FP':>5} {'FN':>5}")
    Fs = []
    for c in CANON:
        P, R, F1, miou, tp, fp, fn = score(gt[c], pred[c], thr)
        Fs.append(F1)
        print(f"{CANON[c]:13} {P:7.3f} {R:7.3f} {F1:7.3f} {miou:8.3f} "
              f"{tp:5d} {fp:5d} {fn:5d}")
    print("-" * 74)
    print(f"{'mean F1':13} {sum(Fs) / len(Fs):23.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
