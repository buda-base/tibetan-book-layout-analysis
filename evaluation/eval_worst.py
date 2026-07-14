#!/usr/bin/env python3
"""Render the worst test-set failures for a DocLayout-YOLO checkpoint.

For every test image: predict, greedily match predictions to ground truth per
class at IoU>=0.5, and count false positives (FP) + false negatives (FN).
Rank images by (FP+FN) and draw the top-N with:
  - matched GT / prediction .......... green
  - false-positive prediction (FP) ... red   (model invented / mislabeled)
  - missed GT (FN) ................... magenta "MISS"
Predictions are labelled "class conf". Output images go to OUT/.

Usage:
  python eval_worst.py <weights> <dataset_dir> <out_dir> [conf] [topn] [device]
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
from doclayout_yolo import YOLOv10

NAMES = {0: "header", 1: "text-area", 2: "footnote", 3: "footer"}
GREEN = (0, 170, 0)
RED = (0, 0, 235)
MAGENTA = (200, 0, 200)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(preds, gts, thr=0.5):
    """preds: [(box,conf)], gts: [box]; same-class. Returns tp_pred_idx set,
    fp_pred_idx list, matched_gt set, fn_gt_idx list."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i][1])
    matched_gt, tp = set(), set()
    for pi in order:
        best, bj = thr, -1
        for gj, gb in enumerate(gts):
            if gj in matched_gt:
                continue
            v = iou(preds[pi][0], gb)
            if v >= best:
                best, bj = v, gj
        if bj >= 0:
            tp.add(pi)
            matched_gt.add(bj)
    fp = [i for i in range(len(preds)) if i not in tp]
    fn = [j for j in range(len(gts)) if j not in matched_gt]
    return tp, fp, matched_gt, fn


def main() -> int:
    weights, ddir, outdir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    conf = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    topn = int(sys.argv[5]) if len(sys.argv) > 5 else 15
    device = sys.argv[6] if len(sys.argv) > 6 else "0"
    outdir.mkdir(parents=True, exist_ok=True)

    img_dir, lbl_dir = ddir / "images/test", ddir / "labels/test"
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    model = YOLOv10(weights)

    records = []  # (err, fp, fn, per-class-fp, path, draw-list)
    for ip in imgs:
        im = cv2.imread(str(ip))
        if im is None:
            continue
        H, W = im.shape[:2]
        # ground truth
        gts = {c: [] for c in NAMES}
        lp = lbl_dir / f"{ip.stem}.txt"
        if lp.exists():
            for ln in lp.read_text().splitlines():
                p = ln.split()
                if len(p) < 5:
                    continue
                c = int(p[0])
                cx, cy, w, h = (float(x) for x in p[1:5])
                gts.setdefault(c, []).append(
                    [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
                )
        # predictions
        r = model.predict(str(ip), conf=conf, imgsz=1024, device=device, verbose=False)[0]
        preds = {c: [] for c in NAMES}
        if r.boxes is not None:
            for b, cf, cl in zip(
                r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()
            ):
                preds.setdefault(int(cl), []).append((b, cf))

        draw = []  # (box, color, label)
        tot_fp = tot_fn = 0
        pc_fp = {}
        for c in NAMES:
            tp, fp, mgt, fn = match(preds[c], gts[c])
            for i, (bx, cf) in enumerate(preds[c]):
                if i in tp:
                    draw.append((bx, GREEN, f"{NAMES[c]} {cf:.2f}"))
                else:
                    draw.append((bx, RED, f"FP {NAMES[c]} {cf:.2f}"))
            for j, gb in enumerate(gts[c]):
                if j in fn:
                    draw.append((gb, MAGENTA, f"MISS {NAMES[c]}"))
                elif j in mgt:
                    draw.append((gb, GREEN, f"GT {NAMES[c]}"))
            tot_fp += len(fp)
            tot_fn += len(fn)
            if fp:
                pc_fp[NAMES[c]] = pc_fp.get(NAMES[c], 0) + len(fp)
        records.append((tot_fp + tot_fn, tot_fp, tot_fn, pc_fp, ip, draw, (H, W)))

    records.sort(key=lambda x: (-x[0], -x[1]))
    print(f"scored {len(records)} test images; top {topn} worst:")
    for rank, (err, fp, fn, pc_fp, ip, draw, (H, W)) in enumerate(records[:topn], 1):
        im = cv2.imread(str(ip))
        for bx, col, lab in draw:
            x1, y1, x2, y2 = (int(v) for v in bx)
            th = max(2, int(round(W / 700)))
            cv2.rectangle(im, (x1, y1), (x2, y2), col, th)
            fs = max(0.6, W / 2200)
            (tw, tht), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            yl = max(0, y1 - 4)
            cv2.rectangle(im, (x1, yl - tht - 4), (x1 + tw + 2, yl + 2), col, -1)
            cv2.putText(im, lab, (x1 + 1, yl - 2), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), 2, cv2.LINE_AA)
        # legend
        leg = f"rank {rank}  FP={fp} FN={fn}  " + " ".join(f"{k}:{v}FP" for k, v in pc_fp.items())
        cv2.rectangle(im, (0, 0), (min(W, 1400), 46), (0, 0, 0), -1)
        cv2.putText(im, leg, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, max(0.8, W / 2000),
                    (255, 255, 255), 2, cv2.LINE_AA)
        # downscale for viewing
        scale = 1500 / W if W > 1500 else 1.0
        if scale != 1.0:
            im = cv2.resize(im, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
        fpc = "_".join(f"{k}{v}" for k, v in pc_fp.items()) or "none"
        out = outdir / f"rank{rank:02d}_fp{fp}_fn{fn}_{fpc}__{ip.stem}.jpg"
        cv2.imwrite(str(out), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"  {rank:2d}. FP={fp} FN={fn} {pc_fp}  {ip.name}")
    print("done ->", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
