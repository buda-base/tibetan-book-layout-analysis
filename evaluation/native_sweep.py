#!/usr/bin/env python3
"""Native per-class confidence sweep (NO merging, NO class remapping).

Unlike canon_sweep.py (which merges text-area into one envelope and combines
header+footer), this scores every class exactly as the model emits it and
exactly as the dataset labels it — the real serving behaviour, including
multiple text-area boxes per page (e.g. two columns). For each native class it
greedily matches predictions to GT at IoU>=0.5 and sweeps the confidence
threshold, writing P/R/F1/TP/FP/FN to a CSV.

Usage:
  python native_sweep.py <weights> <dataset_dir> <out_csv> [framework] [device]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

GRID = [round(0.01 + 0.01 * i, 2) for i in range(99)]  # 0.01 .. 0.99


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def pr_at(gt_by_img, preds, conf, thr=0.5):
    npos = sum(len(v) for v in gt_by_img.values())
    sel = sorted([p for p in preds if p[2] >= conf], key=lambda x: -x[2])
    matched = {img: [False] * len(bs) for img, bs in gt_by_img.items()}
    tp = fp = 0
    for img, box, _ in sel:
        best, bj = 0.0, -1
        for j, g in enumerate(gt_by_img.get(img, [])):
            if matched[img][j]:
                continue
            v = iou(box, g)
            if v > best:
                best, bj = v, j
        if best >= thr and bj >= 0:
            tp += 1
            matched[img][bj] = True
        else:
            fp += 1
    fn = npos - tp
    P = tp / (tp + fp) if tp + fp else 1.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F1, tp, fp, fn


def load_names(ddir: Path):
    names = {}
    yml = ddir / "data.yaml"
    if yml.exists():
        in_names = False
        for ln in yml.read_text().splitlines():
            if ln.strip().startswith("names:"):
                in_names = True
                continue
            if in_names:
                s = ln.strip()
                if not s or not s[0].isdigit():
                    break
                k, v = s.split(":", 1)
                names[int(k)] = v.strip()
    return names or {0: "header", 1: "text-area", 2: "footnote", 3: "footer"}


def load_model(framework, weights):
    if framework == "rtdetr":
        from ultralytics import RTDETR
        return RTDETR(weights)
    from doclayout_yolo import YOLOv10
    return YOLOv10(weights)


def main() -> int:
    weights, ddir = sys.argv[1], Path(sys.argv[2])
    out_csv = sys.argv[3]
    framework = sys.argv[4] if len(sys.argv) > 4 else "rtdetr"
    device = sys.argv[5] if len(sys.argv) > 5 else "0"
    names = load_names(ddir)
    img_dir, lbl_dir = ddir / "images/test", ddir / "labels/test"

    model = load_model(framework, weights)
    results = model.predict(source=str(img_dir), conf=0.01, imgsz=1024,
                            device=device, stream=True, verbose=False)

    gt = {c: {} for c in names}       # class -> {img_idx: [boxes]}
    preds = {c: [] for c in names}    # class -> [(img_idx, box, score)]
    n_img = 0
    for r in results:
        idx = n_img
        n_img += 1
        H, W = r.orig_shape
        stem = Path(r.path).stem
        lp = lbl_dir / f"{stem}.txt"
        if lp.exists():
            for ln in lp.read_text().splitlines():
                p = ln.split()
                if len(p) < 5:
                    continue
                c = int(p[0])
                cx, cy, w, h = (float(x) for x in p[1:5])
                gt.setdefault(c, {}).setdefault(idx, []).append(
                    [(cx - w / 2) * W, (cy - h / 2) * H,
                     (cx + w / 2) * W, (cy + h / 2) * H])
        if r.boxes is not None:
            for b, cf, cl in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                 r.boxes.cls.tolist()):
                preds.setdefault(int(cl), []).append((idx, b, cf))

    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["class", "conf", "P", "R", "F1", "TP", "FP", "FN"])
        best = {}
        for c in sorted(names):
            for conf in GRID:
                P, R, F1, tp, fp, fn = pr_at(gt.get(c, {}), preds.get(c, []), conf)
                wr.writerow([names[c], conf, f"{P:.4f}", f"{R:.4f}",
                             f"{F1:.4f}", tp, fp, fn])
                if c not in best or F1 > best[c][1]:
                    best[c] = (conf, F1, P, R)
    print(f"wrote {out_csv} over {n_img} images (native per-class, IoU 0.5)")
    for c in sorted(names):
        conf, F1, P, R = best[c]
        print(f"  {names[c]:12} max-F1={F1:.3f} @conf {conf:.2f}  (P {P:.3f}, R {R:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
