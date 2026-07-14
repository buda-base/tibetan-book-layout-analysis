#!/usr/bin/env python3
"""Canonical-space evaluation for cross-curriculum comparison.

Every model's predictions AND the dataset GT are mapped into one common label
space so all curricula can be compared on identical footing:

    0 header-footer  header and footer COMBINED into one class. Boxes are kept
                     separate and matched individually (proper detection AP) --
                     they are NOT merged into a single envelope, which would be
                     geometric nonsense (header top + footer bottom = whole page).
    1 text-area      all boxes MERGED into one envelope per page, for BOTH gt and
                     prediction. This performs the text-area merge as post-
                     processing for models that were not trained on merged boxes.
    2 footnote       unchanged.

The <remap> argument maps the model's native class ids -> canonical ids:
    4-class model (0 h,1 ta,2 fn,3 footer):  "0:0,1:1,2:2,3:0"
    3-class model (0 hf,1 ta,2 fn):          "0:0,1:1,2:2"

Per canonical class we report AP@0.5, AP@[.5:.95], and P/R/F1 + meanIoU at an
operating confidence.

Usage:
  python canon_eval.py <weights> <dataset_dir> <remap> [conf_op] [conf_floor] [device] [framework]
"""
from __future__ import annotations

import sys
from pathlib import Path

CANON = {0: "header-footer", 1: "text-area", 2: "footnote"}
MERGE_CLASSES = {1}                       # text-area -> single envelope
IOUS = [0.5 + 0.05 * i for i in range(10)]  # .50 .55 ... .95


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def envelope(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def voc_ap(rec, prec):
    mrec = [0.0] + list(rec) + [1.0]
    mpre = [0.0] + list(prec) + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def ap_class(gt_by_img, preds, thr):
    """gt_by_img: {img: [box,...]}; preds: [(img, box, score)]."""
    npos = sum(len(v) for v in gt_by_img.values())
    if npos == 0:
        return float("nan")
    preds = sorted(preds, key=lambda x: -x[2])
    matched = {img: [False] * len(bs) for img, bs in gt_by_img.items()}
    tp, fp = [], []
    for img, box, _ in preds:
        best, bj = 0.0, -1
        for j, g in enumerate(gt_by_img.get(img, [])):
            if matched[img][j]:
                continue
            v = iou(box, g)
            if v > best:
                best, bj = v, j
        if best >= thr and bj >= 0:
            tp.append(1); fp.append(0); matched[img][bj] = True
        else:
            tp.append(0); fp.append(1)
    ctp = cfp = 0
    rec, prec = [], []
    for i in range(len(preds)):
        ctp += tp[i]; cfp += fp[i]
        rec.append(ctp / npos); prec.append(ctp / (ctp + cfp))
    return voc_ap(rec, prec)


def op_class(gt_by_img, preds, conf_op, thr=0.5):
    """Operating-point P/R/F1 + meanIoU at conf>=conf_op, IoU>=thr."""
    npos = sum(len(v) for v in gt_by_img.values())
    preds = sorted([p for p in preds if p[2] >= conf_op], key=lambda x: -x[2])
    matched = {img: [False] * len(bs) for img, bs in gt_by_img.items()}
    tp = fp = 0
    ious = []
    for img, box, _ in preds:
        best, bj = 0.0, -1
        for j, g in enumerate(gt_by_img.get(img, [])):
            if matched[img][j]:
                continue
            v = iou(box, g)
            if v > best:
                best, bj = v, j
        if best >= thr and bj >= 0:
            tp += 1; matched[img][bj] = True; ious.append(best)
        else:
            fp += 1
    fn = npos - tp
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    miou = sum(ious) / len(ious) if ious else 0.0
    return P, R, F1, miou


def load_model(framework, weights):
    if framework == "rtdetr":
        from ultralytics import RTDETR
        return RTDETR(weights)
    from doclayout_yolo import YOLOv10
    return YOLOv10(weights)


def main() -> int:
    weights, ddir = sys.argv[1], Path(sys.argv[2])
    remap = {int(k): int(v) for k, v in
             (p.split(":") for p in sys.argv[3].split(","))}
    conf_op = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    conf_floor = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
    device = sys.argv[6] if len(sys.argv) > 6 else "0"
    framework = sys.argv[7] if len(sys.argv) > 7 else "rtdetr"
    img_dir, lbl_dir = ddir / "images/test", ddir / "labels/test"

    model = load_model(framework, weights)
    results = model.predict(source=str(img_dir), conf=conf_floor, imgsz=1024,
                            device=device, stream=True, verbose=False)

    gt = {c: {} for c in CANON}       # class -> {img_idx: [boxes]}
    preds = {c: [] for c in CANON}    # class -> [(img_idx, box, score)]
    n_img = 0
    for r in results:
        idx = n_img
        n_img += 1
        H, W = r.orig_shape
        stem = Path(r.path).stem
        # GT (native -> canonical)
        raw = {c: [] for c in CANON}
        lp = lbl_dir / f"{stem}.txt"
        if lp.exists():
            for ln in lp.read_text().splitlines():
                p = ln.split()
                if len(p) < 5:
                    continue
                cc = remap.get(int(p[0]))
                if cc is None:
                    continue
                cx, cy, w, h = (float(x) for x in p[1:5])
                raw[cc].append([(cx - w / 2) * W, (cy - h / 2) * H,
                                (cx + w / 2) * W, (cy + h / 2) * H])
        for c in CANON:
            if not raw[c]:
                continue
            if c in MERGE_CLASSES:
                gt[c][idx] = [envelope(raw[c])]
            else:
                gt[c][idx] = raw[c]
        # predictions (native -> canonical)
        pr = {c: [] for c in CANON}
        if r.boxes is not None:
            for b, cf, cl in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                 r.boxes.cls.tolist()):
                cc = remap.get(int(cl))
                if cc is not None:
                    pr[cc].append((b, cf))
        for c in CANON:
            if not pr[c]:
                continue
            if c in MERGE_CLASSES:
                box = envelope([b for b, _ in pr[c]])
                preds[c].append((idx, box, max(cf for _, cf in pr[c])))
            else:
                for b, cf in pr[c]:
                    preds[c].append((idx, b, cf))

    print(f"canonical-space eval on {n_img} test images   "
          f"(remap {sys.argv[3]}, text-area merged, header+footer combined)")
    print(f"conf_op={conf_op}  conf_floor={conf_floor}")
    print("=" * 82)
    print(f"{'class':13} {'AP50':>7} {'AP50-95':>8} | {'P':>6} {'R':>6} "
          f"{'F1':>6} {'meanIoU':>8}  (@conf {conf_op})")
    m50, m5095 = [], []
    for c in CANON:
        aps = [ap_class(gt[c], preds[c], t) for t in IOUS]
        ap50 = aps[0]
        ap5095 = sum(aps) / len(aps)
        P, R, F1, miou = op_class(gt[c], preds[c], conf_op)
        m50.append(ap50); m5095.append(ap5095)
        print(f"{CANON[c]:13} {ap50:7.4f} {ap5095:8.4f} | {P:6.3f} {R:6.3f} "
              f"{F1:6.3f} {miou:8.3f}")
    print("-" * 82)
    print(f"{'mean':13} {sum(m50) / len(m50):7.4f} {sum(m5095) / len(m5095):8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
