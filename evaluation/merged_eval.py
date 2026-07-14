#!/usr/bin/env python3
"""Region-level (merged-box) evaluation — an ADDITIONAL test that does not touch
the data. For each image and each class, all boxes of that class are merged into
a single envelope box (min/max corners), for BOTH ground truth and predictions
(so at most one box per class per image). We then score how well the model finds
the right *region* for each class, independent of how finely it splits boxes.

Reports, per class:
  - region mAP@0.5   (AP over images; pred score = max box confidence, conf>=floor)
  - at an operating conf (0.25): precision / recall / F1 and mean IoU of the
    merged boxes on images where both GT and prediction are present.

Usage:
  python merged_eval.py <weights> <dataset_dir> [conf_op] [conf_floor] [device] [framework]
  framework in {doclayout, rtdetr}  (default: doclayout)
"""
from __future__ import annotations

import sys
from pathlib import Path

NAMES = {0: "header", 1: "text-area", 2: "footnote", 3: "footer"}


def load_names(ddir: Path):
    """Read class names from <ddir>/data.yaml; fall back to 4-class default."""
    yml = ddir / "data.yaml"
    if not yml.exists():
        return dict(NAMES)
    names, in_names = {}, False
    for ln in yml.read_text().splitlines():
        if ln.strip().startswith("names:"):
            in_names = True
            continue
        if in_names:
            s = ln.strip()
            if not s or ":" not in s or not s.split(":")[0].strip().isdigit():
                if ln and not ln[0].isspace():
                    break
                continue
            k, v = s.split(":", 1)
            names[int(k.strip())] = v.strip()
    return names or dict(NAMES)


def load_model(framework: str, weights: str):
    if framework == "rtdetr":
        from ultralytics import RTDETR
        return RTDETR(weights)
    from doclayout_yolo import YOLOv10
    return YOLOv10(weights)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def merge(boxes):
    return [
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    ]


def ap50(dets, npos):
    """VOC all-points AP from (score, tp) list at IoU 0.5."""
    if npos == 0:
        return float("nan")
    dets = sorted(dets, key=lambda x: -x[0])
    tp = fp = 0
    rec, prec = [], []
    for s, t in dets:
        tp += t
        fp += 1 - t
        rec.append(tp / npos)
        prec.append(tp / (tp + fp))
    # monotonic decreasing precision envelope, integrate over recall
    mrec = [0.0] + rec + [rec[-1] if rec else 0.0]
    mpre = [0.0] + prec + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def main() -> int:
    weights, ddir = sys.argv[1], Path(sys.argv[2])
    conf_op = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    conf_floor = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
    device = sys.argv[5] if len(sys.argv) > 5 else "0"
    framework = sys.argv[6] if len(sys.argv) > 6 else "doclayout"
    img_dir, lbl_dir = ddir / "images/test", ddir / "labels/test"

    global NAMES
    NAMES = load_names(ddir)

    model = load_model(framework, weights)
    results = model.predict(source=str(img_dir), conf=conf_floor, imgsz=1024,
                            device=device, stream=True, verbose=False)

    dets = {c: [] for c in NAMES}
    npos = {c: 0 for c in NAMES}
    op = {c: {"tp": 0, "fp": 0, "fn": 0, "iou": 0.0, "n": 0} for c in NAMES}

    n_img = 0
    for r in results:
        n_img += 1
        H, W = r.orig_shape
        stem = Path(r.path).stem
        gt = {c: [] for c in NAMES}
        lp = lbl_dir / f"{stem}.txt"
        if lp.exists():
            for ln in lp.read_text().splitlines():
                p = ln.split()
                if len(p) < 5:
                    continue
                c = int(p[0]); cx, cy, w, h = (float(x) for x in p[1:5])
                gt[c].append([(cx - w / 2) * W, (cy - h / 2) * H,
                              (cx + w / 2) * W, (cy + h / 2) * H])
        preds = {c: [] for c in NAMES}
        if r.boxes is not None:
            for b, cf, cl in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                 r.boxes.cls.tolist()):
                preds[int(cl)].append((b, cf))

        for c in NAMES:
            gtm = merge(gt[c]) if gt[c] else None
            if gtm is not None:
                npos[c] += 1
            if preds[c]:
                mp = merge([b for b, _ in preds[c]])
                score = max(cf for _, cf in preds[c])
                tp = 1 if (gtm is not None and iou(mp, gtm) >= 0.5) else 0
                dets[c].append((score, tp))
            # operating point
            opp = [b for b, cf in preds[c] if cf >= conf_op]
            pm = merge(opp) if opp else None
            if pm is not None and gtm is not None:
                v = iou(pm, gtm)
                op[c]["iou"] += v; op[c]["n"] += 1
                if v >= 0.5:
                    op[c]["tp"] += 1
                else:
                    op[c]["fp"] += 1; op[c]["fn"] += 1
            elif pm is not None:
                op[c]["fp"] += 1
            elif gtm is not None:
                op[c]["fn"] += 1

    print(f"region-level (merged one-box-per-class) eval on {n_img} test images")
    print(f"conf_op={conf_op}  conf_floor={conf_floor}")
    print("=" * 78)
    print(f"{'class':11} {'regionmAP50':>11} | {'P':>6} {'R':>6} {'F1':>6} {'meanIoU':>8}  (@conf {conf_op})")
    maps = []
    for c in NAMES:
        a = ap50(dets[c], npos[c])
        maps.append(a)
        tp, fp, fn = op[c]["tp"], op[c]["fp"], op[c]["fn"]
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F1 = 2 * P * R / (P + R) if P + R else 0.0
        miou = op[c]["iou"] / op[c]["n"] if op[c]["n"] else 0.0
        print(f"{NAMES[c]:11} {a:11.4f} | {P:6.3f} {R:6.3f} {F1:6.3f} {miou:8.3f}")
    print("-" * 78)
    print(f"{'mean':11} {sum(maps)/len(maps):11.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
