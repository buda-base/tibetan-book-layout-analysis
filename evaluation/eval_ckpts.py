#!/usr/bin/env python3
"""Evaluate one or more checkpoints on the TEST split and print overall +
per-class mAP50 and mAP50-95.

Usage:
  python eval_ckpts.py --framework {doclayout,rtdetr} --data <data.yaml> \
      --weights w1.pt,w2.pt [--imgsz 1024] [--batch 1] [--device 0] [--split test]
"""
from __future__ import annotations

import argparse


def load_model(framework: str, weights: str):
    if framework == "doclayout":
        from doclayout_yolo import YOLOv10
        return YOLOv10(weights)
    if framework == "rtdetr":
        from ultralytics import RTDETR
        return RTDETR(weights)
    raise ValueError(framework)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", required=True, choices=["doclayout", "rtdetr"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True, help="comma-separated .pt paths")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device", default="0")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    for w in args.weights.split(","):
        w = w.strip()
        if not w:
            continue
        print("=" * 72)
        print(f"WEIGHTS: {w}   split={args.split}")
        m = load_model(args.framework, w)
        r = m.val(data=args.data, split=args.split, imgsz=args.imgsz,
                  batch=args.batch, device=args.device, verbose=False,
                  save_json=False, plots=False)
        b = r.box
        names = r.names
        f1 = 2 * b.mp * b.mr / (b.mp + b.mr) if (b.mp + b.mr) else 0.0
        print(f"  overall  mAP50={b.map50:.4f}  mAP50-95={b.map:.4f}  "
              f"P={b.mp:.4f} R={b.mr:.4f} F1={f1:.4f}")
        print(f"    {'class':11} {'P':>6} {'R':>6} {'F1':>6} {'mAP50':>7} {'mAP50-95':>9}")
        for i, ci in enumerate(b.ap_class_index):
            p, rc = float(b.p[i]), float(b.r[i])
            cf1 = 2 * p * rc / (p + rc) if (p + rc) else 0.0
            print(f"    {names[int(ci)]:11} {p:6.3f} {rc:6.3f} {cf1:6.3f} "
                  f"{b.ap50[i]:7.4f} {b.maps[int(ci)]:9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
