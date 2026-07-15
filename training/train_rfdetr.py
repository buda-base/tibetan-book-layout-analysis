#!/usr/bin/env python3
"""Fine-tune RF-DETR (Roboflow) on our COCO-format layout dataset.

Deliberately mirrors the winning tam2col RT-DETR recipe as closely as the two
frameworks allow: same dataset (dataset_v5_tam2col, 4 classes, text-area merged
except two-column pages), ~1024 input resolution, 100 epochs, effective batch 8,
early stopping with patience 20. Starts from Roboflow's Apache-2.0 COCO-pretrained
RF-DETR-Large base so the released model is cleanly licensed.

Usage:
  python train_rfdetr.py --dataset <coco_dir> --out <run_dir>
                         [--epochs 100] [--batch 4] [--grad-accum 2]
                         [--resolution 1008]
"""
from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--resolution", type=int, default=1008)  # 56*18, ~= RT-DETR 1024
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()

    from rfdetr import RFDETRLarge

    model = RFDETRLarge(resolution=args.resolution)
    model.train(
        dataset_dir=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        output_dir=args.out,
        early_stopping=True,
        early_stopping_patience=args.patience,
    )
    print("RFDETR_TRAIN_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
