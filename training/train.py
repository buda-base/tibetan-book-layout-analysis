#!/usr/bin/env python3
"""Train a full-page HFF layout detector and validate on the test split.

Two frameworks are supported (choose with --framework):
  * ``ultralytics`` — YOLO26 etc. via ``from ultralytics import YOLO``
    (integrates directly with HFF-Remover's EricYolo/YOLO11 detectors).
  * ``doclayout``   — DocLayout-YOLO via ``from doclayout_yolo import YOLOv10``.
    Run this from the dedicated ``.venv_doclayout`` (see setup_doclayout.sh).

After ``train()`` the model is validated on the dataset ``test`` split and the
per-class mAP@0.5 / mAP@0.5:0.95 are printed.

Usage:
    python train.py --data dataset/data.yaml --framework ultralytics \
        --model yolo26s.pt --imgsz 1280 --epochs 100 --name yolo26s_1280
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path


def _patch_torch_load() -> None:
    """Force weights_only=False by default.

    PyTorch >=2.6 flipped torch.load's weights_only default to True; the
    DocLayout-YOLO fork's post-training strip_optimizer/final_eval then crashes
    trying to unpickle its own model class. We trust these local checkpoints.
    Uses setdefault, so any explicit weights_only=True (e.g. from Ultralytics'
    own safe loader) is still honoured.
    """
    import torch

    if getattr(torch.load, "_wo_patched", False):
        return
    _orig = torch.load

    @functools.wraps(_orig)
    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    _load._wo_patched = True
    torch.load = _load


_patch_torch_load()


def load_model(framework: str, model: str):
    if framework == "ultralytics":
        from ultralytics import YOLO

        return YOLO(model)
    if framework == "rtdetr":
        from ultralytics import RTDETR

        return RTDETR(model)
    if framework == "doclayout":
        from doclayout_yolo import YOLOv10

        return YOLOv10(model)
    raise SystemExit(f"Unknown framework: {framework}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--framework",
        choices=["ultralytics", "rtdetr", "doclayout"],
        default="ultralytics",
    )
    parser.add_argument(
        "--model",
        default="yolo26s.pt",
        help="weights/config: yolo26{n,s,m,l}.pt, or a DocLayout-YOLO .pt",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch",
        type=float,
        default=-1,
        help="-1 = auto (fill ~60%% VRAM); or an int batch size",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="hff")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--rect", action="store_true",
        help="rectangular training (efficient for wide pehcha pages)",
    )
    parser.add_argument(
        "--amp", default="true", choices=["true", "false"],
        help="mixed precision; set false for DocLayout-YOLO (its AMP check "
        "tries to load yolov8n.pt which the fork can't auto-download)",
    )
    parser.add_argument(
        "--save-period", type=int, default=-1,
        help="save a checkpoint every N epochs (-1 = only last/best)",
    )
    args = parser.parse_args()
    amp = args.amp == "true"

    batch = int(args.batch) if float(args.batch).is_integer() and args.batch > 0 else args.batch

    model = load_model(args.framework, args.model)
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        rect=args.rect,
        amp=amp,
        save_period=args.save_period,
    )

    print("\n=== Validating on TEST split ===", flush=True)
    metrics = model.val(
        data=str(args.data),
        split="test",
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=f"{args.name}_test",
    )
    names = getattr(model, "names", {}) or {}
    try:
        print(f"\ntest mAP@0.5     : {metrics.box.map50:.4f}")
        print(f"test mAP@0.5:0.95: {metrics.box.map:.4f}")
        print("per-class AP@0.5:")
        for i, ap in enumerate(metrics.box.ap50):
            print(f"  {names.get(i, i)!s:>10}: {ap:.4f}")
    except Exception as exc:  # pragma: no cover
        print(f"(could not pretty-print metrics: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
