#!/usr/bin/env python3
"""Score a trained model on the test split with HFF-Remover's evaluator.

Produces a report directly comparable to the existing EricYolo / DocLayout
benchmark reports: it runs inference on the test images, writes predictions as
normalised COCO-format label files (``class cx cy w h conf``), then calls
``hff_remover.evaluate`` for per-class AP and mAP@0.5 / mAP@0.5:0.95.

Class ids are already canonical (0:header,1:text-area,2:footnote,3:footer) in
the merged dataset, so no remapping is needed.

Usage:
    python eval_hff.py --data dataset/data.yaml \
        --weights runs/detect/yolo26s_1280/weights/best.pt --imgsz 1280
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List

import yaml

# Canonical names, matching HFF-Remover/example_evaluate.py
CLASS_NAMES = {0: "header", 1: "text-area", 2: "footnote", 3: "footer"}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def test_dirs(data_yaml: Path):
    doc = yaml.safe_load(data_yaml.read_text())
    root = Path(doc.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    img_dir = root / doc["test"]
    lbl_dir = Path(str(img_dir).replace("/images/", "/labels/"))
    return img_dir, lbl_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--framework", choices=["ultralytics", "doclayout"], default="ultralytics")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.001, help="low conf for proper AP")
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", type=Path, default=Path("hff_eval"))
    parser.add_argument(
        "--hff-remover-src",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "HFF-Remover" / "src",
        help="path to HFF-Remover/src for importing hff_remover.evaluate",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.hff_remover_src))
    from hff_remover.evaluate import evaluate, print_report  # noqa: E402

    img_dir, gt_src = test_dirs(args.data)
    images: List[Path] = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"No test images in {img_dir}")

    gt_dir = args.out / "gt"
    pred_dir = args.out / "pred"
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    # GT: copy the ground-truth label files (already normalised YOLO format).
    for img in images:
        src = gt_src / f"{img.stem}.txt"
        (gt_dir / f"{img.stem}.txt").write_text(src.read_text() if src.is_file() else "")

    # Load model
    if args.framework == "ultralytics":
        from ultralytics import YOLO

        model = YOLO(str(args.weights))
    else:
        from doclayout_yolo import YOLOv10

        model = YOLOv10(str(args.weights))

    # Predict per image; write normalised "class cx cy w h conf" lines.
    for img in images:
        res = model.predict(
            str(img), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False
        )[0]
        lines: List[str] = []
        if res.boxes is not None and len(res.boxes) > 0:
            xywhn = res.boxes.xywhn.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            conf = res.boxes.conf.cpu().numpy()
            for (cx, cy, w, h), c, cf in zip(xywhn, cls, conf):
                lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {cf:.6f}")
        (pred_dir / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    result = evaluate(gt_dir=gt_dir, pred_dir=pred_dir, class_names=CLASS_NAMES)
    print_report(result, class_names=CLASS_NAMES)
    print(f"\nGT labels : {gt_dir}\nPred labels: {pred_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
