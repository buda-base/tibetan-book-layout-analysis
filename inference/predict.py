#!/usr/bin/env python3
"""Run a trained detector over a folder of images and write YOLO-format
label files (one .txt per image: ``class cx cy w h`` normalized), suitable
for importing into the Ultralytics platform as editable pre-annotations.

A label file is written for *every* image (empty if no detections), so the
whole batch shows up as reviewable in the platform. A data.yaml with the
canonical class names is also written.

Usage:
    python predict.py --framework rtdetr \
        --weights runs/detect/runs/detect/rtdetr_l_1024/weights/best.pt \
        --source batch4_images --out batch4_pred --imgsz 1024 --conf 0.25
"""
from __future__ import annotations

import argparse
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def load_model(framework: str, weights: str):
    if framework == "rtdetr":
        from ultralytics import RTDETR

        return RTDETR(weights)
    from ultralytics import YOLO

    return YOLO(weights)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--framework", default="rtdetr", choices=["rtdetr", "ultralytics"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="folder of images")
    ap.add_argument("--out", default="batch4_pred")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    lbl_dir = out / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(p for p in Path(args.source).iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"{len(imgs)} images in {args.source}", flush=True)

    model = load_model(args.framework, args.weights)
    names = model.names

    # Stream straight from the directory (a Python list source makes Ultralytics
    # eager-load every image into RAM -> OOM on a 16 GB box). Non-image files
    # in the folder (e.g. this script) are skipped automatically.
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        stream=True,
        verbose=False,
    )

    n_img = n_box = n_empty = 0
    for r in results:
        stem = Path(r.path).stem
        lines = []
        if r.boxes is not None and len(r.boxes):
            for c, xywhn in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist()):
                x, y, w, h = xywhn
                lines.append(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        (lbl_dir / f"{stem}.txt").write_text(("\n".join(lines) + "\n") if lines else "")
        n_img += 1
        n_box += len(lines)
        n_empty += 0 if lines else 1
        if n_img % 250 == 0:
            print(f"  {n_img}/{len(imgs)} ...", flush=True)

    (out / "data.yaml").write_text(
        "names:\n" + "".join(f"  {i}: {names[i]}\n" for i in sorted(names))
    )
    print(
        f"done: {n_img} label files ({n_empty} empty), {n_box} boxes -> {lbl_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
