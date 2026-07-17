#!/usr/bin/env python3
"""Run PP-DocLayout-L (PaddlePaddle via paddleocr PPStructure, off-the-shelf) on a
folder of images and write YOLO-format predictions in OUR 4-class schema:

    0 header    1 text-area    2 footnote    3 footer

Uses the same label normalisation as HFF-Remover's PPDocLayoutDetector.

Output: one <stem>.txt per image with rows "cls cx cy w h conf" (normalized).
Resumable: images whose label file already exists are skipped.

Usage:
  python pp_doclayout_predict.py --source <img_dir> --out <out_dir>
                                 [--conf 0.2] [--limit N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

_LABEL_MAP = {
    "header": 0,
    "page_header": 0,
    "footer": 3,
    "page_footer": 3,
    "page_number": 3,
    "number": 3,
    "footnote": 2,
    "footnotes": 2,
    "text": 1,
    "text-area": 1,
    "text_area": 1,
    "plain_text": 1,
    "plain text": 1,
    "paragraph": 1,
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def norm_label(raw: str) -> int | None:
    key = str(raw).lower().replace("-", "_").replace(" ", "_")
    return _LABEL_MAP.get(key)


def bbox_to_yolo(bbox, W: int, H: int, cls: int, conf: float) -> str:
    if len(bbox) == 4:
        x1, y1, x2, y2 = (float(x) for x in bbox)
    elif len(bbox) == 8:
        xs = bbox[0::2]
        ys = bbox[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        return ""
    cx = ((x1 + x2) / 2) / W
    cy = ((y1 + y2) / 2) / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--model-dir", default="",
                    help="Fine-tuned Paddle inference dir (default: off-the-shelf)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"],
                    help="gpu (default) or cpu; cpu forces MKL-DNN off (see below)")
    args = ap.parse_args()

    src = Path(args.source)
    lbl_dir = Path(args.out) / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for stray in lbl_dir.glob("*.txt.tmp"):
        stray.unlink()  # left over from a run killed mid-write

    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    if args.limit:
        imgs = imgs[: args.limit]
    todo = [p for p in imgs if not (lbl_dir / f"{p.stem}.txt").exists()]
    print(f"{len(imgs)} images ({len(todo)} to do, {len(imgs) - len(todo)} cached) "
          f"-> PP-DocLayout-L (LayoutDetection), conf {args.conf}", flush=True)

    from paddleocr import LayoutDetection

    engine_config = {"device_type": args.device}
    if args.device == "cpu":
        # "paddle" run_mode disables MKL-DNN, which crashes on this CPU/model
        # combo with a PIR attribute conversion error (oneDNN kernel
        # incompatibility).
        engine_config["run_mode"] = "paddle"
    model_kwargs = {"model_name": "PP-DocLayout-L", "engine_config": engine_config}
    if args.model_dir:
        model_kwargs["model_dir"] = str(Path(args.model_dir).expanduser().resolve())
        print(f"model_dir={model_kwargs['model_dir']}", flush=True)
    model = LayoutDetection(**model_kwargs)

    from tqdm import tqdm

    dropped = {}
    done = 0
    for i, ip in enumerate(tqdm(todo, desc="PP-DocLayout-L", mininterval=0.1), 1):
        out = model.predict(str(ip), batch_size=1, layout_nms=True, threshold=args.conf)
        lines = []
        for res in out:
            data = res.json if hasattr(res, "json") else None
            if isinstance(data, dict) and "res" in data:
                data = data["res"]
            if data is None and hasattr(res, "save_to_json"):
                import json as _json, tempfile, os
                d = tempfile.mkdtemp()
                res.save_to_json(save_path=d)
                for fn in os.listdir(d):
                    data = _json.loads(Path(d, fn).read_text())
                    break
            if not isinstance(data, dict):
                continue
            boxes = data.get("boxes") or []
            for item in boxes:
                if not isinstance(item, dict):
                    continue
                label = item.get("label") or item.get("type") or item.get("category")
                score = float(item.get("score", item.get("confidence", 1.0)))
                if score < args.conf:
                    continue
                cls = norm_label(str(label))
                if cls is None:
                    dropped[str(label)] = dropped.get(str(label), 0) + 1
                    continue
                coord = item.get("coordinate") or item.get("bbox") or item.get("box")
                if coord is None:
                    continue
                from PIL import Image
                with Image.open(ip) as im:
                    W, H = im.size
                line = bbox_to_yolo(coord, W, H, cls, score)
                if line:
                    lines.append(line)
        dst = lbl_dir / f"{ip.stem}.txt"
        tmp = dst.with_suffix(".txt.tmp")
        tmp.write_text("\n".join(lines))
        tmp.replace(dst)  # atomic: a kill mid-run can't leave a half-written label file
        done += 1

    print(f"done: {done} images -> {lbl_dir}", flush=True)
    if dropped:
        print("dropped labels:", dict(sorted(dropped.items(), key=lambda kv: -kv[1])),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
