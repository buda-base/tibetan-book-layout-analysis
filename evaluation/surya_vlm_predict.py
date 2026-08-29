#!/usr/bin/env python3
"""Run Surya 2's VLM layout model (LayoutPredictor / surya-ocr-2) off-the-shelf
on a folder of images and write YOLO-format predictions in OUR 4-class schema:

    0 header    1 text-area    2 footnote    3 footer

This is NOT FastLayoutPredictor (RF-DETR / surya_layout2). It shares a
SuryaInferenceManager with OCR/table-rec and needs a vLLM (GPU) or llama.cpp
backend. Text-area is left as the individual blocks the VLM returns -- the
canonical evaluator merges them into a single envelope per page.

Output: one <stem>.txt per image with rows "cls cx cy w h conf" (normalized).
Also writes run_meta.json with the *actually imported* package version,
checkpoint id, and backend -- do not trust the pip pin without this file.

Resumable: images whose label file already exists are skipped.

Usage:
  python surya_vlm_predict.py --source <img_dir> --out <out_dir> [--batch 4]
                              [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

# Surya VLM canonical labels -> our class id (None = drop)
# Keep this aligned with evaluation/surya_predict.py so the two Surya rows
# differ only by detector, not by mapping.
LABEL_MAP = {
    "PageHeader": 0,
    "PageFooter": 3,
    "Footnote": 2,
    "Text": 1,
    "SectionHeader": 1,
    "Caption": 1,
    "ListGroup": 1,
    "ListItem": 1,
    "Bibliography": 1,
    "Code": 1,
    "TableOfContents": 1,
    "Form": 1,
    "TextInlineMath": 1,
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def poly_to_bbox(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def collect_runtime_meta() -> dict:
    """Record what is actually imported/running, not what we intended to pin."""
    import importlib.metadata as md

    meta: dict = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "argv": sys.argv,
        "env": {
            k: os.environ.get(k)
            for k in (
                "SURYA_INFERENCE_BACKEND",
                "SURYA_INFERENCE_URL",
                "SURYA_INFERENCE_PARALLEL",
                "SURYA_INFERENCE_KEEP_ALIVE",
                "SURYA_GUIDED_LAYOUT",
                "SURYA_MODEL_CHECKPOINT",
                "LAYOUT_MODEL_CHECKPOINT",
            )
            if os.environ.get(k)
        },
    }
    try:
        import surya
        meta["surya_package_file"] = getattr(surya, "__file__", None)
        meta["surya_package_version_attr"] = getattr(surya, "__version__", None)
    except Exception as e:
        meta["surya_import_error"] = repr(e)
    try:
        meta["surya_ocr_dist_version"] = md.version("surya-ocr")
    except Exception as e:
        meta["surya_ocr_dist_version_error"] = repr(e)
    for pkg in ("torch", "transformers", "huggingface-hub", "vllm", "openai"):
        try:
            meta[f"{pkg}_dist_version"] = md.version(pkg)
        except Exception:
            pass
    try:
        from surya.settings import settings
        dump = {}
        for name in (
            "LAYOUT_MODEL_CHECKPOINT", "FOUNDATION_MODEL_CHECKPOINT",
            "MODEL_CHECKPOINT", "LAYOUT_BATCH_SIZE", "RECOGNITION_BATCH_SIZE",
            "INFERENCE_BACKEND", "INFERENCE_URL",
        ):
            if hasattr(settings, name):
                dump[name] = getattr(settings, name)
        # pydantic-settings: dump all fields we can
        if hasattr(settings, "model_dump"):
            try:
                full = settings.model_dump()
                for k, v in full.items():
                    if any(s in k.upper() for s in (
                            "CHECKPOINT", "MODEL", "INFERENCE", "LAYOUT",
                            "BACKEND", "VLLM")):
                        dump[k] = v
            except Exception:
                pass
        meta["surya_settings"] = {k: str(v) for k, v in dump.items()}
    except Exception as e:
        meta["surya_settings_error"] = repr(e)
    try:
        import torch
        meta["torch"] = {
            "version": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "device": (torch.cuda.get_device_name(0)
                       if torch.cuda.is_available() else None),
        }
    except Exception as e:
        meta["torch_error"] = repr(e)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=4,
                    help="images per LayoutPredictor call")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    lbl_dir = out / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    if args.limit:
        imgs = imgs[: args.limit]
    todo = [p for p in imgs if not (lbl_dir / f"{p.stem}.txt").exists()]
    print(f"{len(imgs)} images ({len(todo)} to do, {len(imgs) - len(todo)} cached) "
          f"-> Surya 2 VLM LayoutPredictor, batch {args.batch}",
          flush=True)

    meta = collect_runtime_meta()
    print("runtime:", json.dumps({
        k: meta.get(k) for k in (
            "surya_ocr_dist_version", "surya_package_version_attr",
            "surya_package_file", "surya_settings",
        )
    }, indent=2, default=str), flush=True)

    from surya.inference import SuryaInferenceManager
    from surya.layout import LayoutPredictor

    manager = SuryaInferenceManager()
    predictor = LayoutPredictor(manager)
    # Try to capture the live checkpoint after the manager exists.
    for attr in ("model_checkpoint", "checkpoint", "model_id", "repo_id"):
        if hasattr(manager, attr):
            meta[f"manager_{attr}"] = str(getattr(manager, attr))
    meta["manager_type"] = type(manager).__name__
    meta["predictor_type"] = type(predictor).__name__
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    done = 0
    unknown: dict[str, int] = {}
    errors = 0
    for i in range(0, len(todo), args.batch):
        chunk = todo[i : i + args.batch]
        pil = []
        for p in chunk:
            im = Image.open(p).convert("RGB")
            pil.append(im)
        results = predictor(pil)
        for p, im, res in zip(chunk, pil, results):
            W, H = im.size
            if getattr(res, "error", False):
                errors += 1
                print(f"  LAYOUT ERROR {p.name}", flush=True)
            lines = []
            for b in (getattr(res, "bboxes", None) or []):
                label = getattr(b, "label", None)
                cls = LABEL_MAP.get(label)
                if cls is None:
                    unknown[str(label)] = unknown.get(str(label), 0) + 1
                    continue
                if getattr(b, "polygon", None):
                    x1, y1, x2, y2 = poly_to_bbox(b.polygon)
                else:
                    bb = getattr(b, "bbox", None)
                    if not bb or len(bb) < 4:
                        continue
                    x1, y1, x2, y2 = bb[:4]
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H
                w = (x2 - x1) / W
                h = (y2 - y1) / H
                conf = float(b.confidence) if getattr(b, "confidence", None) is not None else 1.0
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.4f}")
            (lbl_dir / f"{p.stem}.txt").write_text("\n".join(lines))
            done += 1
        print(f"  {min(i + args.batch, len(todo))}/{len(todo)} ...", flush=True)

    meta["done"] = done
    meta["errors"] = errors
    meta["dropped_labels"] = dict(sorted(unknown.items(), key=lambda kv: -kv[1]))
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"done: {done} images ({errors} layout errors) -> {lbl_dir}", flush=True)
    if unknown:
        print("dropped (not in our schema):", meta["dropped_labels"], flush=True)
    print("wrote", out / "run_meta.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
