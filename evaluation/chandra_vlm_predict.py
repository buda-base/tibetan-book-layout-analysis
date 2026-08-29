#!/usr/bin/env python3
"""Run Datalab's Chandra 2 (datalab-to/chandra-ocr-2) OCR+layout VLM off-the-shelf
on a folder of images and write YOLO-format predictions in OUR 4-class schema:

    0 header    1 text-area    2 footnote    3 footer

Chandra is an OCR model: the `ocr_layout` prompt returns HTML whose top-level
<div>s carry data-label / data-bbox (normalized 0-1000). We use chandra's own
parser (`InferenceManager.generate(...).chunks`) which yields per-block
{bbox:[x1,y1,x2,y2] in pixels, label, content}. We keep only the header /
footer / footnote / body-text labels and drop figures/tables/etc. Text-area is
left as the individual blocks Chandra returns -- the canonical evaluator merges
them into a single envelope per page.

Talks to a vLLM server (launched separately, e.g. `chandra_vllm --gpu a10`) via
settings.VLLM_API_BASE. Chandra returns no per-block confidence, so rows are
written WITHOUT a confidence column -> a single, un-thresholdable operating
point (scored like Azure DI / Google Doc AI).

Output: one <stem>.txt per image with rows "cls cx cy w h" (normalized).
Also writes run_meta.json with the *actually imported* package version and the
model checkpoint / vLLM base actually used.

Resumable: images whose label file already exists are skipped.

Usage:
  python chandra_vlm_predict.py --source <img_dir> --out <out_dir> [--batch 16]
                                [--limit N] [--max-workers 16]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

# Chandra canonical layout labels (hyphenated) -> our class id (None = drop)
LABEL_MAP = {
    "Page-Header": 0,
    "Page-Footer": 3,
    "Footnote": 2,
    "Text": 1,
    "Section-Header": 1,
    "Caption": 1,
    "List-Group": 1,
    "List-Item": 1,
    "Bibliography": 1,
    "Code-Block": 1,
    "Table-Of-Contents": 1,
    "Form": 1,
    "Complex-Block": 1,
    "Equation-Block": 1,
    # dropped: Image, Figure, Table, Diagram, Chemical-Block, Blank-Page
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def collect_runtime_meta() -> dict:
    import importlib.metadata as md

    meta: dict = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "argv": sys.argv,
        "env": {k: os.environ.get(k) for k in (
            "VLLM_API_BASE", "VLLM_MODEL_NAME", "MODEL_CHECKPOINT",
            "MAX_OUTPUT_TOKENS", "BBOX_SCALE") if os.environ.get(k)},
    }
    try:
        meta["chandra_ocr_dist_version"] = md.version("chandra-ocr")
    except Exception as e:
        meta["chandra_ocr_dist_version_error"] = repr(e)
    for pkg in ("openai", "transformers", "torch"):
        try:
            meta[f"{pkg}_dist_version"] = md.version(pkg)
        except Exception:
            pass
    try:
        from chandra.settings import settings
        meta["chandra_settings"] = {
            "MODEL_CHECKPOINT": settings.MODEL_CHECKPOINT,
            "VLLM_API_BASE": settings.VLLM_API_BASE,
            "VLLM_MODEL_NAME": settings.VLLM_MODEL_NAME,
            "MAX_OUTPUT_TOKENS": settings.MAX_OUTPUT_TOKENS,
            "BBOX_SCALE": settings.BBOX_SCALE,
            "IMAGE_DPI": settings.IMAGE_DPI,
            "MIN_IMAGE_DIM": settings.MIN_IMAGE_DIM,
        }
    except Exception as e:
        meta["chandra_settings_error"] = repr(e)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16,
                    help="images submitted per generate() call")
    ap.add_argument("--max-workers", type=int, default=16,
                    help="client-side concurrency to the vLLM server")
    ap.add_argument("--max-retries", type=int, default=0,
                    help="Chandra re-generation retries. Its repeat-token "
                         "detector false-positives on Tibetan script and forces "
                         "up to 6 full re-decodes/page, so default 0.")
    ap.add_argument("--max-output-tokens", type=int, default=4000,
                    help="cap decode length. Chandra tries to fully OCR the page "
                         "and loops on Tibetan it can't read, hitting the 12384 "
                         "default cap (~5 min/page). 4000 bounds runaway pages "
                         "while still emitting top-to-bottom layout divs.")
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
          f"-> Chandra 2 ocr_layout, batch {args.batch}, workers {args.max_workers}",
          flush=True)

    meta = collect_runtime_meta()
    print("runtime:", json.dumps({k: meta.get(k) for k in (
        "chandra_ocr_dist_version", "chandra_settings")}, indent=2, default=str),
        flush=True)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock

    from chandra.model import InferenceManager
    from chandra.model.schema import BatchInputItem

    manager = InferenceManager(method="vllm")

    stats = {"done": 0, "errors": 0, "tok_max": 0, "tok_sum": 0}
    unknown: dict[str, int] = {}
    lock = Lock()

    def process(p: Path):
        im = Image.open(p).convert("RGB")
        W, H = im.size
        item = BatchInputItem(image=im, prompt_type="ocr_layout")
        o = manager.generate([item], max_retries=args.max_retries,
                             max_failure_retries=0,
                             max_output_tokens=args.max_output_tokens,
                             include_headers_footers=True)[0]
        tc = int(getattr(o, "token_count", 0) or 0)
        err = bool(getattr(o, "error", False))
        lines = []
        local_unknown: dict[str, int] = {}
        for b in (o.chunks or []):
            label = b.get("label")
            cls = LABEL_MAP.get(label)
            if cls is None:
                local_unknown[str(label)] = local_unknown.get(str(label), 0) + 1
                continue
            bb = b.get("bbox")
            if not bb or len(bb) < 4:
                continue
            x1, y1, x2, y2 = bb[:4]
            if x2 <= x1 or y2 <= y1:
                continue
            cx = ((x1 + x2) / 2) / W
            cy = ((y1 + y2) / 2) / H
            w = (x2 - x1) / W
            h = (y2 - y1) / H
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        # atomic write
        tmp = lbl_dir / f"{p.stem}.txt.tmp"
        tmp.write_text("\n".join(lines))
        tmp.rename(lbl_dir / f"{p.stem}.txt")
        return p, tc, err, local_unknown

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(process, p): p for p in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            p = futs[fut]
            try:
                _, tc, err, lu = fut.result()
            except Exception as e:
                print(f"  FAIL {p.name}: {e!r}", flush=True)
                (lbl_dir / f"{p.stem}.txt").write_text("")
                with lock:
                    stats["errors"] += 1
                    stats["done"] += 1
                continue
            with lock:
                stats["done"] += 1
                stats["tok_sum"] += tc
                stats["tok_max"] = max(stats["tok_max"], tc)
                if err:
                    stats["errors"] += 1
                for k, v in lu.items():
                    unknown[k] = unknown.get(k, 0) + v
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} (tok avg "
                      f"{stats['tok_sum'] / max(1, stats['done']):.0f} "
                      f"max {stats['tok_max']}, errors {stats['errors']})",
                      flush=True)

    done = stats["done"]
    errors = stats["errors"]
    tok_max = stats["tok_max"]
    tok_sum = stats["tok_sum"]

    meta["done"] = done
    meta["errors"] = errors
    meta["max_retries"] = args.max_retries
    meta["token_count_max"] = tok_max
    meta["token_count_avg"] = round(tok_sum / done, 1) if done else 0
    meta["dropped_labels"] = dict(sorted(unknown.items(), key=lambda kv: -kv[1]))
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"done: {done} images ({errors} errors) -> {lbl_dir}", flush=True)
    if unknown:
        print("dropped (not in our schema):", meta["dropped_labels"], flush=True)
    print("wrote", out / "run_meta.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
