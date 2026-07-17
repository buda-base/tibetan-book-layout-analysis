#!/usr/bin/env python3
"""Run Google Document AI's Layout Parser processor over a folder of page
images and write YOLO-format label files in OUR 4-class space, so the output
can be scored with eval_pred_files.py exactly like any of our own models.

Layout Parser tags text blocks with a `type`. We map those to our classes:

    header      -> 0 header
    footer      -> 3 footer
    (no footnote type exists in Document AI's Layout Parser)
    everything else (paragraph, subtitle, heading-1..5, list, table, image)
                -> 1 text-area

Document AI's Layout Parser gives no per-block confidence, so no score column
is written (all detections are treated as equally confident at scoring time,
like Azure DI -- a single, un-thresholdable operating point).

Setup:
    pip install google-cloud-documentai
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

The script will reuse an existing LAYOUT_PARSER_PROCESSOR in the given
project/location, or create one on first run (requires the Document AI API to
be enabled on the project and documentai.processors.create permission).

Usage:
    python google_docai_predict.py --source testset/images/test --out gdocai_pred \
        --project bdrcetextscorpus --location us
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted
from google.cloud import documentai_v1 as documentai

IMG_EXTS = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

TYPE_TO_CLASS = {
    "header": 0,
    "footer": 3,
    # everything else (paragraph, subtitle, heading-1..5, ...) -> 1 text-area
}


def get_or_create_processor(client: documentai.DocumentProcessorServiceClient,
                             project: str, location: str,
                             display_name: str) -> str:
    parent = client.common_location_path(project, location)
    for p in client.list_processors(parent=parent):
        if p.type_ == "LAYOUT_PARSER_PROCESSOR" and p.display_name == display_name:
            return p.name
    print(f"no existing '{display_name}' LAYOUT_PARSER_PROCESSOR found, creating one...",
          flush=True)
    processor = client.create_processor(
        parent=parent,
        processor=documentai.Processor(
            display_name=display_name, type_="LAYOUT_PARSER_PROCESSOR"),
    )
    return processor.name


def walk_blocks(blocks, out):
    for b in blocks:
        bbox = b.bounding_box
        cls = 1
        if b.text_block and b.text_block.type_:
            cls = TYPE_TO_CLASS.get(b.text_block.type_, 1)
        if bbox and (bbox.normalized_vertices or bbox.vertices):
            out.append((cls, bbox))
        children = []
        if b.text_block:
            children = b.text_block.blocks
        elif b.table_block:
            children = []  # table cells have their own nested structure; skip
        elif b.list_block:
            children = []
        if children:
            walk_blocks(children, out)


def bbox_to_norm(bbox, W, H) -> tuple[float, float, float, float] | None:
    if bbox.normalized_vertices:
        xs = [v.x for v in bbox.normalized_vertices]
        ys = [v.y for v in bbox.normalized_vertices]
    elif bbox.vertices:
        xs = [v.x / W for v in bbox.vertices]
        ys = [v.y / H for v in bbox.vertices]
    else:
        return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    return (min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0), w, h)


def process_one(client, name: str, ip: Path, mime: str) -> list[str]:
    request = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(content=ip.read_bytes(), mime_type=mime),
        process_options=documentai.ProcessOptions(
            layout_config=documentai.ProcessOptions.LayoutConfig(
                return_bounding_boxes=True)),
    )
    for attempt in range(6):
        try:
            result = client.process_document(request=request)
            break
        except ResourceExhausted:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    doc = result.document
    page = doc.pages[0] if doc.pages else None
    W = page.dimension.width if page and page.dimension.width else 1.0
    H = page.dimension.height if page and page.dimension.height else 1.0

    boxes = []
    if doc.document_layout and doc.document_layout.blocks:
        walk_blocks(doc.document_layout.blocks, boxes)

    lines = []
    for cls, bbox in boxes:
        norm = bbox_to_norm(bbox, W, H)
        if norm is None:
            continue
        cx, cy, w, h = norm
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="folder of page images")
    ap.add_argument("--out", default="gdocai_pred", help="output folder for labels")
    ap.add_argument("--project", required=True, help="GCP project id")
    ap.add_argument("--location", default="us", help="Document AI region (us/eu)")
    ap.add_argument("--processor-id", default="",
                    help="reuse this processor id instead of listing/creating one")
    ap.add_argument("--display-name", default="tibetan-layout-eval",
                    help="display name to find/create the LAYOUT_PARSER_PROCESSOR")
    ap.add_argument("--processor-version", default="",
                    help="processor version id (e.g. pretrained-layout-parser-v1.6-2026-01-13); "
                         "default processor_version is PDF/DOCX-only and rejects images")
    ap.add_argument("--limit", type=int, default=0, help="cap #images (0 = all)")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests")
    args = ap.parse_args()

    client = documentai.DocumentProcessorServiceClient(
        client_options={"api_endpoint": f"{args.location}-documentai.googleapis.com"})

    if args.processor_id:
        name = client.processor_path(args.project, args.location, args.processor_id)
    else:
        name = get_or_create_processor(client, args.project, args.location, args.display_name)
    if args.processor_version:
        name = f"{name}/processorVersions/{args.processor_version}"
    print(f"using processor: {name}", flush=True)

    out = Path(args.out)
    lbl_dir = out / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(p for p in Path(args.source).iterdir() if p.suffix.lower() in IMG_EXTS)
    if args.limit:
        imgs = imgs[: args.limit]
    todo = [ip for ip in imgs if not (lbl_dir / f"{ip.stem}.txt").exists()]
    print(f"{len(imgs)} images ({len(todo)} to do, {len(imgs) - len(todo)} cached) "
          f"-> Google Document AI (Layout Parser), {args.workers} workers", flush=True)

    def process(ip: Path) -> str:
        dst = lbl_dir / f"{ip.stem}.txt"
        try:
            lines = process_one(client, name, ip, IMG_EXTS[ip.suffix.lower()])
        except (GoogleAPICallError, Exception) as e:  # keep going on single-page failures
            return f"!! {ip.name}: {e}"
        dst.write_text("\n".join(lines) + ("\n" if lines else ""))
        return "ok"

    done = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, ip): ip for ip in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r.startswith("!!"):
                fail += 1
                print(f"  {r}", flush=True)
            else:
                done += 1
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} ...", flush=True)

    (out / "data.yaml").write_text(
        "names:\n  0: header\n  1: text-area\n  2: footnote\n  3: footer\n")
    print(f"done: {done} analyzed, {fail} failed -> {lbl_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
