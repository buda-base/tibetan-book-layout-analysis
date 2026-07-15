#!/usr/bin/env python3
"""Run Azure AI Document Intelligence (prebuilt-layout) over a folder of page
images and write YOLO-format label files in OUR 4-class space, so the output can
be scored with eval_pred_files.py exactly like any of our own models.

Azure's layout model tags paragraphs with a `role`. We map those roles to our
classes as follows:

    pageHeader                 -> 0 header
    pageFooter, pageNumber     -> 3 footer
    footnote                   -> 2 footnote
    (no role, title,
     sectionHeading, other)    -> 1 text-area

Every body paragraph becomes a text-area box; our evaluator merges them into one
envelope per page, matching how we score our own models. Azure returns no
per-region confidence, so no score column is written (all detections are treated
as equally confident at scoring time — Azure gives you a single, un-thresholdable
answer).

Setup:
    pip install requests
    export AZURE_DI_ENDPOINT="https://<your-resource>.cognitiveservices.azure.com"
    export AZURE_DI_KEY="<key>"

Usage:
    python azure_di_predict.py --source testset/images/test --out azure_pred
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
API_VERSION = "2024-11-30"

ROLE_TO_CLASS = {
    "pageHeader": 0,
    "pageFooter": 3,
    "pageNumber": 3,
    "footnote": 2,
    # everything else (None, title, sectionHeading, ...) -> 1 text-area
}


def analyze(endpoint: str, key: str, data: bytes) -> dict:
    url = (f"{endpoint.rstrip('/')}/documentintelligence/documentModels/"
           f"prebuilt-layout:analyze?api-version={API_VERSION}")
    headers = {"Ocp-Apim-Subscription-Key": key,
               "Content-Type": "application/octet-stream"}
    r = requests.post(url, headers=headers, data=data, timeout=120)
    if r.status_code == 429:
        time.sleep(int(r.headers.get("Retry-After", "5")) + 1)
        r = requests.post(url, headers=headers, data=data, timeout=120)
    r.raise_for_status()
    op = r.headers["Operation-Location"]
    for _ in range(120):
        time.sleep(1.5)
        g = requests.get(op, headers={"Ocp-Apim-Subscription-Key": key}, timeout=60)
        g.raise_for_status()
        j = g.json()
        st = j.get("status")
        if st == "succeeded":
            return j["analyzeResult"]
        if st == "failed":
            raise RuntimeError(f"analyze failed: {j}")
    raise TimeoutError("analyze polling timed out")


def poly_to_norm_bbox(poly, W, H):
    xs, ys = poly[0::2], poly[1::2]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
    w, h = (x2 - x1) / W, (y2 - y1) / H
    return cx, cy, w, h


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="folder of page images")
    ap.add_argument("--out", default="azure_pred", help="output folder for labels")
    ap.add_argument("--limit", type=int, default=0, help="cap #images (0 = all)")
    ap.add_argument("--workers", type=int, default=8, help="parallel requests")
    args = ap.parse_args()

    endpoint = os.environ["AZURE_DI_ENDPOINT"]
    key = os.environ["AZURE_DI_KEY"]

    out = Path(args.out)
    lbl_dir = out / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(p for p in Path(args.source).iterdir()
                  if p.suffix.lower() in IMG_EXTS)
    if args.limit:
        imgs = imgs[:args.limit]
    todo = [ip for ip in imgs if not (lbl_dir / f"{ip.stem}.txt").exists()]
    print(f"{len(imgs)} images ({len(todo)} to do, {len(imgs) - len(todo)} cached) "
          f"-> Azure DI (prebuilt-layout, {API_VERSION}), {args.workers} workers",
          flush=True)

    def process(ip: Path) -> str:
        dst = lbl_dir / f"{ip.stem}.txt"
        try:
            res = analyze(endpoint, key, ip.read_bytes())
        except Exception as e:  # keep going on single-page failures
            return f"!! {ip.name}: {e}"
        pages = res.get("pages", [])
        if not pages:
            dst.write_text("")
            return "empty"
        W = pages[0].get("width") or 1.0
        H = pages[0].get("height") or 1.0
        lines = []
        for par in res.get("paragraphs", []):
            regs = par.get("boundingRegions") or []
            if not regs:
                continue
            cls = ROLE_TO_CLASS.get(par.get("role"), 1)
            cx, cy, w, h = poly_to_norm_bbox(regs[0]["polygon"], W, H)
            cx = min(max(cx, 0.0), 1.0); cy = min(max(cy, 0.0), 1.0)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
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
