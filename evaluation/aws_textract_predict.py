#!/usr/bin/env python3
"""Run Amazon Textract (AnalyzeDocument, FeatureTypes=["LAYOUT"]) over a folder
of page images and write YOLO-format label files in OUR 4-class space, so the
output can be scored with eval_pred_files.py exactly like any of our own models.

Textract's Layout feature tags blocks with a BlockType. We map those to our
classes as follows:

    LAYOUT_HEADER                          -> 0 header
    LAYOUT_FOOTER, LAYOUT_PAGE_NUMBER      -> 3 footer
    (no footnote type exists in Textract)
    everything else LAYOUT_* (TEXT, TITLE,
    SECTION_HEADER, LIST, TABLE, FIGURE,
    KEY_VALUE, ...)                        -> 1 text-area

Every LAYOUT_* block has a per-block Confidence (0-100); we write it normalized
to [0,1] as the 6th column so the canonical evaluator can sweep it exactly like
any other model.

Setup:
    pip install boto3
    aws configure   # or set AWS_PROFILE / rely on the default profile

Usage:
    python aws_textract_predict.py --source testset/images/test --out aws_pred
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

IMG_EXTS = {".jpg", ".jpeg", ".png"}

BLOCKTYPE_TO_CLASS = {
    "LAYOUT_HEADER": 0,
    "LAYOUT_FOOTER": 3,
    "LAYOUT_PAGE_NUMBER": 3,
    # everything else LAYOUT_* (TEXT, TITLE, SECTION_HEADER, LIST, TABLE,
    # FIGURE, KEY_VALUE, ...) -> 1 text-area
}

THROTTLE_ERRORS = {
    "ThrottlingException",
    "ProvisionedThroughputExceededException",
    "LimitExceededException",
    "InternalServerError",
}


def analyze(client, data: bytes) -> list:
    for attempt in range(6):
        try:
            resp = client.analyze_document(
                Document={"Bytes": data}, FeatureTypes=["LAYOUT"]
            )
            return resp["Blocks"]
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in THROTTLE_ERRORS and attempt < 5:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("analyze_document: too many retries")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", required=True, help="folder of page images")
    ap.add_argument("--out", default="aws_pred", help="output folder for labels")
    ap.add_argument("--region", default="us-east-1", help="AWS region for Textract")
    ap.add_argument("--profile", default=None, help="AWS profile (default: AWS_PROFILE/default)")
    ap.add_argument("--limit", type=int, default=0, help="cap #images (0 = all)")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests")
    args = ap.parse_args()

    session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    client = session.client("textract", region_name=args.region)

    out = Path(args.out)
    lbl_dir = out / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(p for p in Path(args.source).iterdir() if p.suffix.lower() in IMG_EXTS)
    if args.limit:
        imgs = imgs[: args.limit]
    todo = [ip for ip in imgs if not (lbl_dir / f"{ip.stem}.txt").exists()]
    print(
        f"{len(imgs)} images ({len(todo)} to do, {len(imgs) - len(todo)} cached) "
        f"-> Amazon Textract (Layout), region {args.region}, {args.workers} workers",
        flush=True,
    )

    def process(ip: Path) -> str:
        dst = lbl_dir / f"{ip.stem}.txt"
        try:
            blocks = analyze(client, ip.read_bytes())
        except Exception as e:  # keep going on single-page failures
            return f"!! {ip.name}: {e}"
        lines = []
        for b in blocks:
            bt = b.get("BlockType", "")
            if not bt.startswith("LAYOUT_"):
                continue
            cls = BLOCKTYPE_TO_CLASS.get(bt, 1)
            bbox = b.get("Geometry", {}).get("BoundingBox")
            if not bbox:
                continue
            cx = bbox["Left"] + bbox["Width"] / 2
            cy = bbox["Top"] + bbox["Height"] / 2
            w, h = bbox["Width"], bbox["Height"]
            conf = b.get("Confidence", 100.0) / 100.0
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.4f}")
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
        "names:\n  0: header\n  1: text-area\n  2: footnote\n  3: footer\n"
    )
    print(f"done: {done} analyzed, {fail} failed -> {lbl_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
