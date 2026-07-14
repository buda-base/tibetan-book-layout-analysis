#!/usr/bin/env python3
"""Geometric / logical consistency audit of a YOLO HFF dataset.

Checks every label file across all splits and flags likely annotation mistakes.
IoU is computed in PIXEL space (needs image dims) so aspect ratio is honoured;
vertical-order rules use normalized y (0=top, 1=bottom).

Rules (ERROR = almost certainly wrong, WARN = suspicious / review):
  dup_same     WARN   two boxes, same class, IoU > IOU_DUP           (near-duplicate)
  dup_diff     ERROR  two boxes, different class, IoU > IOU_DUP      (conflicting label)
  footer_above_header  ERROR  a footer sits above a header
  header_below_text    ERROR  a header center is below the text-area center
  footnote_not_below   ERROR  a footnote center is above the text-area center
  footnote_above_header ERROR a footnote sits above a header
  footnote_below_footer WARN  a footnote sits below a footer
  header_low   WARN   header center in bottom half of page (cy > 0.55)
  footer_high  WARN   footer center in top half of page (cy < 0.45)
  hf_small     WARN   header/footer box area < SMALL_AREA of the page
  hf_corner    WARN   header/footer box center sits in an image corner
  oob          ERROR  box extends outside [0,1]
  degenerate   ERROR  box width or height <= 0 (or < MIN_WH)
  no_text      WARN   image has header/footer/footnote but no text-area

Outputs: prints a summary, writes audit_report.csv, and renders up to
EXAMPLES_PER_TYPE annotated example images per issue type into <out>/<type>/.

Usage: python audit_data.py <dataset_dir> <out_dir> [examples_per_type]
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from PIL import Image

NAMES = {0: "header", 1: "text-area", 2: "footnote", 3: "footer"}
SPLITS = ["train", "val", "test"]

IOU_DUP = 0.90
VMARGIN = 0.02      # vertical tolerance (fraction of page height)
MIN_WH = 1e-4       # normalized min side
SMALL_AREA = 0.0003  # header/footer < 0.03% of page area -> genuinely tiny outlier
                     # (footer median ~0.12%, header median ~0.84%, so this is a
                     #  true outlier, not the common small folio-number footer)
CORNER = 0.08       # center within CORNER of both a horiz. and vert. edge = corner
                     # (0.08 -> ~120 h/f; the loose 0.12 flagged ~970 legit ones)
EXAMPLES_PER_TYPE = 10

SEV = {
    "dup_same": "WARN", "dup_diff": "ERROR",
    "footer_above_header": "ERROR", "header_below_text": "ERROR",
    "footnote_not_below": "ERROR", "footnote_above_header": "ERROR",
    "footnote_below_footer": "WARN", "header_low": "WARN", "footer_high": "WARN",
    "hf_small": "WARN", "hf_corner": "WARN",
    "oob": "ERROR", "degenerate": "ERROR", "no_text": "WARN",
}
COLOR = {0: (0, 140, 255), 1: (0, 170, 0), 2: (200, 0, 200), 3: (0, 0, 235)}


def iou_px(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    ex_per = int(sys.argv[3]) if len(sys.argv) > 3 else EXAMPLES_PER_TYPE
    out.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    per_split = defaultdict(Counter)
    rows = []                       # (split, image, type, sev, detail)
    ex_saved = Counter()            # per issue-type examples rendered
    n_img = n_box = 0

    for split in SPLITS:
        img_dir, lbl_dir = root / "images" / split, root / "labels" / split
        if not img_dir.is_dir():
            continue
        for ip in sorted(img_dir.iterdir()):
            if ip.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            n_img += 1
            lp = lbl_dir / f"{ip.stem}.txt"
            try:
                W, H = Image.open(ip).size
            except Exception:
                continue
            boxes = []  # (cls, cx, cy, w, h, xyxy_px)
            if lp.exists():
                for ln in lp.read_text().splitlines():
                    p = ln.split()
                    if len(p) < 5:
                        continue
                    c = int(p[0])
                    cx, cy, w, h = (float(x) for x in p[1:5])
                    xy = [(cx - w / 2) * W, (cy - h / 2) * H,
                          (cx + w / 2) * W, (cy + h / 2) * H]
                    boxes.append((c, cx, cy, w, h, xy))
            n_box += len(boxes)

            img_issues = []  # (type, detail, [bad_box_idx])

            # per-box: oob, degenerate, tiny header/footer, corner header/footer
            for i, (c, cx, cy, w, h, xy) in enumerate(boxes):
                if w <= MIN_WH or h <= MIN_WH:
                    img_issues.append(("degenerate", f"{NAMES[c]} w={w:.4f} h={h:.4f}", [i]))
                if cx - w / 2 < -1e-3 or cy - h / 2 < -1e-3 or cx + w / 2 > 1 + 1e-3 or cy + h / 2 > 1 + 1e-3:
                    img_issues.append(("oob", f"{NAMES[c]} outside [0,1]", [i]))
                if c in (0, 3):  # header / footer only
                    if w * h < SMALL_AREA:
                        img_issues.append(("hf_small",
                            f"{NAMES[c]} area={w * h * 100:.3f}% (w={w:.3f} h={h:.3f})", [i]))
                    near_x = cx < CORNER or cx > 1 - CORNER
                    near_y = cy < CORNER or cy > 1 - CORNER
                    if near_x and near_y:
                        cx_s = "L" if cx < CORNER else "R"
                        cy_s = "T" if cy < CORNER else "B"
                        img_issues.append(("hf_corner",
                            f"{NAMES[c]} in {cy_s}{cx_s} corner (cx={cx:.2f} cy={cy:.2f})", [i]))

            # pairwise IoU duplicates
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    v = iou_px(boxes[i][5], boxes[j][5])
                    if v > IOU_DUP:
                        ci, cj = boxes[i][0], boxes[j][0]
                        if ci == cj:
                            img_issues.append(("dup_same", f"{NAMES[ci]} IoU={v:.2f}", [i, j]))
                        else:
                            img_issues.append(("dup_diff", f"{NAMES[ci]} vs {NAMES[cj]} IoU={v:.2f}", [i, j]))

            idx = defaultdict(list)
            for i, b in enumerate(boxes):
                idx[b[0]].append(i)
            headers, texts, foots, footers = (idx[0], idx[1], idx[2], idx[3])

            # text region center (union) from text-area boxes
            if texts:
                t_cy = sum(boxes[i][2] for i in texts) / len(texts)
                t_top = min(boxes[i][2] - boxes[i][4] / 2 for i in texts)
                t_bot = max(boxes[i][2] + boxes[i][4] / 2 for i in texts)
                t_ctr = (t_top + t_bot) / 2
            else:
                t_ctr = None

            # footer above header
            for fi in footers:
                for hi in headers:
                    if boxes[fi][2] + VMARGIN < boxes[hi][2]:
                        img_issues.append(("footer_above_header",
                            f"footer cy={boxes[fi][2]:.2f} < header cy={boxes[hi][2]:.2f}", [fi, hi]))
            # header below text center
            if t_ctr is not None:
                for hi in headers:
                    if boxes[hi][2] > t_ctr + VMARGIN:
                        img_issues.append(("header_below_text",
                            f"header cy={boxes[hi][2]:.2f} > text ctr={t_ctr:.2f}", [hi]))
                for ni in foots:
                    if boxes[ni][2] + VMARGIN < t_ctr:
                        img_issues.append(("footnote_not_below",
                            f"footnote cy={boxes[ni][2]:.2f} < text ctr={t_ctr:.2f}", [ni]))
            # footnote above header
            for ni in foots:
                for hi in headers:
                    if boxes[ni][2] + VMARGIN < boxes[hi][2]:
                        img_issues.append(("footnote_above_header",
                            f"footnote cy={boxes[ni][2]:.2f} < header cy={boxes[hi][2]:.2f}", [ni, hi]))
            # footnote below footer
            for ni in foots:
                for fi in footers:
                    if boxes[ni][2] > boxes[fi][2] + VMARGIN:
                        img_issues.append(("footnote_below_footer",
                            f"footnote cy={boxes[ni][2]:.2f} > footer cy={boxes[fi][2]:.2f}", [ni, fi]))
            # header low / footer high
            for hi in headers:
                if boxes[hi][2] > 0.55:
                    img_issues.append(("header_low", f"header cy={boxes[hi][2]:.2f}", [hi]))
            for fi in footers:
                if boxes[fi][2] < 0.45:
                    img_issues.append(("footer_high", f"footer cy={boxes[fi][2]:.2f}", [fi]))
            # no text but has margin elements
            if not texts and (headers or foots or footers):
                img_issues.append(("no_text", "no text-area box", []))

            for typ, detail, bad in img_issues:
                counts[typ] += 1
                per_split[split][typ] += 1
                rows.append((split, ip.name, typ, SEV[typ], detail))
                if ex_saved[typ] < ex_per:
                    _render(ip, boxes, bad, typ, detail, out)
                    ex_saved[typ] += 1

    # report
    (out / "audit_report.csv").parent.mkdir(parents=True, exist_ok=True)
    with (out / "audit_report.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "image", "type", "severity", "detail"])
        w.writerows(rows)

    print(f"audited {n_img} images / {n_box} boxes in {root}")
    print("=" * 64)
    errs = [t for t in counts if SEV[t] == "ERROR"]
    warns = [t for t in counts if SEV[t] == "WARN"]
    tot_e = sum(counts[t] for t in errs)
    tot_w = sum(counts[t] for t in warns)
    print(f"{'ISSUE':24} {'SEV':6} {'count':>7}   train/val/test")
    for t in sorted(counts, key=lambda x: (SEV[x] != "ERROR", -counts[x])):
        tv = f"{per_split['train'][t]}/{per_split['val'][t]}/{per_split['test'][t]}"
        print(f"{t:24} {SEV[t]:6} {counts[t]:7d}   {tv}")
    print("-" * 64)
    print(f"TOTAL ERRORS={tot_e}  WARNINGS={tot_w}")
    print(f"report -> {out/'audit_report.csv'}   examples -> {out}/<type>/")
    return 0


def _render(ip, boxes, bad, typ, detail, out):
    im = cv2.imread(str(ip))
    if im is None:
        return
    H, W = im.shape[:2]
    th = max(2, int(round(W / 700)))
    bad = set(bad)
    for i, (c, cx, cy, w, h, xy) in enumerate(boxes):
        x1, y1, x2, y2 = (int(v) for v in xy)
        col = (0, 0, 255) if i in bad else COLOR[c]
        t = th + 2 if i in bad else th
        cv2.rectangle(im, (x1, y1), (x2, y2), col, t)
        lab = ("BAD " if i in bad else "") + NAMES[c]
        fs = max(0.6, W / 2200)
        cv2.putText(im, lab, (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, col, 2, cv2.LINE_AA)
    banner = f"{typ}: {detail}"
    cv2.rectangle(im, (0, 0), (min(W, 1600), 46), (0, 0, 0), -1)
    cv2.putText(im, banner, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, max(0.8, W / 2000),
                (255, 255, 255), 2, cv2.LINE_AA)
    scale = 1500 / W if W > 1500 else 1.0
    if scale != 1.0:
        im = cv2.resize(im, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    d = out / typ
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{ip.stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    raise SystemExit(main())
