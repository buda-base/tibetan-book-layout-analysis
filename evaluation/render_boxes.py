#!/usr/bin/env python3
"""Overlay YOLO-format boxes on an image for visual inspection of annotation
conventions. Downscales the output for quick viewing. Not part of the metric
pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# consistent colors across our schema and DocLayNet shared classes
COLORS = {
    "header": (220, 30, 30),        # red
    "page-header": (220, 30, 30),
    "footer": (30, 90, 220),        # blue
    "page-footer": (30, 90, 220),
    "footnote": (240, 140, 0),      # orange
    "text-area": (20, 160, 60),     # green
    "text": (20, 160, 60),
    "other": (150, 150, 150),
}


def load_boxes(label_path: Path, names: dict):
    out = []
    if not label_path.exists():
        return out
    for ln in label_path.read_text().splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        c = int(p[0])
        cx, cy, w, h = map(float, p[1:5])
        out.append((names.get(c, f"cls{c}"), cx, cy, w, h))
    return out


def render(img_path: Path, boxes, out_path: Path, max_dim=1400, title=None):
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    scale = min(1.0, max_dim / max(W, H))
    if scale < 1.0:
        im = im.resize((int(W * scale), int(H * scale)))
    W2, H2 = im.size
    dr = ImageDraw.Draw(im, "RGBA")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    lw = max(2, int(round(3 * scale)) + 1)
    for name, cx, cy, w, h in boxes:
        x1 = (cx - w / 2) * W2; y1 = (cy - h / 2) * H2
        x2 = (cx + w / 2) * W2; y2 = (cy + h / 2) * H2
        col = COLORS.get(name, COLORS["other"])
        dr.rectangle([x1, y1, x2, y2], outline=col, width=lw)
        dr.rectangle([x1, y1, x2, y2], fill=col + (40,))
        tag = name
        tb = dr.textbbox((0, 0), tag, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, y1 - th - 4)
        dr.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=col + (230,))
        dr.text((x1 + 3, ty + 2), tag, fill=(255, 255, 255), font=font)
    if title:
        tb = dr.textbbox((0, 0), title, font=font)
        dr.rectangle([0, 0, tb[2] + 10, tb[3] + 8], fill=(0, 0, 0, 200))
        dr.text((5, 3), title, fill=(255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--label", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--schema", default="ours", choices=["ours", "doclaynet"])
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    if args.schema == "ours":
        names = {0: "header", 1: "text-area", 2: "footnote", 3: "footer"}
    else:  # DocLayNet YOLO ids (category_id-1)
        names = {0: "caption", 1: "footnote", 2: "formula", 3: "list-item",
                 4: "page-footer", 5: "page-header", 6: "picture",
                 7: "section-header", 8: "table", 9: "text", 10: "title"}
    boxes = load_boxes(args.label, names)
    render(args.image, boxes, args.out, title=args.title)
    print(f"wrote {args.out} ({len(boxes)} boxes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
