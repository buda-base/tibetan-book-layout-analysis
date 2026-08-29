#!/usr/bin/env python3
"""Select + render representative candidate figures for the paper.

Three figure families (each produces several candidates to choose from):

  fig1_regions   : an annotated page showing our four regions (header,
                   text-area envelope, footnote, footer) from GT.
  fig2_contam    : contamination made concrete -- a GT header/footer that a
                   DocLayNet detector (docling layout-heron, off-the-shelf)
                   swallows into its single text-area envelope, so it would be
                   cropped and OCR'd as body. GT boxes outlined; the predicted
                   text-area envelope drawn as a translucent fill.
  fig3_domain    : same Tibetan page, our tight tam2col boxes (left) vs the
                   DocLayNet detector's loose / fragmented boxes (right).

Not part of the metric pipeline. Reads archived YOLO dumps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from literature_metrics import (  # noqa: E402
    covered_frac, envelope_xyxy, iou_xyxy, load_sizes, read_yolo, to_px,
)

GT = Path("/home/eroux/azure_di_eval/testset/labels/test")
IMG = Path("/home/eroux/azure_di_eval/testset/images/test")
OURS = Path("/home/eroux/azure_di_eval/tam2col_pred/labels")
HERON = Path("/home/eroux/azure_di_eval/docling_heron_ots_pred/docling_heron_pred/labels")
SURYA = Path("/home/eroux/azure_di_eval/surya_pred/labels")

NAMES = {0: "header", 1: "text-area", 2: "footnote", 3: "footer"}
COL = {
    "header": (220, 30, 30), "footer": (30, 90, 220),
    "footnote": (240, 140, 0), "text-area": (20, 160, 60), "other": (150, 150, 150),
}


def font(sz):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def load_page(img_dir, stem, max_dim=1500):
    im = Image.open(img_dir / f"{stem}.jpg").convert("RGB")
    W, H = im.size
    scale = min(1.0, max_dim / max(W, H))
    if scale < 1.0:
        im = im.resize((int(W * scale), int(H * scale)))
    return im, scale


def draw_box(dr, xyxy_px, scale, color, label=None, width=3, fill_alpha=0,
             f=None):
    x1, y1, x2, y2 = (v * scale for v in xyxy_px)
    if fill_alpha:
        dr.rectangle([x1, y1, x2, y2], fill=color + (fill_alpha,))
    dr.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label:
        tb = dr.textbbox((0, 0), label, font=f)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, y1 - th - 4)
        dr.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color + (235,))
        dr.text((x1 + 3, ty + 2), label, fill=(255, 255, 255), font=f)


def title_bar(im, text, f):
    dr = ImageDraw.Draw(im, "RGBA")
    tb = dr.textbbox((0, 0), text, font=f)
    dr.rectangle([0, 0, tb[2] + 12, tb[3] + 10], fill=(0, 0, 0, 205))
    dr.text((6, 4), text, fill=(255, 255, 255), font=f)
    return im


def boxes_px(label_path, W, H, conf_floor=0.0):
    out = []
    for b in read_yolo(label_path, conf_floor=conf_floor):
        out.append((b["cls"], to_px(b, W, H), b["conf"]))
    return out


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def stems_all_4(sizes):
    out = []
    for stem in sorted(p.stem for p in GT.glob("*.txt")):
        if stem not in sizes:
            continue
        cls = {b["cls"] for b in read_yolo(GT / f"{stem}.txt")}
        if {0, 1, 2, 3} <= cls:
            out.append(stem)
    return out


def stems_contam(sizes, pred_dir, want_cls=(0, 3), cover=0.7):
    """Pages where a GT header/footer is not detected by pred of its class but
    is >=cover covered by the predicted text-area envelope. Ranked by the
    absorbed box's area (bigger = more legible in a figure)."""
    cands = []
    for stem in sorted(p.stem for p in GT.glob("*.txt")):
        if stem not in sizes:
            continue
        W, H = sizes[stem]
        g = boxes_px(GT / f"{stem}.txt", W, H)
        p = boxes_px(pred_dir / f"{stem}.txt", W, H)
        ta = [xy for c, xy, _ in p if c == 1]
        env = envelope_xyxy(ta) if ta else None
        if env is None:
            continue
        for c, gb, _ in g:
            if c not in want_cls:
                continue
            same = [xy for cc, xy, _ in p if cc == c]
            det = any(iou_xyxy(gb, pb) >= 0.5 for pb in same)
            if not det and covered_frac(gb, env) >= cover:
                area = (gb[2] - gb[0]) * (gb[3] - gb[1]) / (W * H)
                cands.append((area, stem, NAMES[c], covered_frac(gb, env)))
    cands.sort(reverse=True)
    return cands


def stems_fragmented(sizes, pred_dir, min_ta=6):
    """Pages where the detector emits many text-area boxes (loose paragraph
    granularity) while our GT is a single envelope."""
    out = []
    for stem in sorted(p.stem for p in GT.glob("*.txt")):
        if stem not in sizes:
            continue
        W, H = sizes[stem]
        p = read_yolo(pred_dir / f"{stem}.txt")
        n_ta = sum(1 for b in p if b["cls"] == 1)
        g_ta = sum(1 for b in read_yolo(GT / f"{stem}.txt") if b["cls"] == 1)
        if n_ta >= min_ta:
            out.append((n_ta, g_ta, stem))
    out.sort(reverse=True)
    return out


# ---------------------------------------------------------------------------
# renderers
# ---------------------------------------------------------------------------

def render_regions(stem, sizes, out_path):
    W, H = sizes[stem]
    im, scale = load_page(IMG, stem)
    dr = ImageDraw.Draw(im, "RGBA")
    f = font(20)
    g = boxes_px(GT / f"{stem}.txt", W, H)
    # text-area as one envelope
    ta = [xy for c, xy, _ in g if c == 1]
    if ta:
        draw_box(dr, envelope_xyxy(ta), scale, COL["text-area"],
                 "text-area", width=4, fill_alpha=45, f=f)
    for c, xy, _ in g:
        if c == 1:
            continue
        draw_box(dr, xy, scale, COL[NAMES[c]], NAMES[c], width=3,
                 fill_alpha=55, f=f)
    title_bar(im, "Our four regions (ground truth)", f)
    im.save(out_path)
    return out_path


def render_contam(stem, sizes, pred_dir, out_path, absorbed_cls):
    W, H = sizes[stem]
    im, scale = load_page(IMG, stem)
    dr = ImageDraw.Draw(im, "RGBA")
    f = font(20)
    p = boxes_px(pred_dir / f"{stem}.txt", W, H)
    ta = [xy for c, xy, _ in p if c == 1]
    env = envelope_xyxy(ta) if ta else None
    if env:
        draw_box(dr, env, scale, COL["text-area"],
                 "detector text-area (\u2192 OCR)", width=4, fill_alpha=60, f=f)
    g = boxes_px(GT / f"{stem}.txt", W, H)
    for c, xy, _ in g:
        if c in (0, 3):
            lbl = NAMES[c] + (" (absorbed)" if NAMES[c] == absorbed_cls else "")
            draw_box(dr, xy, scale, COL[NAMES[c]], lbl, width=4, f=f)
    title_bar(im, "Contamination: running head swallowed by the body box", f)
    im.save(out_path)
    return out_path


def render_pred(stem, sizes, pred_dir, out_path, title, envelope_ta=False):
    W, H = sizes[stem]
    im, scale = load_page(IMG, stem)
    dr = ImageDraw.Draw(im, "RGBA")
    f = font(20)
    p = boxes_px(pred_dir / f"{stem}.txt", W, H)
    if envelope_ta:
        ta = [xy for c, xy, _ in p if c == 1]
        if ta:
            draw_box(dr, envelope_xyxy(ta), scale, COL["text-area"],
                     "text-area", width=4, fill_alpha=45, f=f)
        rest = [(c, xy) for c, xy, _ in p if c != 1]
    else:
        rest = [(c, xy) for c, xy, _ in p]
    for c, xy in rest:
        draw_box(dr, xy, scale, COL.get(NAMES.get(c, "other"), COL["other"]),
                 NAMES.get(c, "?"), width=3, fill_alpha=30, f=f)
    title_bar(im, title, f)
    im.save(out_path)
    return out_path


def side_by_side(left, right, out_path, pad=16):
    a = Image.open(left)
    b = Image.open(right)
    H = max(a.height, b.height)
    W = a.width + b.width + pad
    c = Image.new("RGB", (W, H), (255, 255, 255))
    c.paste(a, (0, 0))
    c.paste(b, (a.width + pad, 0))
    c.save(out_path)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/paper_fig_candidates"))
    ap.add_argument("--n", type=int, default=3, help="candidates per family")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sizes = load_sizes(IMG, cache=Path(
        "/home/eroux/BUDA/softs/tibetan-book-layout-analysis/evaluation/"
        "eval_results/literature/test_image_sizes.json"))
    print(f"{len(sizes)} sizes", flush=True)

    # fig1
    all4 = stems_all_4(sizes)
    print(f"\nfig1 4-region candidates: {len(all4)} pages have all 4 GT classes")
    for i, stem in enumerate(all4[:args.n]):
        render_regions(stem, sizes, args.out / f"fig1_regions_{i}_{stem}.png")
        print(f"  [{i}] {stem}")

    # fig2 contamination (heron swallows a header/footer)
    contam = stems_contam(sizes, HERON, want_cls=(0, 3), cover=0.7)
    print(f"\nfig2 contamination candidates (heron): {len(contam)}")
    for i, (area, stem, cls, cov) in enumerate(contam[:args.n]):
        render_contam(stem, sizes, HERON,
                      args.out / f"fig2_contam_{i}_{stem}.png", cls)
        print(f"  [{i}] {stem}  absorbed={cls} area={area:.3f} cover={cov:.2f}")

    # fig3 domain mismatch (same page: ours tight vs heron loose)
    frag = stems_fragmented(sizes, HERON, min_ta=6)
    print(f"\nfig3 domain-mismatch candidates (heron >=6 text boxes): {len(frag)}")
    for i, (n_ta, g_ta, stem) in enumerate(frag[:args.n]):
        left = render_pred(stem, sizes, OURS,
                           args.out / f"_tmp_ours_{stem}.png",
                           "Ours (RT-DETR-l tam2col)", envelope_ta=True)
        right = render_pred(stem, sizes, HERON,
                            args.out / f"_tmp_heron_{stem}.png",
                            "DocLayNet detector (layout-heron)", envelope_ta=False)
        side_by_side(left, right, args.out / f"fig3_domain_{i}_{stem}.png")
        print(f"  [{i}] {stem}  heron_text_boxes={n_ta} gt_text_boxes={g_ta}")

    print(f"\nwrote candidates to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
