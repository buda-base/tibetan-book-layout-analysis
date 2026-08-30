#!/usr/bin/env python3
"""G4 — raw per-split per-class box counts for the v4 dataset (Table 1).

Counts boxes from the RAW v4 YOLO annotations (Hub tag `v4`, 4 native classes
0 header / 1 text-area / 2 footnote / 3 footer) — NOT the tam2col-merged
curriculum labels (where text-area is collapsed to one page/column envelope).

Each split's image set is taken from the split list (`data/splits/v4/<split>.txt`);
every listed stem's label is read from `<labels-root>/<split>/<stem>.txt`. Prints
the composition table and writes a JSON blob.

Usage:
  python evaluation/v4_composition.py \\
      --labels-root /path/to/TDLA@v4/labels \\
      --splits-dir  data/splits/v4 \\
      --out evaluation/eval_results/tdla-v4/composition_v4.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CLASSES = {0: "header", 1: "text-area", 2: "footnote", 3: "footer"}
SPLITS = ("train", "val", "test")


def stem_of(line: str) -> str:
    return Path(line.strip()).stem


def count_split(labels_root: Path, split: str, split_list: Path):
    counts = {name: 0 for name in CLASSES.values()}
    n_images = 0
    n_missing = 0
    for line in split_list.read_text().splitlines():
        if not line.strip():
            continue
        stem = stem_of(line)
        lp = labels_root / split / f"{stem}.txt"
        if not lp.exists():
            n_missing += 1
            continue
        n_images += 1
        for ln in lp.read_text().splitlines():
            p = ln.split()
            if len(p) < 5:
                continue
            cid = int(p[0])
            name = CLASSES.get(cid)
            if name is not None:
                counts[name] += 1
    counts["total"] = sum(counts[n] for n in CLASSES.values())
    return {"n_images": n_images, "n_missing_label": n_missing, "boxes": counts}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", type=Path, required=True,
                    help="TDLA@v4 labels dir containing train/val/test subdirs")
    ap.add_argument("--splits-dir", type=Path, required=True,
                    help="dir with train.txt/val.txt/test.txt image lists")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    per_split = {}
    for s in SPLITS:
        per_split[s] = count_split(args.labels_root, s, args.splits_dir / f"{s}.txt")

    corpus = {name: sum(per_split[s]["boxes"][name] for s in SPLITS)
              for name in list(CLASSES.values()) + ["total"]}
    corpus_images = sum(per_split[s]["n_images"] for s in SPLITS)

    blob = {
        "dataset_tag": "v4",
        "classes": list(CLASSES.values()),
        "per_split": per_split,
        "corpus": {"n_images": corpus_images, "boxes": corpus},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blob, indent=2))

    order = ["header", "footer", "text-area", "footnote", "total"]
    w = 12
    print(f"{'split':<8}{'images':>8}" + "".join(f"{c:>{w}}" for c in order))
    for s in SPLITS:
        b = per_split[s]["boxes"]
        print(f"{s:<8}{per_split[s]['n_images']:>8}"
              + "".join(f"{b[c]:>{w}}" for c in order))
    print(f"{'CORPUS':<8}{corpus_images:>8}"
          + "".join(f"{corpus[c]:>{w}}" for c in order))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
