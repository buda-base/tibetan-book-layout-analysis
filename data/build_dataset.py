#!/usr/bin/env python3
"""Merge the two converted YOLO datasets into one training dataset.

Rules (per the project brief):
  * The final **test** split is dataset-1's ``test`` split ONLY (never expanded).
  * The final **val** split is dataset-1's ``val`` split.
  * The final **train** split is dataset-1 ``train`` + ALL of dataset-2
    (dataset-2's own splits are ignored).
  * Class indices from both datasets are remapped to the canonical order
    ``0:header, 1:text-area, 2:footnote, 3:footer`` (matches
    ``HFF-Remover/example_evaluate.py``).
  * Leakage guard: any train image whose page-id (filename stem) or exact byte
    hash matches a test image is dropped from train.

Images are symlinked (saves disk); label files are rewritten with remapped
class ids. Filenames are prefixed with a per-source tag to avoid collisions.

Usage:
    python build_dataset.py --ds1 converted/tdla-*/data.yaml \
                            --ds2 converted/tdlabatch3-*/data.yaml --out dataset
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Canonical class order for the whole project.
CANONICAL: Dict[str, int] = {
    "header": 0,
    "text-area": 1,
    "footnote": 2,
    "footer": 3,
}

# Map many spellings/synonyms from the Platform onto the canonical labels.
SYNONYMS: Dict[str, str] = {
    "header": "header",
    "headers": "header",
    "head": "header",
    "text-area": "text-area",
    "textarea": "text-area",
    "text area": "text-area",
    "text_area": "text-area",
    "text": "text-area",
    "plain text": "text-area",
    "plain_text": "text-area",
    "body": "text-area",
    "footnote": "footnote",
    "footnotes": "footnote",
    "foot note": "footnote",
    "footer": "footer",
    "footers": "footer",
    "page footer": "footer",
    "page_footer": "footer",
    "page number": "footer",
    "page_number": "footer",
    "pagenumber": "footer",
}


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def load_names(data_yaml: Path) -> Dict[int, str]:
    """Return ``{class_id: name}`` from a YOLO data.yaml."""
    doc = yaml.safe_load(data_yaml.read_text())
    names = doc.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    raise ValueError(f"Could not read 'names' from {data_yaml}")


def build_remap(names: Dict[int, str], src: str) -> Dict[int, int]:
    """Map a dataset's raw class ids -> canonical ids via name synonyms."""
    remap: Dict[int, int] = {}
    for cid, raw in names.items():
        canon = SYNONYMS.get(_norm(raw))
        if canon is None:
            raise SystemExit(
                f"[{src}] class {cid} '{raw}' has no canonical mapping. "
                f"Add it to SYNONYMS. Known raw names: {names}"
            )
        remap[cid] = CANONICAL[canon]
    return remap


def dataset_root(data_yaml: Path) -> Path:
    """Resolve the dataset root that contains images/ and labels/."""
    doc = yaml.safe_load(data_yaml.read_text())
    path = doc.get("path")
    if path:
        root = Path(path)
        if not root.is_absolute():
            root = (data_yaml.parent / root).resolve()
    else:
        root = data_yaml.parent.resolve()
    return root


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_split_images(root: Path, split: str) -> List[Path]:
    img_dir = root / "images" / split
    if not img_dir.is_dir():
        return []
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def label_for(img: Path) -> Path:
    return img.parent.parent.parent / "labels" / img.parent.name / f"{img.stem}.txt"


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remap_label_text(text: str, remap: Dict[int, int]) -> str:
    out_lines: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cid = int(float(parts[0]))
        if cid not in remap:
            continue
        parts[0] = str(remap[cid])
        out_lines.append(" ".join(parts))
    return "\n".join(out_lines) + ("\n" if out_lines else "")


def write_pair(
    img: Path,
    remap: Dict[int, int],
    out_root: Path,
    split: str,
    tag: str,
    class_counter: Counter,
) -> None:
    """Symlink an image and write its remapped label into the output split."""
    dst_stem = f"{tag}__{img.stem}"
    dst_img = out_root / "images" / split / f"{dst_stem}{img.suffix.lower()}"
    dst_lbl = out_root / "labels" / split / f"{dst_stem}.txt"
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)

    if dst_img.exists() or dst_img.is_symlink():
        dst_img.unlink()
    dst_img.symlink_to(img.resolve())

    src_lbl = label_for(img)
    text = src_lbl.read_text() if src_lbl.is_file() else ""
    remapped = remap_label_text(text, remap)
    dst_lbl.write_text(remapped)
    for line in remapped.splitlines():
        class_counter[int(line.split()[0])] += 1


def add_split(
    images: List[Path],
    remap: Dict[int, int],
    out_root: Path,
    split: str,
    tag: str,
    class_counter: Counter,
    skip_stems: Optional[set] = None,
    skip_hashes: Optional[set] = None,
) -> Tuple[int, int]:
    """Add images to a split. Returns (added, skipped_for_leakage)."""
    added = skipped = 0
    for img in images:
        if skip_stems is not None and img.stem in skip_stems:
            skipped += 1
            continue
        if skip_hashes is not None and file_hash(img) in skip_hashes:
            skipped += 1
            continue
        write_pair(img, remap, out_root, split, tag, class_counter)
        added += 1
    return added, skipped


def resolve_yaml(pattern: str) -> Path:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise SystemExit(f"No data.yaml matched: {pattern}")
    if len(matches) > 1:
        raise SystemExit(f"Pattern matched multiple files, be more specific: {matches}")
    return Path(matches[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds1", required=True, help="data.yaml of dataset-1 (has test split)")
    parser.add_argument("--ds2", required=True, help="data.yaml of dataset-2 (all -> train)")
    parser.add_argument("--out", type=Path, default=Path("dataset"))
    args = parser.parse_args()

    ds1_yaml = resolve_yaml(args.ds1)
    ds2_yaml = resolve_yaml(args.ds2)
    ds1_root = dataset_root(ds1_yaml)
    ds2_root = dataset_root(ds2_yaml)

    remap1 = build_remap(load_names(ds1_yaml), "ds1")
    remap2 = build_remap(load_names(ds2_yaml), "ds2")
    print(f"ds1 root: {ds1_root}\n  class remap (raw->canon): {remap1}")
    print(f"ds2 root: {ds2_root}\n  class remap (raw->canon): {remap2}")

    out = args.out.resolve()
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts: Dict[str, Counter] = {s: Counter() for s in ("train", "val", "test")}

    # --- Test split: dataset-1 test ONLY ---
    ds1_test = list_split_images(ds1_root, "test")
    add_split(ds1_test, remap1, out, "test", "ds1", counts["test"])
    test_stems = {p.stem for p in ds1_test}
    test_hashes = {file_hash(p) for p in ds1_test}
    print(f"\ntest: {len(ds1_test)} images (dataset-1 test)")

    # --- Val split: dataset-1 val ---
    ds1_val = list_split_images(ds1_root, "val")
    add_split(ds1_val, remap1, out, "val", "ds1", counts["val"])
    print(f"val:  {len(ds1_val)} images (dataset-1 val)")

    # --- Train split: dataset-1 train + ALL of dataset-2, minus leakage ---
    ds1_train = list_split_images(ds1_root, "train")
    a1, s1 = add_split(
        ds1_train, remap1, out, "train", "ds1", counts["train"],
        skip_stems=test_stems, skip_hashes=test_hashes,
    )
    ds2_all: List[Path] = []
    for split in ("train", "val", "test"):
        ds2_all += list_split_images(ds2_root, split)
    a2, s2 = add_split(
        ds2_all, remap2, out, "train", "ds2", counts["train"],
        skip_stems=test_stems, skip_hashes=test_hashes,
    )
    print(
        f"train: {a1 + a2} images "
        f"(ds1 {a1}/{len(ds1_train)}, ds2 {a2}/{len(ds2_all)}); "
        f"dropped for test-leakage: ds1 {s1}, ds2 {s2}"
    )

    # --- data.yaml ---
    id_to_name = {v: k for k, v in CANONICAL.items()}
    data = {
        "path": str(out),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: id_to_name[i] for i in sorted(id_to_name)},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    print("\nper-class box counts:")
    for split in ("train", "val", "test"):
        pretty = {id_to_name[c]: n for c, n in sorted(counts[split].items())}
        print(f"  {split:5s}: {pretty}")
    print(f"\nWrote {out}/data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
