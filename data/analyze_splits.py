#!/usr/bin/env python3
"""Audit the merged HFF dataset for split hygiene and leakage before release.

Checks, from strongest to weakest signal:
  1. Exact CONTENT duplicates (md5 of pixels) shared across splits.
  2. Same PAGE identity across splits (filename minus the __augNN suffix) --
     catches an original in one split and its augmentation in another.
  3. Same BOOK/VOLUME across splits (work / image-group key) -- catches
     book-level leakage where different pages of one book span train & test.
  4. Count of augmented (__aug) images per split (should be ~0 in val/test
     for a clean benchmark).
Also reports per-split image/label counts and per-class box counts.
"""
from __future__ import annotations

import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset")
SPLITS = ["train", "val", "test"]


def strip_name(fn: str) -> str:
    n = re.sub(r"^ds\d+__", "", fn)
    n = re.sub(r"\.[^.]+$", "", n)
    return n


def is_aug(base: str) -> bool:
    return bool(re.search(r"__aug\d*$", base)) or base.endswith("aug")


def page_id(base: str) -> str:
    """Identity of the underlying page, ignoring augmentation index."""
    return re.sub(r"__aug\d*$", "", base)


def book_key(base: str) -> str:
    """Best-effort book / volume / image-group identifier."""
    b = page_id(base)
    m = re.search(r"__p\d+$", b)
    if m:  # verbose style: IE..__VE..__TITLE__p#### -> everything before page
        return b[: m.start()]
    parts = b.split("__")
    if len(parts) >= 2 and parts[0][:1] == "W":  # ds2: W<work>__I<grp>__I<grp>####
        return parts[0]
    # BDRC single token e.g. I00KG023480023 or 42170014: strip trailing 4 = page
    m = re.match(r"^(.*?)(\d{3,4})$", b)
    return m.group(1) if m else b


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    names = {s: sorted(p.name for p in (ROOT / "images" / s).iterdir()) for s in SPLITS}

    print("=" * 70)
    print("SPLIT SIZES")
    for s in SPLITS:
        nlab = len(list((ROOT / "labels" / s).glob("*.txt")))
        naug = sum(is_aug(strip_name(n)) for n in names[s])
        print(f"  {s:5s}: {len(names[s]):5d} images | {nlab:5d} labels | {naug:5d} augmented")

    # ---- per-class box counts ----
    print("=" * 70)
    print("CLASS DISTRIBUTION (boxes per split)")
    cls_names = {0: "header", 1: "text-area", 2: "footnote", 3: "footer"}
    for s in SPLITS:
        c = Counter()
        empty = 0
        for lp in (ROOT / "labels" / s).glob("*.txt"):
            lines = [ln for ln in lp.read_text().splitlines() if ln.strip()]
            if not lines:
                empty += 1
            for ln in lines:
                c[int(ln.split()[0])] += 1
        dist = "  ".join(f"{cls_names.get(k, k)}={c.get(k,0)}" for k in range(4))
        print(f"  {s:5s}: {dist}   (empty-label files: {empty})")

    # ---- build per-split keysets ----
    md5map = {s: {} for s in SPLITS}  # hash -> [names]
    pagemap = {s: defaultdict(list) for s in SPLITS}
    bookmap = {s: set() for s in SPLITS}
    for s in SPLITS:
        for n in names[s]:
            base = strip_name(n)
            pagemap[s][page_id(base)].append(n)
            bookmap[s].add(book_key(base))
            p = (ROOT / "images" / s / n).resolve()
            try:
                md5map[s][md5(p)] = md5map[s].get(md5(p), []) + [n]
            except OSError as e:
                print(f"  !! cannot read {n}: {e}")

    def cross(a, b, getset):
        return getset(a) & getset(b)

    print("=" * 70)
    print("1) EXACT CONTENT DUPLICATES ACROSS SPLITS (md5)")
    any_c = False
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        inter = set(md5map[a]) & set(md5map[b])
        if inter:
            any_c = True
            print(f"  !! {a} <-> {b}: {len(inter)} identical images")
            for h in list(inter)[:8]:
                print(f"       {md5map[a][h][0]}  ==  {md5map[b][h][0]}")
    if not any_c:
        print("  OK - no pixel-identical images shared across splits")

    print("=" * 70)
    print("2) SAME PAGE IDENTITY ACROSS SPLITS (original vs augmentation)")
    any_p = False
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        inter = set(pagemap[a]) & set(pagemap[b])
        if inter:
            any_p = True
            print(f"  !! {a} <-> {b}: {len(inter)} shared page identities")
            for k in list(inter)[:8]:
                print(f"       {k}:  {a}={pagemap[a][k]}  {b}={pagemap[b][k]}")
    if not any_p:
        print("  OK - no page appears (in any aug form) in more than one split")

    print("=" * 70)
    print("3) SAME BOOK / VOLUME ACROSS SPLITS (book-level leakage)")
    any_b = False
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        inter = bookmap[a] & bookmap[b]
        if inter:
            any_b = True
            print(f"  !! {a} <-> {b}: {len(inter)} shared book/volume keys")
            for k in sorted(inter)[:15]:
                print(f"       {k}")
    if not any_b:
        print("  OK - no book/volume shared across splits")
    print(
        f"\n  (#distinct books: train={len(bookmap['train'])}, "
        f"val={len(bookmap['val'])}, test={len(bookmap['test'])})"
    )


if __name__ == "__main__":
    main()
