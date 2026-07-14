#!/usr/bin/env python3
"""Convert Ultralytics-Platform NDJSON export(s) into YOLO dataset folders.

The Platform "download" only yields an NDJSON file whose image URLs are signed
and valid for 7 days. ``convert_ndjson_to_yolo`` downloads the pixels and writes
``images/{split}/`` + ``labels/{split}/`` + ``data.yaml`` (splits come from the
per-image ``split`` field in the NDJSON). Re-export if the URLs have expired.

Usage:
    python convert_ndjson.py a.ndjson b.ndjson --out converted
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from ultralytics.data.converter import convert_ndjson_to_yolo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ndjson",
        nargs="+",
        help="NDJSON export file(s), or ul://user/datasets/slug URIs "
        "(needs ULTRALYTICS_API_KEY for URIs)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("converted"),
        help="Output directory for the converted YOLO folders",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for nd in args.ndjson:
        is_uri = str(nd).startswith("ul://")
        if not is_uri and not Path(nd).is_file():
            raise FileNotFoundError(f"NDJSON not found: {nd}")
        print(f"Converting {nd} -> {args.out}/ ...", flush=True)
        yaml_path = asyncio.run(convert_ndjson_to_yolo(str(nd), str(args.out)))
        print(f"  data.yaml: {yaml_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
