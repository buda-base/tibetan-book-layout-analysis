#!/usr/bin/env python3
"""Confidence sweep for precomputed YOLO label predictions in canonical 3-class space.

Writes a sweep table (like surya_sweep.txt) and the best mean-F1 row.

Usage:
  python canon_sweep_preds.py <pred_labels_dir> <dataset_dir> <out_sweep.txt>
                              [remap] [iou]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

GRID = [round(i * 0.05, 2) for i in range(21)]  # 0.00 .. 1.00


def main() -> int:
    pred_dir = Path(sys.argv[1])
    dataset = sys.argv[2]
    out_sweep = Path(sys.argv[3])
    remap = sys.argv[4] if len(sys.argv) > 4 else "0:0,1:1,2:2,3:0"
    iou = sys.argv[5] if len(sys.argv) > 5 else "0.5"
    script = Path(__file__).with_name("eval_pred_files.py")

    lines = [
        "canonical confidence sweep (header-footer, text-area merged, footnote)",
        f"pred={pred_dir} dataset={dataset} remap={remap} iou>={iou}",
        f"{'conf':<6} {'hf_F1':<8} {'ta_F1':<8} {'fn_F1':<8} {'meanF1':<8}",
    ]
    best = (-1.0, 0.0, {})
    for conf in GRID:
        proc = subprocess.run(
            [sys.executable, str(script), str(pred_dir), dataset, remap, iou, str(conf)],
            capture_output=True, text=True, check=True,
        )
        metrics = {}
        for ln in proc.stdout.splitlines():
            parts = ln.split()
            if parts and parts[0] in ("header-footer", "text-area", "footnote"):
                metrics[parts[0]] = float(parts[3])
            elif "mean F1" in ln:
                metrics["mean"] = float(parts[-1])
        hf = metrics.get("header-footer", 0.0)
        ta = metrics.get("text-area", 0.0)
        fn = metrics.get("footnote", 0.0)
        mean = metrics.get("mean", (hf + ta + fn) / 3)
        lines.append(f"{conf:<6.2f} {hf:<8.3f} {ta:<8.3f} {fn:<8.3f} {mean:<8.3f}")
        if mean > best[0]:
            best = (mean, conf, metrics)

    out_sweep.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_sweep}")
    print(f"best mean F1 {best[0]:.3f} @ conf {best[1]:.2f}")
    for k, v in best[2].items():
        print(f"  {k}: {v:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
