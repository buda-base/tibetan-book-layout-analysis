#!/usr/bin/env python3
"""Score the multi-seed RT-DETR-l tam2col retraining for training-seed variance.

Each seed's test-set prediction dump (produced by seed_run.py on the GPU box,
same YOLO format + conf 0.05 floor as the archived tam2col_pred) is scored on
the 860-page test set with the canonical schema, at the paper operating
confidence (0.50). We report, across seeds:

  * canonical mean-F1  (mean of header-footer / text-area-envelope / footnote F1)
  * each canonical class F1
  * canonical text-area AP50-95 (COCO)

and the seed-to-seed mean / std / min / max -- the missing "noise floor" for the
0.93-0.96 architecture band. This is TRAINING-seed variance (each dump is an
independent full training run), complementary to the test-set bootstrap in
bootstrap_variance.py.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from literature_metrics import (  # noqa: E402
    _pr_class, apply_schema, coco_map, iter_pages, load_sizes,
)

CANON = ["header-footer", "text-area", "footnote"]


def score_dump(gt_dir, pred_dir, sizes, conf):
    pages = list(iter_pages(gt_dir, pred_dir, sizes, conf_floor=0.0))
    # canonical class F1 at the operating conf (aggregate over pages)
    agg = {n: [0, 0, 0] for n in CANON}
    for stem, W, H, g, p in pages:
        gg = apply_schema(g, "canonical", W, H)
        pp = apply_schema(p, "canonical", W, H)
        for n in CANON:
            gt_list = {stem: [xy for xy, _ in gg[n]]} if gg[n] else {}
            pred_list = {stem: pp[n]} if pp[n] else {}
            r = _pr_class(gt_list, pred_list, conf)
            agg[n][0] += r["tp"]; agg[n][1] += r["fp"]; agg[n][2] += r["fn"]
    f1 = {}
    for n in CANON:
        tp, fp, fn = agg[n]
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        f1[n] = 2 * P * R / (P + R) if P + R else 0.0
    mean_f1 = sum(f1.values()) / len(CANON)
    # canonical COCO AP, mean over all canonical classes + text-area alone
    cm = coco_map(pages, "canonical")
    ta = cm["per_class"].get("text-area", {})
    return {
        "n_pages": len(pages),
        "mean_F1": mean_f1,
        "F1": f1,
        "canonical_mean_AP50_95": cm["mean_ap5095"],
        "canonical_mean_AP50": cm["mean_ap50"],
        "text_area_AP50_95": ta.get("ap5095", float("nan")),
        "text_area_AP50": ta.get("ap50", float("nan")),
    }


def summarize(vals):
    return {
        "mean": st.mean(vals),
        "std": st.pstdev(vals) if len(vals) > 1 else 0.0,
        "sample_std": st.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
        "values": vals,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path,
                    default=Path("/home/eroux/azure_di_eval/testset/labels/test"))
    ap.add_argument("--img-dir", type=Path,
                    default=Path("/home/eroux/azure_di_eval/testset/images/test"))
    ap.add_argument("--dumps-root", type=Path,
                    default=Path("/home/eroux/azure_di_eval/seed_variance"),
                    help="dir containing seed0/labels, seed1/labels, ...")
    ap.add_argument("--out", type=Path, default=Path(
        "/home/eroux/BUDA/softs/tibetan-book-layout-analysis/evaluation/"
        "eval_results/literature/seed_variance.json"))
    ap.add_argument("--conf", type=float, default=0.50)
    args = ap.parse_args()

    sizes = load_sizes(args.img_dir, cache=args.out.parent / "test_image_sizes.json")
    seed_dirs = sorted(args.dumps_root.glob("seed*/labels"))
    if not seed_dirs:
        print(f"no seed dumps under {args.dumps_root}", file=sys.stderr)
        return 2

    per_seed = {}
    for sd in seed_dirs:
        seed = sd.parent.name  # seed0, seed1, ...
        r = score_dump(args.gt_dir, sd, sizes, args.conf)
        per_seed[seed] = r
        print(f"{seed}: mean_F1={r['mean_F1']:.4f}  "
              f"hf={r['F1']['header-footer']:.4f} "
              f"ta={r['F1']['text-area']:.4f} "
              f"fn={r['F1']['footnote']:.4f}  "
              f"canon_mAP50-95={r['canonical_mean_AP50_95']:.4f}  "
              f"ta_AP50-95={r['text_area_AP50_95']:.4f}  "
              f"({r['n_pages']} pages)", flush=True)

    mean_f1_vals = [per_seed[s]["mean_F1"] for s in per_seed]
    canon_ap_vals = [per_seed[s]["canonical_mean_AP50_95"] for s in per_seed]
    ta_ap_vals = [per_seed[s]["text_area_AP50_95"] for s in per_seed]
    hf_vals = [per_seed[s]["F1"]["header-footer"] for s in per_seed]
    ta_f1_vals = [per_seed[s]["F1"]["text-area"] for s in per_seed]
    fn_vals = [per_seed[s]["F1"]["footnote"] for s in per_seed]

    summary = {
        "canonical_mean_F1": summarize(mean_f1_vals),
        "header_footer_F1": summarize(hf_vals),
        "text_area_F1": summarize(ta_f1_vals),
        "footnote_F1": summarize(fn_vals),
        "canonical_mean_AP50_95": summarize(canon_ap_vals),
        "text_area_AP50_95": summarize(ta_ap_vals),
    }
    out = {
        "method": ("training-seed variance: independent RT-DETR-l tam2col runs "
                   "(seeds 0..N), scored on the 860-page test set, canonical "
                   "schema, operating conf 0.50"),
        "conf": args.conf,
        "recipe": ("rtdetr-l.pt, imgsz 1024, epochs 100, patience 20, batch 8, "
                   "deterministic=True; only the seed differs"),
        "per_seed": per_seed,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    print("\n=== training-seed variance summary ===", flush=True)
    for k, s in summary.items():
        print(f"  {k:20s} mean={s['mean']:.4f}  std={s['std']:.4f}  "
              f"[{s['min']:.4f}, {s['max']:.4f}]  n={s['n']}", flush=True)
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
