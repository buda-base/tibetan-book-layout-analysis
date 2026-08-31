# tdla-v4 — unified F1 / mAP scoring (one driver for all 13 systems)

The v4 canonical F1/mAP was previously produced by **two** drivers that disagreed
for two models. This note (a) diagnoses the divergence, (b) picks one canonical
path, and (c) re-emits F1 + canonical mAP for all 13 systems through it, with
Hidden-Trespass / COTe / LED computed at the **same** operating point as each
system's F1.

## The two drivers

- `canon_sweep_preds.py` → `eval_pred_files.py` (the old README "Ours" headline).
- `run_v4_lit_eval.py` → `literature_metrics.py` (the HT/COTe pass, `literature/`).

Both claim: canonical 3-class space (header+footer combined but matched
individually, text-area merged to one page/column envelope, footnote as-is);
pycocotools bbox mAP 0.50:0.05:0.95; F1 by greedy IoU≥0.5 at the best-mean-F1
operating point.

## Root cause (confirmed empirically)

**RF-DETR text-area: 0.996 (headline) vs 0.954 (literature) — envelope
construction.** The headline path drops predicted body boxes *below the operating
confidence before* forming the page text-area envelope, so the OCR-crop envelope
is built only from boxes the model keeps at that point. The literature path built
**one** envelope from *all* predicted text-area boxes and then gated the whole
envelope by its max confidence — so RF-DETR's spurious low-confidence text-area
boxes stayed in the envelope and inflated it, dropping IoU<0.5 on 35/833 pages
(tp 826→791, fp/fn 3→38). Reproduced exactly: per-conf envelope = **0.9964**,
fixed max-conf-gated envelope = **0.9542**. Greedy sort order is irrelevant here
(one box per page).

**heron footnote: 0.809 (headline) vs 0.833 (literature) — operating-point
grid.** Sort order (by area vs by confidence) gives *identical* F1. The headline
sweeps confidence in 0.05 steps and lands the global best-mean-F1 at conf
0.00/0.05 (footnote F1 0.809); the literature 0.01 grid lands it at conf 0.08
(footnote F1 0.833, fp 15→11, at negligible cost to header-footer / text-area).

Together these also flipped the RF-DETR↔heron ordering between the two tables.

## Canonical path chosen

`run_v4_lit_eval.py` + `literature_metrics.py`, **fixed** so the canonical
text-area envelope respects the operating confidence (rebuilt per-conf from
surviving boxes) on the fine 0.01 grid. Rationale:

- It matches the intended protocol: predictions below the operating point are not
  predictions, so they must not contribute to the OCR-crop envelope. This
  reproduces the headline's protocol-correct text-area F1 (RF-DETR 0.996).
- Hidden-Trespass / contamination / COTe / LED **already** build the envelope
  from conf-gated boxes, so F1 and the area-based metrics are now consistent —
  one operating point per system for every number.
- Canonical COCO mAP was already single-source (`coco_map`); unchanged.

Code change: `apply_schema(..., conf_floor)` drops sub-threshold boxes before
grouping/enveloping; `best_f1_sweep` / `operating_points` re-materialise
predictions per confidence (GT is confidence-independent, materialised once).

## Unified per-system table (v4 833-page test, best-mean-F1 point)

| system | mean F1 | conf | hf F1 | ta F1 | fn F1 | mean AP50 | mean AP50-95 | shared mAP@50-95 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| RT-DETR-l tam2col (ours, seed0) | 0.959 | 0.74 | 0.952 | 0.999 | 0.925 | 0.974 | 0.786 | 0.650 |
| PP-DocLayout-L (ours) | 0.958 | 0.68 | 0.951 | 0.997 | 0.925 | 0.959 | 0.781 | 0.641 |
| RF-DETR-Large (ours) | 0.927 | 0.26 | 0.949 | 0.996 | 0.835 | 0.925 | 0.667 | 0.604 |
| Docling layout-heron (ours) | 0.925 | 0.08 | 0.944 | 0.998 | 0.833 | 0.912 | 0.735 | 0.593 |
| DocLayout-YOLO (ours) | 0.897 | 0.29 | 0.931 | 0.980 | 0.779 | 0.919 | 0.720 | 0.586 |
| Surya 2 VLM | 0.777 | 0.00 | 0.865 | 0.993 | 0.474 | 0.656 | 0.404 | 0.200 |
| PP-DocLayout-L (OTS) | 0.670 | 0.30 | 0.464 | 0.869 | 0.677 | 0.580 | 0.353 | 0.169 |
| Azure DI prebuilt-layout | 0.618 | 0.00 | 0.611 | 0.990 | 0.252 | 0.489 | 0.331 | 0.103 |
| Chandra 2 | 0.593 | 0.00 | 0.699 | 0.695 | 0.385 | 0.411 | 0.231 | 0.152 |
| Docling layout-heron (OTS) | 0.590 | 0.58 | 0.488 | 0.989 | 0.292 | 0.472 | 0.266 | 0.075 |
| DocLayout-YOLO DocStructBench (OTS) | 0.503 | 0.08 | 0.605 | 0.904 | 0.000 | 0.442 | 0.281 | 0.031 |
| Google DocAI Layout Parser | 0.373 | 0.00 | 0.151 | 0.967 | 0.000 | 0.325 | 0.231 | 0.006 |
| AWS Textract Layout | 0.356 | 0.00 | 0.232 | 0.835 | 0.000 | 0.263 | 0.144 | 0.004 |

Full per-system JSON + HT/COTe/LED: [`literature/RESULTS.md`](literature/RESULTS.md),
`literature/metrics.json`. Shared mAP from `shared_map/`.

## What changed vs the previous README "Ours" table

| model | before | after | why |
|---|--:|--:|---|
| Docling layout-heron | 0.917 (fn 0.809 @0.00) | **0.925** (fn 0.833 @0.08) | fine 0.01 operating-point grid |
| RF-DETR-Large | 0.926 @0.25 | 0.927 @0.26 | fine grid (per-class F1 unchanged; text-area already 0.996 in the headline) |
| PP-DocLayout-L | 0.957 @0.70 | 0.958 @0.68 | fine grid |
| RT-DETR-l | 0.944 (5-seed *mean*) | **0.959** (seed0, unified single run) | single F1 per system; the 5-seed band stays as the variance reference |
| DocLayout-YOLO | 0.897 | 0.897 | unchanged |
| heron (OTS) | 0.588 | 0.590 | fine grid (fn 0.282→0.292) |
| DocLayout-YOLO DocStruct (OTS) | 0.502 @0.15 | 0.503 @0.08 | fine grid |

Commercial (Azure/Google/Textract), Surya, Chandra, PP-OTS: unchanged (single
un-thresholdable point or grid-insensitive).

The headline table and the HT/COTe pass now report a single F1 per system.
