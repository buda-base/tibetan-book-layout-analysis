# tdla-v4 — COTe + Hidden-Trespass evaluation

Area-based failure analysis on the leak-free **v4 833-page test** (`/home/eroux/tmp/dataset_tdlav4_tam2col/labels/test`). Same metric code as the v2 literature note (`literature_metrics.py`); this driver runs it over the v4 system set and reports each system at its **own best-mean-F1 operating point**. Predictions are the archived YOLO dumps under `s3://.../tdlav4/eval/...`; nothing was re-inferred here.

- Canonical 3-class space: header+footer combined (matched individually), text-area merged to one page/column envelope, footnote as-is; IoU≥0.5.
- COTe via `cotescore` 0.2.0; the text-area envelope is the body SSU.
- Scoring wall-clock: **418.0s** CPU.

## Headline — canonical COCO mAP + F1 @ best-mean point

| system | mean F1 | conf | hf F1 | ta F1 | fn F1 | mean AP50 | mean AP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RT-DETR-l tam2col (ours, v4; seed0) | 0.957 | 0.74 | 0.952 | 0.994 | 0.925 | 0.974 | 0.786 |
| RF-DETR-Large tam2col (ours, v4) | 0.913 | 0.26 | 0.949 | 0.954 | 0.835 | 0.925 | 0.667 |
| Docling layout-heron tam2col (ours, v4) | 0.925 | 0.08 | 0.944 | 0.998 | 0.833 | 0.912 | 0.735 |
| DocLayout-YOLO tam2col (ours, v4) | 0.896 | 0.29 | 0.931 | 0.978 | 0.779 | 0.919 | 0.720 |
| PP-DocLayout-L tam2col (ours, v4) | 0.958 | 0.68 | 0.951 | 0.998 | 0.925 | 0.959 | 0.781 |
| PP-DocLayout-L (off-the-shelf) | 0.672 | 0.33 | 0.458 | 0.872 | 0.688 | 0.580 | 0.353 |
| Docling layout-heron (off-the-shelf) | 0.573 | 0.58 | 0.488 | 0.938 | 0.292 | 0.472 | 0.266 |
| DocLayout-YOLO DocStructBench (off-the-shelf) | 0.508 | 0.17 | 0.616 | 0.907 | 0.000 | 0.442 | 0.281 |
| Surya 2 layout VLM (surya-ocr-2) | 0.777 | 0.00 | 0.865 | 0.993 | 0.474 | 0.656 | 0.404 |
| Chandra 2 (chandra-ocr-2) | 0.593 | 0.00 | 0.699 | 0.695 | 0.385 | 0.411 | 0.231 |
| Azure DI prebuilt-layout | 0.618 | 0.00 | 0.611 | 0.990 | 0.252 | 0.489 | 0.331 |
| AWS Textract Layout | 0.356 | 0.00 | 0.232 | 0.835 | 0.000 | 0.263 | 0.144 |
| Google DocAI Layout Parser | 0.373 | 0.00 | 0.151 | 0.967 | 0.000 | 0.325 | 0.231 |

## Hidden Trespass (area-based text-area → clutter bleed)

Primary metric. For each class *c* ∈ {header-footer, footnote}, micro-averaged
over the test set (Σ area over pages):

- **HT_c** = area(E ∩ U_c) / area(G_c) — *hidden* bleed: class-*c* GT area that
  is undetected (no same-class pred at IoU≥0.5) yet sits inside the predicted
  text-area envelope *E* (the OCR crop). Continuous overlap, no 50% cut-off.
- **R_c** = area(E ∩ D_c) / area(G_c) — *removed-before-OCR*: detected class-*c*
  GT also inside *E*; a post-processor can punch it back out.
- **total_c** = HT_c + R_c = area(E ∩ G_c) / area(G_c).

Count-based **contam.** (undetected AND ≥50% absorbed, fraction of *regions*) is
the secondary intuition column.

| system | hf HT | hf R | hf total | hf contam. (count) | fn HT | fn R | fn total | fn contam. (count) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RT-DETR-l tam2col (ours, v4; seed0) | 0.008 | 0.002 | 0.010 | 0.3% | 0.037 | 0.055 | 0.092 | 2.6% |
| RF-DETR-Large tam2col (ours, v4) | 0.020 | 0.002 | 0.022 | 1.1% | 0.216 | 0.010 | 0.226 | 13.2% |
| Docling layout-heron tam2col (ours, v4) | 0.002 | 0.003 | 0.005 | 0.1% | 0.074 | 0.000 | 0.074 | 5.3% |
| DocLayout-YOLO tam2col (ours, v4) | 0.004 | 0.002 | 0.005 | 0.2% | 0.106 | 0.000 | 0.106 | 7.9% |
| PP-DocLayout-L tam2col (ours, v4) | 0.003 | 0.002 | 0.005 | 0.3% | 0.037 | 0.000 | 0.037 | 2.6% |
| PP-DocLayout-L (off-the-shelf) | 0.171 | 0.062 | 0.233 | 14.6% | 0.277 | 0.069 | 0.346 | 39.5% |
| Docling layout-heron (off-the-shelf) | 0.078 | 0.133 | 0.211 | 3.9% | 0.090 | 0.031 | 0.122 | 7.9% |
| DocLayout-YOLO DocStructBench (off-the-shelf) | 0.120 | 0.075 | 0.195 | 6.0% | 0.207 | 0.000 | 0.207 | 23.7% |
| Surya 2 layout VLM (surya-ocr-2) | 0.020 | 0.001 | 0.022 | 0.8% | 0.135 | 0.000 | 0.135 | 10.5% |
| Chandra 2 (chandra-ocr-2) | 0.019 | 0.001 | 0.019 | 0.8% | 0.020 | 0.000 | 0.020 | 2.6% |
| Azure DI prebuilt-layout | 0.181 | 0.005 | 0.186 | 9.8% | 0.184 | 0.000 | 0.184 | 21.1% |
| AWS Textract Layout | 0.160 | 0.009 | 0.168 | 16.6% | 0.644 | 0.000 | 0.644 | 55.3% |
| Google DocAI Layout Parser | 0.438 | 0.005 | 0.443 | 65.3% | 0.929 | 0.000 | 0.929 | 97.4% |

## COTe decomposition + LED

| system | COTe | Coverage | Overlap | **Trespass** | Excess | ta→peri Trespass | LED-Merge hf | LED-Missing hf | LED-Merge fn | LED-Missing fn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RT-DETR-l tam2col (ours, v4; seed0) | 0.975 | 0.976 | 0.000 | 0.001 | 0.022 | 0.001 | 10 | 55 | 0 | 1 |
| RF-DETR-Large tam2col (ours, v4) | 0.974 | 0.977 | 0.001 | 0.002 | 0.023 | 0.003 | 22 | 37 | 0 | 4 |
| Docling layout-heron tam2col (ours, v4) | 0.979 | 0.981 | 0.001 | 0.001 | 0.025 | 0.002 | 11 | 19 | 0 | 3 |
| DocLayout-YOLO tam2col (ours, v4) | 0.942 | 0.944 | 0.001 | 0.001 | 0.022 | 0.003 | 10 | 52 | 0 | 8 |
| PP-DocLayout-L tam2col (ours, v4) | 0.978 | 0.978 | 0.000 | 0.000 | 0.026 | 0.001 | 15 | 35 | 0 | 1 |
| PP-DocLayout-L (off-the-shelf) | 0.732 | 0.740 | 0.001 | 0.006 | 0.030 | 0.010 | 3 | 373 | 0 | 16 |
| Docling layout-heron (off-the-shelf) | 0.902 | 0.911 | 0.005 | 0.005 | 0.017 | 0.006 | 1 | 148 | 0 | 2 |
| DocLayout-YOLO DocStructBench (off-the-shelf) | 0.772 | 0.780 | 0.004 | 0.005 | 0.035 | 0.007 | 54 | 162 | 0 | 38 |
| Surya 2 layout VLM (surya-ocr-2) | 0.921 | 0.922 | 0.000 | 0.001 | 0.006 | 0.003 | 2 | 56 | 0 | 5 |
| Chandra 2 (chandra-ocr-2) | 0.619 | 0.620 | 0.000 | 0.000 | 0.003 | 0.001 | 2 | 572 | 0 | 27 |
| Azure DI prebuilt-layout | 0.914 | 0.919 | 0.000 | 0.005 | 0.015 | 0.007 | 11 | 236 | 0 | 11 |
| AWS Textract Layout | 0.714 | 0.721 | 0.000 | 0.006 | 0.031 | 0.010 | 1 | 746 | 0 | 38 |
| Google DocAI Layout Parser | 0.911 | 0.925 | 0.000 | 0.014 | 0.114 | 0.015 | 14 | 1029 | 0 | 38 |

## Cross-check (Spearman across all systems)

Area-based Hidden Trespass vs library metrics, 13 systems: ρ(HT, COTe-Trespass text-area→peripheral) = **0.984**; ρ(HT, LED-Merge) = **-0.071**.

Legacy count-based contamination: ρ(contamination, COTe-Trespass) = **0.907**; ρ(contamination, LED-Merge) = **-0.033**.

