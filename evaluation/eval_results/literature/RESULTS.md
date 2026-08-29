# Literature-protocol evaluation

Re-evaluation of stored test-set predictions under standard COCO mAP
(pycocotools, matching DocLayNet Table 2), plus COTe [@cote-score] and
LED [@led-benchmark] structural metrics. Predictions were **not** re-inferred;
this note scores the archived YOLO dumps. Domain-transfer cell
`ours → DocLayNet test` is filled from `metrics_ours_on_doclaynet.json`
when that file sits next to this note.

- Test set: **860** pages (`/home/eroux/azure_di_eval/testset/labels/test`).
- COCO: AP@0.50 and AP@0.50:0.95, area=all, maxDets=100, 101-point interpolation.
- Schema **(a) canonical**: header+footer combined (boxes matched individually),
  text-area merged to one envelope, footnote as-is.
- Schema **(b) DocLayNet-aligned**: page-header / page-footer separate,
  text-area native boxes (no envelope), footnote as-is.
- P/R/F1 reported at the paper's operating confidence (best mean-F1 for
  thresholdable systems; the only point for Azure). A per-class best-F1
  sweep is in the JSON.
- COTe uses the released `cotescore` library; the text-area envelope is the
  body Structural Semantic Unit. LED Missing/Merge/Split follow Heo et al.
- Wall-clock for this scoring pass: **295.9s** on CPU
  (`local-cpu`).

## Task 1 — COCO mAP on our 860-page test set

### Schema (a) — canonical 3-class

| system | hf AP50 | hf AP50-95 | ta AP50 | ta AP50-95 | fn AP50 | fn AP50-95 | mean AP50 | mean AP50-95 | mean F1 @paper conf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RT-DETR-l tam2col (ours) | 0.960 | 0.681 | 0.980 | 0.906 | 0.990 | 0.822 | 0.977 | 0.803 | 0.957 |
| RF-DETR-L tam2col (ours) | 0.973 | 0.732 | 0.969 | 0.797 | 0.972 | 0.815 | 0.971 | 0.781 | 0.953 |
| DocLayout-YOLO tam2col (ours) | 0.960 | 0.700 | 1.000 | 0.979 | 0.924 | 0.722 | 0.961 | 0.800 | 0.947 |
| PP-DocLayout-L tam2col | 0.973 | 0.726 | 0.970 | 0.901 | 0.929 | 0.795 | 0.957 | 0.807 | 0.950 |
| Docling layout-heron tam2col | 0.935 | 0.638 | 0.957 | 0.734 | 0.984 | 0.853 | 0.959 | 0.742 | 0.928 |
| Surya fast layout (RF-DETR, DocLayNet-style) | 0.884 | 0.282 | 0.977 | 0.743 | 0.327 | 0.164 | 0.729 | 0.396 | 0.771 |
| Surya 2 layout VLM (surya-ocr-2) | 0.774 | 0.271 | 0.972 | 0.853 | 0.220 | 0.104 | 0.655 | 0.409 | 0.776 |
| Chandra 2 (chandra-ocr-2) | 0.529 | 0.180 | 0.465 | 0.353 | 0.135 | 0.078 | 0.377 | 0.204 | 0.551 |
| Docling layout-heron (DocLayNet RT-DETRv2) | 0.290 | 0.067 | 0.979 | 0.776 | 0.252 | 0.121 | 0.507 | 0.321 | 0.622 |
| PP-DocLayout-L (off-the-shelf) | 0.310 | 0.069 | 0.926 | 0.659 | 0.558 | 0.352 | 0.598 | 0.360 | 0.679 |
| DocLayout-YOLO DocStructBench (off-the-shelf) | 0.529 | 0.152 | 0.867 | 0.726 | 0.000 | 0.000 | 0.465 | 0.293 | 0.520 |
| Azure AI Document Intelligence prebuilt-layout | 0.404 | 0.112 | 0.974 | 0.826 | 0.185 | 0.096 | 0.521 | 0.345 | 0.673 |
| AWS Textract Layout | 0.042 | 0.007 | 0.808 | 0.478 | 0.000 | 0.000 | 0.283 | 0.162 | 0.360 |
| Google Document AI Layout Parser | 0.080 | 0.012 | 0.951 | 0.728 | 0.000 | 0.000 | 0.344 | 0.246 | 0.407 |

Per-class P/R/F1 at the paper operating point (schema a):

| system | conf | hf P | hf R | hf F1 | ta P | ta R | ta F1 | fn P | fn R | fn F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RT-DETR-l tam2col (ours) | 0.50 | 0.943 | 0.956 | 0.949 | 0.988 | 0.988 | 0.988 | 0.933 | 0.933 | 0.933 |
| RF-DETR-L tam2col (ours) | 0.30 | 0.956 | 0.970 | 0.963 | 0.973 | 0.973 | 0.973 | 0.913 | 0.933 | 0.923 |
| DocLayout-YOLO tam2col (ours) | 0.30 | 0.938 | 0.959 | 0.948 | 1.000 | 0.994 | 0.997 | 0.929 | 0.867 | 0.897 |
| PP-DocLayout-L tam2col | 0.75 | 0.976 | 0.932 | 0.954 | 0.979 | 0.975 | 0.977 | 0.952 | 0.889 | 0.920 |
| Docling layout-heron tam2col | 0.05 | 0.926 | 0.954 | 0.940 | 0.978 | 0.869 | 0.920 | 0.913 | 0.933 | 0.923 |
| Surya fast layout (RF-DETR, DocLayNet-style) | 0.30 | 0.883 | 0.908 | 0.895 | 0.980 | 0.980 | 0.980 | 0.362 | 0.556 | 0.439 |
| Surya 2 layout VLM (surya-ocr-2) | 0.00 | 0.883 | 0.873 | 0.878 | 0.989 | 0.987 | 0.988 | 0.397 | 0.556 | 0.463 |
| Chandra 2 (chandra-ocr-2) | 0.00 | 0.899 | 0.576 | 0.702 | 0.740 | 0.612 | 0.670 | 0.667 | 0.178 | 0.281 |
| Docling layout-heron (DocLayNet RT-DETRv2) | 0.50 | 0.442 | 0.528 | 0.481 | 0.988 | 0.987 | 0.988 | 0.302 | 0.578 | 0.397 |
| PP-DocLayout-L (off-the-shelf) | 0.30 | 0.534 | 0.442 | 0.484 | 0.964 | 0.822 | 0.887 | 0.889 | 0.533 | 0.667 |
| DocLayout-YOLO DocStructBench (off-the-shelf) | 0.20 | 0.643 | 0.672 | 0.657 | 0.968 | 0.845 | 0.902 | 0.000 | 0.000 | 0.000 |
| Azure AI Document Intelligence prebuilt-layout | 0.00 | 0.680 | 0.579 | 0.625 | 0.989 | 0.989 | 0.989 | 0.370 | 0.444 | 0.404 |
| AWS Textract Layout | 0.00 | 0.253 | 0.147 | 0.186 | 0.896 | 0.890 | 0.893 | 0.000 | 0.000 | 0.000 |
| Google Document AI Layout Parser | 0.00 | 0.429 | 0.172 | 0.246 | 0.973 | 0.975 | 0.974 | 0.000 | 0.000 | 0.000 |

### Schema (b) — DocLayNet-aligned 4-class (no envelope merge)

| system | page-header AP50 / AP50-95 | page-footer AP50 / AP50-95 | footnote AP50 / AP50-95 | text-area AP50 / AP50-95 | mean AP50 | mean AP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| RT-DETR-l tam2col (ours) | 0.974 / 0.707 | 0.942 / 0.640 | 0.990 / 0.822 | 0.724 / 0.688 | 0.907 | 0.714 |
| RF-DETR-L tam2col (ours) | 0.982 / 0.756 | 0.962 / 0.696 | 0.972 / 0.815 | 0.731 / 0.690 | 0.912 | 0.739 |
| DocLayout-YOLO tam2col (ours) | 0.960 / 0.713 | 0.959 / 0.682 | 0.924 / 0.722 | 0.706 / 0.665 | 0.887 | 0.696 |
| PP-DocLayout-L tam2col | 0.976 / 0.748 | 0.964 / 0.687 | 0.929 / 0.795 | 0.743 / 0.698 | 0.903 | 0.732 |
| Docling layout-heron tam2col | 0.951 / 0.647 | 0.961 / 0.663 | 0.984 / 0.853 | 0.667 / 0.618 | 0.891 | 0.695 |
| Surya fast layout (RF-DETR, DocLayNet-style) | 0.895 / 0.310 | 0.874 / 0.254 | 0.327 / 0.164 | 0.406 / 0.220 | 0.625 | 0.237 |
| Surya 2 layout VLM (surya-ocr-2) | 0.746 / 0.252 | 0.809 / 0.297 | 0.220 / 0.104 | 0.109 / 0.058 | 0.471 | 0.178 |
| Chandra 2 (chandra-ocr-2) | 0.751 / 0.251 | 0.239 / 0.090 | 0.135 / 0.078 | 0.168 / 0.102 | 0.323 | 0.130 |
| Docling layout-heron (DocLayNet RT-DETRv2) | 0.484 / 0.120 | 0.121 / 0.022 | 0.252 / 0.121 | 0.496 / 0.292 | 0.338 | 0.139 |
| PP-DocLayout-L (off-the-shelf) | 0.507 / 0.124 | 0.131 / 0.023 | 0.558 / 0.352 | 0.639 / 0.470 | 0.459 | 0.242 |
| DocLayout-YOLO DocStructBench (off-the-shelf) | 0.279 / 0.090 | 0.000 / 0.000 | 0.000 / 0.000 | 0.285 / 0.161 | 0.141 | 0.063 |
| Azure AI Document Intelligence prebuilt-layout | 0.446 / 0.169 | 0.336 / 0.076 | 0.185 / 0.096 | 0.174 / 0.090 | 0.285 | 0.108 |
| AWS Textract Layout | 0.002 / 0.001 | 0.050 / 0.008 | 0.000 / 0.000 | 0.296 / 0.152 | 0.087 | 0.040 |
| Google Document AI Layout Parser | 0.078 / 0.010 | 0.100 / 0.021 | 0.000 / 0.000 | 0.052 / 0.021 | 0.058 | 0.013 |

Schema (b) text-area is **paragraph/block granularity** for off-the-shelf
DocLayNet-style detectors and **page-envelope (or two-column) granularity**
for `tam2col` models. Those AP numbers are **not** comparable across the
two groups; they are reported so a reader can line our native boxes up with
DocLayNet Table 2's `Text` column, with that caveat. Mean AP in (b) averages
all four classes including text-area.

Shared-class mean (page-header, page-footer, footnote only — text-area excluded):

| system | shared AP50 | shared AP50-95 |
|---|---:|---:|
| RT-DETR-l tam2col (ours) | 0.969 | 0.723 |
| RF-DETR-L tam2col (ours) | 0.972 | 0.755 |
| DocLayout-YOLO tam2col (ours) | 0.948 | 0.706 |
| PP-DocLayout-L tam2col | 0.956 | 0.743 |
| Docling layout-heron tam2col | 0.965 | 0.721 |
| Surya fast layout (RF-DETR, DocLayNet-style) | 0.698 | 0.243 |
| Surya 2 layout VLM (surya-ocr-2) | 0.592 | 0.218 |
| Chandra 2 (chandra-ocr-2) | 0.375 | 0.140 |
| Docling layout-heron (DocLayNet RT-DETRv2) | 0.285 | 0.088 |
| PP-DocLayout-L (off-the-shelf) | 0.399 | 0.166 |
| DocLayout-YOLO DocStructBench (off-the-shelf) | 0.093 | 0.030 |
| Azure AI Document Intelligence prebuilt-layout | 0.322 | 0.114 |
| AWS Textract Layout | 0.017 | 0.003 |
| Google Document AI Layout Parser | 0.060 | 0.010 |

## Task 2 — domain transfer

Public DocLayNet detector: **Docling layout-heron**
(`docling-project/docling-layout-heron`, RT-DETRv2-R50, trained on the
DocLayNet-family mix; published COCO mAP **0.699** on DocLayNet v1 with
no post-processing [@docling-heron]). Class map for shared classes:

- our `header` ↔ DocLayNet `Page-header`
- our `footer` ↔ DocLayNet `Page-footer`
- our `footnote` ↔ DocLayNet `Footnote`
- our `text-area` envelope is **not** comparable to DocLayNet paragraph `Text`;
  it is reported separately in schema (b) and **not** averaged into the 2×2.

### DocLayNet detector → our test (shared classes)

| system | page-header AP50 / AP50-95 | page-footer AP50 / AP50-95 | footnote AP50 / AP50-95 | shared mAP@0.50:0.95 | hf F1 | fn F1 | hf contam. | hf recov. | hf hidden-IoU | fn contam. | fn recov. | fn hidden-IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RT-DETR-l tam2col (ours) | 0.974 / 0.707 | 0.942 / 0.640 | 0.990 / 0.822 | 0.723 | 0.958 / 0.931 | 0.933 | 0.6% | 0.3% | 0.1% | 2.2% | 2.2% | 2.2% |
| RF-DETR-L tam2col (ours) | 0.982 / 0.756 | 0.962 / 0.696 | 0.972 / 0.815 | 0.755 | 0.966 / 0.956 | 0.923 | 0.1% | 0.1% | 0.0% | 6.7% | 6.7% | 4.4% |
| DocLayout-YOLO tam2col (ours) | 0.960 / 0.713 | 0.959 / 0.682 | 0.924 / 0.722 | 0.706 | 0.947 / 0.950 | 0.897 | 0.1% | 0.3% | 0.0% | 0.0% | 0.0% | 0.0% |
| PP-DocLayout-L tam2col | 0.976 / 0.748 | 0.964 / 0.687 | 0.929 / 0.795 | 0.743 | 0.954 / 0.954 | 0.920 | 0.2% | 0.1% | 0.0% | 6.7% | 0.0% | 2.2% |
| Docling layout-heron tam2col | 0.951 / 0.647 | 0.961 / 0.663 | 0.984 / 0.853 | 0.721 | 0.950 / 0.926 | 0.923 | 0.1% | 0.3% | 0.1% | 2.2% | 2.2% | 2.2% |
| Surya fast layout (RF-DETR, DocLayNet-style) | 0.895 / 0.310 | 0.874 / 0.254 | 0.327 / 0.164 | 0.243 | 0.894 / 0.894 | 0.439 | 1.2% | 5.4% | 0.5% | 15.6% | 6.7% | 13.3% |
| Docling layout-heron (DocLayNet RT-DETRv2) | 0.484 / 0.120 | 0.121 / 0.022 | 0.252 / 0.121 | 0.088 | 0.613 / 0.312 | 0.397 | 3.4% | 13.1% | 0.8% | 8.9% | 13.3% | 4.4% |

### 2×2 summary — shared-class COCO mAP@0.50:0.95

| | our 860-page test (shared 3 classes) | DocLayNet v1.2 test (shared 3 classes) |
|---|---:|---:|
| our RT-DETR-l tam2col | 0.723 | 0.006 |
| Docling layout-heron (DocLayNet) | 0.088 | 0.699 *(published 11-class; see caveat)* |

Caveat on the bottom-right cell: heron's published 0.699 is **11-class**
DocLayNet mAP, not the 3 shared classes. A shared-class-only number requires
re-running heron on DocLayNet test; until that JSON lands we quote the
published figure and do not average it with the 3-class cells.

Both detectors score **0.66–0.72** shared mAP on their own corpus and
collapse below **0.09** on the other — fine-tuning does not transfer
across corpora in either direction.

### Ours → DocLayNet v1.2 test (shared classes)

RT-DETR-l tam2col on **4999** DocLayNet v1.2 test pages 
(conf 0.50 operating point; dump conf 0.05). Shared-class 
mean excludes text-area (page/column envelope vs DocLayNet paragraph 
`Text`).

| | page-header AP50 / AP50-95 | page-footer AP50 / AP50-95 | footnote AP50 / AP50-95 | shared mAP@0.50 | shared mAP@0.50:0.95 |
|---|---:|---:|---:|---:|---:|
| our RT-DETR-l tam2col | 0.010 / 0.002 | 0.056 / 0.009 | 0.022 / 0.006 | 0.029 | 0.006 |

| | header F1 | footer F1 | footnote F1 | hf contam. | fn contam. | COTe | Trespass | Excess | LED-Merge hf | LED-Missing hf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| our RT-DETR-l tam2col | 0.031 | 0.164 | 0.130 | 39.9% | 27.1% | 0.908 | 0.020 | 0.397 | 5 | 4125 |

Transfer is near-zero on the shared classes: header recall at the paper
operating point is 2.0% (66/3366), and 39.9% of DocLayNet header-footer
GT is absorbed into the predicted text-area envelope. High COTe
(0.908) is coverage of that envelope, not header/footer quality;
Excess 0.397 is the page-scale text-area vs paragraph-`Text` mismatch.
LED-Merge stays tiny (5) because the failure mode is Missing, not
same-class glue — the same pattern as Task 3 on our test set.

## Task 3 — COTe and LED vs contamination

| system | hf contam. | fn contam. | COTe | Coverage | Overlap | **Trespass** | Excess | LED-Merge hf | LED-Missing hf | LED-Merge fn | LED-Missing fn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RT-DETR-l tam2col (ours) | 0.6% | 2.2% | 0.983 | 0.986 | 0.001 | 0.001 | 0.020 | 6 | 36 | 0 | 3 |
| RF-DETR-L tam2col (ours) | 0.1% | 6.7% | 0.985 | 0.989 | 0.002 | 0.002 | 0.018 | 7 | 22 | 0 | 3 |
| DocLayout-YOLO tam2col (ours) | 0.1% | 0.0% | 0.976 | 0.977 | 0.001 | 0.000 | 0.018 | 4 | 40 | 0 | 6 |
| PP-DocLayout-L tam2col | 0.2% | 6.7% | 0.985 | 0.985 | 0.000 | 0.001 | 0.019 | 4 | 79 | 0 | 5 |
| Docling layout-heron tam2col | 0.1% | 2.2% | 0.865 | 0.866 | 0.002 | 0.000 | 0.011 | 8 | 45 | 0 | 3 |
| Surya fast layout (RF-DETR, DocLayNet-style) | 1.2% | 15.6% | 0.917 | 0.926 | 0.004 | 0.005 | 0.014 | 14 | 36 | 0 | 7 |
| Surya 2 layout VLM (surya-ocr-2) | 1.0% | 13.3% | 0.920 | 0.923 | 0.001 | 0.002 | 0.007 | 5 | 63 | 0 | 7 |
| Chandra 2 (chandra-ocr-2) | 0.9% | 0.0% | 0.598 | 0.599 | 0.000 | 0.000 | 0.004 | 4 | 553 | 0 | 36 |
| Docling layout-heron (DocLayNet RT-DETRv2) | 3.4% | 8.9% | 0.901 | 0.918 | 0.010 | 0.007 | 0.021 | 4 | 90 | 0 | 4 |
| PP-DocLayout-L (off-the-shelf) | 11.9% | 42.2% | 0.736 | 0.745 | 0.002 | 0.007 | 0.027 | 8 | 316 | 0 | 21 |
| DocLayout-YOLO DocStructBench (off-the-shelf) | 6.2% | 20.0% | 0.770 | 0.779 | 0.003 | 0.006 | 0.027 | 31 | 170 | 0 | 45 |
| Azure AI Document Intelligence prebuilt-layout | 12.3% | 22.2% | 0.914 | 0.921 | 0.000 | 0.007 | 0.019 | 5 | 265 | 0 | 11 |
| AWS Textract Layout | 16.7% | 55.6% | 0.761 | 0.768 | 0.000 | 0.007 | 0.030 | 0 | 727 | 0 | 45 |
| Google Document AI Layout Parser | 57.3% | 97.8% | 0.912 | 0.925 | 0.000 | 0.013 | 0.091 | 16 | 883 | 0 | 45 |

### Contamination vs hidden contamination

Paper **contamination** = missed as own class and ≥50% inside the predicted
text-area envelope (the OCR crop). That is also **hidden** under envelope
geometry: there is no same-class box to punch out.

- **recoverable**: detected as own class (IoU≥0.5) *and* ≥50% inside the
  TA envelope — dual label; a post-processor can subtract the header/footnote
  from the crop.
- **hidden-IoU**: *not* detected as own class, but IoU≥0.5 against a *native*
  text-area box (no envelope merge). The model emitted a similarly-sized
  body box on that region. Page-level TA envelopes rarely hit this bar.

| system | hf contam.=hidden | hf recov. | hf in-TA | hf as-TA IoU | hf hidden-IoU | fn contam.=hidden | fn recov. | fn in-TA | fn as-TA IoU | fn hidden-IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RT-DETR-l tam2col (ours) | 0.6% | 0.3% | 0.9% | 0.1% | 0.1% | 2.2% | 2.2% | 4.4% | 4.4% | 2.2% |
| RF-DETR-L tam2col (ours) | 0.1% | 0.1% | 0.2% | 0.0% | 0.0% | 6.7% | 6.7% | 13.3% | 6.7% | 4.4% |
| DocLayout-YOLO tam2col (ours) | 0.1% | 0.3% | 0.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| PP-DocLayout-L tam2col | 0.2% | 0.1% | 0.3% | 0.0% | 0.0% | 6.7% | 0.0% | 6.7% | 2.2% | 2.2% |
| Docling layout-heron tam2col | 0.1% | 0.3% | 0.4% | 0.3% | 0.1% | 2.2% | 2.2% | 4.4% | 4.4% | 2.2% |
| Surya fast layout (RF-DETR, DocLayNet-style) | 1.2% | 5.4% | 6.6% | 5.4% | 0.5% | 15.6% | 6.7% | 22.2% | 20.0% | 13.3% |
| Surya 2 layout VLM (surya-ocr-2) | 1.0% | 0.2% | 1.2% | 0.7% | 0.7% | 13.3% | 2.2% | 15.6% | 11.1% | 11.1% |
| Chandra 2 (chandra-ocr-2) | 0.9% | 0.0% | 0.9% | 0.9% | 0.9% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Docling layout-heron (DocLayNet RT-DETRv2) | 3.4% | 13.1% | 16.5% | 13.2% | 0.8% | 8.9% | 13.3% | 22.2% | 17.8% | 4.4% |
| PP-DocLayout-L (off-the-shelf) | 11.9% | 5.0% | 16.9% | 3.9% | 3.8% | 42.2% | 6.7% | 48.9% | 35.6% | 33.3% |
| DocLayout-YOLO DocStructBench (off-the-shelf) | 6.2% | 6.8% | 13.0% | 6.5% | 3.3% | 20.0% | 0.0% | 20.0% | 15.6% | 15.6% |
| Azure AI Document Intelligence prebuilt-layout | 12.3% | 0.2% | 12.5% | 10.2% | 10.2% | 22.2% | 0.0% | 22.2% | 13.3% | 13.3% |
| AWS Textract Layout | 16.7% | 0.6% | 17.3% | 5.8% | 5.8% | 55.6% | 0.0% | 55.6% | 33.3% | 33.3% |
| Google Document AI Layout Parser | 57.3% | 1.3% | 58.6% | 18.7% | 18.6% | 97.8% | 0.0% | 97.8% | 53.3% | 53.3% |

### Do COTe-Trespass and LED-Merge rank systems like contamination?

Lower is better for all three (contamination rate, mean Trespass, Merge count).
Ranks are 1 = least damage.

| system | contamination rank | COTe-Trespass rank | LED-Merge rank | agree T? | agree M? |
|---|---:|---:|---:|---|---|
| RT-DETR-l tam2col (ours) | 5 | 5 | 8 | yes | no |
| RF-DETR-L tam2col (ours) | 3 | 6 | 9 | no | no |
| DocLayout-YOLO tam2col (ours) | 1 | 1 | 2 | yes | no |
| PP-DocLayout-L tam2col | 4 | 4 | 3 | yes | no |
| Docling layout-heron tam2col | 2 | 2 | 10 | yes | no |
| Surya fast layout (RF-DETR, DocLayNet-style) | 8 | 8 | 12 | yes | no |
| Surya 2 layout VLM (surya-ocr-2) | 7 | 7 | 6 | yes | no |
| Chandra 2 (chandra-ocr-2) | 6 | 3 | 4 | no | no |
| Docling layout-heron (DocLayNet RT-DETRv2) | 9 | 12 | 5 | no | no |
| PP-DocLayout-L (off-the-shelf) | 12 | 11 | 11 | no | no |
| DocLayout-YOLO DocStructBench (off-the-shelf) | 10 | 9 | 14 | no | no |
| Azure AI Document Intelligence prebuilt-layout | 11 | 10 | 7 | no | no |
| AWS Textract Layout | 13 | 13 | 1 | yes | no |
| Google Document AI Layout Parser | 14 | 14 | 13 | yes | no |

Exact rank agreement with contamination: COTe-Trespass 8/14,
LED-Merge 0/14. Spearman correlations are in the JSON
(`rank_agreement`). Disagreements are discussed in the results note body
after the numbers; typical sources are (i) Trespass counting *any* cross-SSU
pixel including header↔footer, not only body-absorb, and (ii) LED-Merge
requiring a *same-class* pred that covers two GT boxes, whereas
contamination fires when the *text-area* envelope swallows a miss.

Spearman ρ(contamination, COTe-Trespass) = **0.934**; 
ρ(contamination, LED-Merge) = **0.284**.

## Caveats

1. **Class granularity.** Schema (a) merges text-area; schema (b) does not.
   DocLayNet `Text` is paragraph-level; our `text-area` is a page (or column)
   envelope. Shared-class means drop text-area for that reason.
2. **Axis-aligned boxes on rotated scans.** Annotations and predictions are
   AABB; slight page skew loosens IoU identically for every system.
3. **IoU.** COCO mAP uses the standard 0.50:0.05:0.95 sweep. Operating-point
   P/R/F1 and contamination still use IoU ≥ 0.5, matching the paper.
4. **DocLayout-YOLO off-the-shelf** has no page-footer class (DocStructBench
   `abandon` was mapped onto header); schema (b) page-footer AP is ~0.
5. **Azure** emits no confidence; COCO AP treats every box as score 1.0, so
   AP collapses toward the single-threshold P/R operating point.
6. **Seeds.** Each checkpoint is a single training run (Ultralytics `seed=0`,
   RF-DETR `seed=null`). The 0.93–0.96 fine-tuned F1 band is across
   *architectures*, not seeds. Multi-seed retraining was not repeated here;
   see the reproducibility appendix.
7. **Character Error Vector** [@character-error-vector] was not wired: it
   needs an OCR stage on predicted crops plus character-level GT, which this
   layout dump does not contain.

## How to reproduce

```bash
source /home/eroux/pvenvs/1/bin/activate
python evaluation/run_literature_eval.py \
    --gt-dir /home/eroux/azure_di_eval/testset/labels/test \
    --img-dir /home/eroux/azure_di_eval/testset/images/test \
    --out-dir evaluation/eval_results/literature
```

See `evaluation/eval_results/REPRODUCIBILITY.md` for seeds, hardware,
hyperparameters, and the DocLayNet-transfer commands.

