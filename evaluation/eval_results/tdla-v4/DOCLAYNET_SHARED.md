# tdla-v4 — DocLayNet-aligned shared-class mAP, domain transfer, and raw composition

Three additions for the paper, all on the leak-free **v4 833-page test** unless
stated. Everything except the one direction-A inference re-scores existing
prediction dumps. Same pycocotools `COCOeval` protocol as elsewhere (bbox, IoU
0.50:0.05:0.95, 101-pt, maxDets 100, conf floor 0.05).

Scripts: `evaluation/run_v4_shared_map.py` (G1 + diagnostic),
`evaluation/score_ours_on_doclaynet.py` + `evaluation/rtdetr_predict_doclaynet_stream.py`
(G2 direction A), `evaluation/v4_composition.py` (G4). Metric code:
`evaluation/literature_metrics.py` (`doclaynet` schema, `SHARED` classes,
`iou_recall_diagnostic`). Raw per-system JSON under `shared_map/`.

## G1 — DocLayNet-aligned shared 3-class mAP (all 13 v4 systems)

Shared class space maps `header → page-header`, `footer → page-footer`,
`footnote → footnote`; **text-area is excluded** (our page/column envelope is not
comparable to DocLayNet's paragraph-level `Text`). Unlike the canonical
`header-footer` class, **page-header and page-footer stay separate here**. GT is
the raw v4 test annotations.

| system | shared mAP@50:95 | shared mAP@50 | page-header AP | page-footer AP | footnote AP |
|---|---:|---:|---:|---:|---:|
| RT-DETR-l tam2col (ours) | **0.650** | 0.959 | 0.617 | 0.501 | 0.832 |
| PP-DocLayout-L tam2col (ours) | 0.641 | 0.947 | 0.622 | 0.503 | 0.799 |
| RF-DETR-Large tam2col (ours) | 0.604 | 0.925 | 0.606 | 0.501 | 0.704 |
| Docling layout-heron tam2col (ours) | 0.593 | 0.890 | 0.623 | 0.489 | 0.666 |
| DocLayout-YOLO tam2col (ours) | 0.586 | 0.901 | 0.612 | 0.494 | 0.652 |
| PP-DocLayout-L (off-the-shelf) | 0.169 | 0.376 | 0.097 | 0.025 | 0.384 |
| Docling layout-heron (off-the-shelf) | 0.075 | 0.271 | 0.118 | 0.027 | 0.079 |
| DocLayout-YOLO DocStructBench (OTS) | 0.031 | 0.101 | 0.092 | 0.000 | 0.000 |
| Surya 2 layout VLM | 0.200 | 0.578 | 0.265 | 0.213 | 0.123 |
| Chandra 2 | 0.152 | 0.397 | 0.249 | 0.070 | 0.137 |
| Azure DI | 0.103 | 0.295 | 0.180 | 0.092 | 0.035 |
| AWS Textract | 0.004 | 0.018 | 0.000 | 0.011 | 0.000 |
| Google DocAI | 0.006 | 0.029 | 0.008 | 0.009 | 0.000 |

Our five fine-tunes cluster at **0.59–0.65** shared mAP@50:95 (RT-DETR-l on top),
an order of magnitude above every off-the-shelf / commercial / VLM system
(≤0.20). The much lower absolute value than the canonical F1 (0.90–0.96) is
expected and comes from three effects, not a regression: (i) mAP@50:95 is far
stricter than F1 at a single IoU 0.5 — our shared mAP@**50** is 0.89–0.96; (ii)
splitting header/footer and dropping the text-area envelope removes our two
easiest signals; (iii) **page-footer** is the hard shared class for everyone
(~0.50 AP even for our models) because footers are small, sparse, and
side/edge-placed on Tibetan pecha. The ranking still matches the canonical
table (RT-DETR ≳ PP-DocLayout-L > RF-DETR ≳ heron ≳ DocLayout-YOLO).

## G2 — domain transfer (shared-class space above)

### Direction B — DocLayNet detector on our books (reuse dumps)

The DocLayNet-trained **Docling layout-heron**, run off-the-shelf on our v4 833
test, scores shared mAP@50:95 **0.075** (row above) — versus **0.655** for the
same shared three classes on the DocLayNet v1.2 test itself
(`eval_results/literature/metrics_heron_on_doclaynet.json`; cf. heron's published
DocLayNet mAP ≈ 0.699, so our re-score is faithful). A ~9× collapse crossing to
Tibetan pecha.

**Box-convention diagnostic (IoU≥0.1 vs ≥0.5, conf floor 0.05):**

| class | recall@0.1 | recall@0.5 | mIoU@0.1 | mIoU@0.5 |
|---|---:|---:|---:|---:|
| page-header | 1.000 | 0.747 | 0.576 | 0.648 |
| page-footer | 0.991 | **0.387** | 0.466 | 0.664 |
| footnote | 1.000 | 0.842 | 0.512 | 0.733 |

heron **localizes essentially every** header / footer / footnote region on our
books at IoU≥0.1 (recall 0.99–1.00) — it is not blind to them — but its boxes do
not tightly align: page-footer recall falls from **0.99 at IoU≥0.1 to 0.39 at
IoU≥0.5**, and mean matched IoU off-domain is ~0.65–0.73 vs **0.80–0.90 in-domain**
on DocLayNet. So most of the off-diagonal is a **box-convention mismatch** (loose
/ shifted boxes — DocLayNet's top/bottom page-furniture vs our side- and
edge-placed Tibetan furniture), not genuine misses. For contrast, our in-domain
RT-DETR-l on the same books holds recall 0.96–1.00 at **both** thresholds with
mean IoU 0.79–0.91:

| class (ours in-domain) | recall@0.1 | recall@0.5 | mIoU@0.1 | mIoU@0.5 |
|---|---:|---:|---:|---:|
| page-header | 0.989 | 0.973 | 0.820 | 0.829 |
| page-footer | 0.993 | 0.958 | 0.773 | 0.787 |
| footnote | 1.000 | 1.000 | 0.905 | 0.905 |

### Direction A — our production model on DocLayNet (new inference)

RT-DETR-l tam2col, tdlav4 weights (`seed-variance-tdlav4/seed0/best.pt`), run on
the DocLayNet v1.2 **test** set (4,999 pages), scored on the shared three classes.
Metrics: `doclaynet_transfer/metrics_ours_on_doclaynet_v4.json` (also
`s3://.../tdlav4/eval/doclaynet_transfer/`).

Shared mAP@50:95 **0.0045** (mAP@50 0.0199) — near-zero, matching the v2-weights
transfer (0.0057) and confirming the gap is symmetric: our Tibetan-pecha detector
does not fire on born-digital English pages either. Per-class AP@50:95:
page-header 0.0023, page-footer 0.0095, footnote 0.0017.

**Box-convention / miss diagnostic (IoU≥0.1 vs ≥0.5):**

| class | recall@0.1 | recall@0.5 | mIoU@0.1 | mIoU@0.5 |
|---|---:|---:|---:|---:|
| page-header | 0.534 | 0.069 | 0.337 | 0.627 |
| page-footer | 0.795 | 0.157 | 0.402 | 0.611 |
| footnote | 0.351 | 0.088 | 0.373 | 0.619 |

The failure mode is the **opposite** of direction B. There, the DocLayNet
detector *localised every* Tibetan region at IoU≥0.1 and only misplaced the
boxes. Here, our detector *does not even overlap* much of DocLayNet's furniture
at IoU≥0.1 — it only covers 53% of page-headers and 35% of footnotes, and its
mean matched IoU at 0.1 is low (0.34–0.40) — i.e. **genuine misses**, because our
model looks for side- and edge-placed Tibetan furniture and simply does not fire
on DocLayNet's top/bottom page-header/footer band. Both directions land at
near-zero shared mAP (0.075 and 0.004), but for complementary reasons —
box-convention mismatch one way, non-firing / missed regions the other.

## G4 — raw v4 composition (Table 1)

Box counts from the **raw** v4 YOLO annotations (Hub tag `v4`, 4 native classes —
NOT the tam2col-merged curriculum where text-area is collapsed to one envelope),
per split. Image set per split is the v4 split list; JSON:
`composition_v4.json`.

| split | images | header | footer | text-area | footnote | total |
|---|---:|---:|---:|---:|---:|---:|
| train | 6,743 | 6,431 | 4,991 | 8,185 | 295 | 19,902 |
| val | 749 | 848 | 575 | 1,096 | 34 | 2,553 |
| test | 833 | 876 | 667 | 1,424 | 38 | 3,005 |
| **corpus** | **8,325** | **8,155** | **6,233** | **10,705** | **367** | **25,460** |

Per-class corpus totals reproduce the reference exactly (header 8,155 /
footer 6,233 / text-area 10,705 / footnote 367 = 25,460 boxes); no correction
needed. Footnote is by far the rarest class (367 boxes, 1.4% of the corpus; only
38 in the test split), which is why footnote is the volatile class in every
table above.

## Reproducibility

- **G1 + diagnostic (CPU):**
  ```
  .venv_eval/bin/python evaluation/run_v4_shared_map.py \
      --gt-dir   <TDLA@v4>/labels/test \
      --img-dir  <TDLA@v4>/images/test \
      --sizes-cache evaluation/eval_results/tdla-v4/literature/test_image_sizes.json \
      --pred-root <local mirror of tdlav4/eval/*/preds/labels> \
      --out-dir evaluation/eval_results/tdla-v4/shared_map
  ```
- **G4 (CPU):**
  ```
  .venv_eval/bin/python evaluation/v4_composition.py \
      --labels-root <TDLA@v4>/labels --splits-dir data/splits/v4 \
      --out evaluation/eval_results/tdla-v4/composition_v4.json
  ```
- **G2 direction A (1 GPU, deterministic conf-floor inference):** on a g5/g6
  (A10G) with `/opt/pytorch` + `ultralytics` + `datasets` + `pycocotools`:
  ```
  python evaluation/rtdetr_predict_doclaynet_stream.py \
      --weights rtdetr_tdlav4_seed0.pt --out out --conf 0.05 --imgsz 1024 --device 0
  python evaluation/score_ours_on_doclaynet.py \
      --pred-dir out/pred_labels --gt-dir out/gt_labels --index out/index.jsonl \
      --out metrics_ours_on_doclaynet_v4.json --conf 0.50 --skip-cote
  ```
  DocLayNet v1.2 test is streamed from the Hub (`docling-project/DocLayNet-v1.2`,
  4,999 pages; one temp PNG at a time, no PDFs kept). Weights:
  `s3://.../seed-variance-tdlav4/seed0/best.pt`.
