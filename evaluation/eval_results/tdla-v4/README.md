# tdla-v4

Scores on Hub tag **`v4`** (8,325 images, **833-page test**).

Full v2 corpus re-split by series (`w_id`): no series in more than one split,
classes and side-margin vs top/bottom headers/footers stratified. Augmented
pages stay in train. Split lists: `data/splits/v4/`.

## S3 namespace: `tdlav4` (never bare `v4`)

Everything trained/evaluated on Hub tag `v4` lives under one prefix, so it can
never be confused with the old internal `_v4_` / `_v5_` model iterations
(e.g. `weights/rtdetr_v4_4cls_best.pt`, which is unrelated):

```
s3://bec.bdrc.io/models/hff-detection/tdlav4/
  DATASET                                  # tag, hub commit, counts, curriculum
  dataset/dataset_tdlav4_tam2col.tar.gz    # the tam2col curriculum (real JPEGs)
  weights/                                 # best checkpoints, *_tdlav4_tam2col*
  runs/                                    # full run dirs + logs
  eval/                                    # metric JSONs (dataset_tag: v4)
```

Run names are always `<arch>_tdlav4_tam2col`.

## v4 fine-tune runs (all on the `tam2col` curriculum)

| model | run name | box | framework |
|---|---|---|---|
| Docling layout-heron | `docling_heron_tdlav4_tam2col` | 8class-train-ssh | HF transformers RTDetrV2 |
| DocLayout-YOLO | `doclayout_tdlav4_tam2col` | paddlocr_job_v1 | doclayout_yolo fork |
| PP-DocLayout-L | `pp_doclayout_tdlav4_tam2col` | torch2 | PaddleX / PaddleDetection |

RT-DETR-l was retrained on v4 as a 5-seed variance fleet
(`s3://.../seed-variance-tdlav4/seed0..seed4` + `seed_variance.json`). RF-DETR-Large
is fine-tuned on v4 separately (`rfdetr_tdlav4_tam2col`, on the freed heron box).
Each run script syncs its best weights + logs to `tdlav4/weights|runs/` on
completion, printing a `*_TDLAV4_COMPLETE` marker.

## Results (canonical 3-class space, IoU>=0.5, best-mean-F1 operating point)

Metric = mean of per-class F1 over {header-footer, text-area, footnote}, with
text-area merged to one envelope per page and header+footer combined but matched
individually. Same protocol as the v2 paper tables. Per-model JSON alongside.

### Ours (fine-tuned on v4)

Last column is the DocLayNet-aligned **shared mAP@50:95** (page-header /
page-footer / footnote, text-area excluded; see
[`DOCLAYNET_SHARED.md`](DOCLAYNET_SHARED.md)) — a stricter, cross-corpus-comparable
number on the same predictions.

| model | mean F1 | header-footer | text-area | footnote | conf | shared mAP@50:95 |
|---|---|---|---|---|---|---:|
| PP-DocLayout-L (`pp_doclayout_tdlav4_tam2col`) | **0.957** | 0.950 | 0.997 | **0.925** | 0.70 | 0.641 |
| RT-DETR-l (`rtdetr` 5-seed mean ± sd) | 0.944 ± 0.010 | 0.951 | 0.990 | 0.891 | 0.50 | **0.650** |
| RF-DETR-Large (`rfdetr_tdlav4_tam2col`) | 0.926 | 0.948 | 0.996 | 0.835 | 0.25 | 0.604 |
| Docling layout-heron (`docling_heron_tdlav4_tam2col`) | 0.917 | 0.944 | 0.998 | 0.809 | 0.00 | 0.593 |
| DocLayout-YOLO (`doclayout_tdlav4_tam2col`) | 0.897 | 0.931 | 0.980 | 0.779 | 0.30 | 0.586 |

PP-DocLayout-L tops the table at **0.957** (above the RT-DETR seed mean by ~1.3
sd), driven by the best footnote class of any system (0.925) and near-perfect
text-area (0.997) over a very wide 0.40–0.80 confidence plateau. heron footnote
confidence is systematically low, so its best-mean-F1 point keeps all boxes;
header-footer and text-area are stable across the whole sweep.

> The PP-DocLayout-L fine-tune trained to epoch 75 (val COCO mAP 0.832) but its
> auto-exported PIR inference graph — built by the training box's newer paddle —
> would not load under the eval box's paddle 3.0.0 (`strides is not right`), which
> is what crashed the on-box self-eval. The graph was re-exported from
> `best_model.pdparams` with paddle 3.0.0 / PaddleDetection on the eval box; the
> re-exported preds + sweep are what the numbers above and the literature pass use.

**RT-DETR-l seed variance** (`seed-variance-tdlav4/seed_variance.json`, 5
independent seeds 0–4, identical `tam2col` recipe, scored on the leak-free v4
833-page test, conf 0.50): canonical mean-F1 **0.944 ± 0.010** (sample sd;
min 0.935 / max 0.959), header-footer **0.951 ± 0.006**, text-area
**0.990 ± 0.003**, footnote **0.891 ± 0.031** (footnote is the volatile class),
canonical mAP@0.50:0.95 **0.773 ± 0.009**. The single best seed (seed 4) reaches
mean-F1 0.959. This ± band is the reference spread for reading the single-run
PP-DocLayout-L / RF-DETR / heron / DocLayout-YOLO numbers above: PP-DocLayout-L
(0.957) sits ~1.3 sd above the RT-DETR seed mean (statistically on par with the
best single seed), RF-DETR (0.926) is ~1.8 sd below it, and heron (0.917) /
DocLayout-YOLO (0.897) are further out — so the ordering
PP-DocLayout-L ≳ RT-DETR > RF-DETR ≳ heron > DocLayout-YOLO is real, but any two
rows within ~1 sd of each other should be treated as a tie.

### Commercial layout APIs (off-the-shelf, no Tibetan fine-tuning)

Run on the identical v4 833-page test set; raw predictions + sweeps under
`s3://.../tdlav4/eval/<provider>/`. Textract carries per-block confidence
(sweepable); Azure and Google give a single un-thresholdable operating point.

| model | mean F1 | header-footer | text-area | footnote |
|---|---|---|---|---|
| Azure DI prebuilt-layout | 0.618 | 0.611 | 0.990 | 0.252 |
| Google DocAI Layout Parser | 0.373 | 0.151 | 0.967 | 0.000 |
| AWS Textract Layout | 0.356 | 0.232 | 0.835 | 0.000 |

Only Azure exposes a footnote role; Textract and Google have no footnote type, so
their footnote F1 is 0 by construction. All three trail our fine-tunes by a wide
margin on Tibetan header/footer geometry.

### Off-the-shelf Datalab models (Surya 2, Chandra 2)

Run on the identical v4 833-page test set (all 833 re-inferred — only 116 overlap
the old v2 test split). Same canonical pipeline (`canon_sweep_preds.py`, remap
`0:0,1:1,2:2,3:0`, IoU≥0.5). Full write-up: `evaluation/Surya_Chandra.md`.

| model (v4 833-page test) | mean F1 | header-footer | text-area | footnote |
|---|---|---|---|---|
| Surya 2 layout VLM (`surya-ocr-2`) | 0.777 | 0.865 | 0.993 | 0.474 |
| Chandra 2 (`chandra-ocr-2`) | 0.593 | 0.699 | 0.695 | 0.385 |

Surya's 650 M VLM reads header/footer and text-area well (0.87 / 0.99) but only
0.47 on footnotes, leaving it at 0.777 — above the commercial APIs yet ~0.17
mean-F1 below our fine-tuned detector (0.944). Chandra 2 has no layout-only mode —
it fully OCRs the page and derives blocks from the transcription, so on Tibetan
(which it cannot read) it loops to the 4000-token cap (avg 3,196 tok/page) and
collapses recall (0.593). Raw predictions:
`s3://.../off-the-shelf-eval-tdlav4/{surya_vlm,chandra_vlm}/`.

> The earlier v2 860-page run gave nearly identical figures (Surya 0.776, Chandra
> 0.551), so the split does not change the story; these v4 numbers are canonical.
> `run_literature_eval.py` still scores its `surya_vlm_ots` / `chandra_ots` rows
> on the v2 GT (consistent with the other v2 rows there).

### Off-the-shelf open-weight detectors (no Tibetan fine-tuning)

The same three architectures we fine-tune, run **off-the-shelf** on the identical
v4 833-page test set (raw preds + sweeps under
`s3://.../tdlav4/eval/ots/{pp_doclayout,docling_heron,doclayout_docstruct}/`).
Same canonical pipeline as every row above (`canon_sweep_preds.py`, remap
`0:0,1:1,2:2,3:0`, IoU≥0.5, best-mean-F1 point). This isolates how much of each
fine-tune's score comes from Tibetan adaptation vs the pretrained checkpoint.

| model (off-the-shelf) | run name | mean F1 | header-footer | text-area | footnote | conf |
|---|---|---|---|---|---|---|
| PP-DocLayout-L (`PP-DocLayout-L`) | `pp_doclayout_ots` | 0.670 | 0.464 | 0.869 | 0.677 | 0.30 |
| Docling layout-heron (`docling-layout-heron`) | `docling_heron_ots` | 0.588 | 0.488 | 0.993 | 0.282 | 0.55 |
| DocLayout-YOLO (DocStructBench) | `doclayout_docstruct_ots` | 0.502 | 0.615 | 0.889 | 0.000 | 0.15 |

All three read the **text-area** envelope well (0.87–0.99) — that transfers from
generic document layout — but collapse on the Tibetan-specific classes. PP-DocLayout-L
leads (0.670) purely because it is the only one with a usable **footnote** class
(0.677); its header-footer is weakest (0.464) since it never learned side-margin
headers. heron reads text-area near-perfectly (0.993) but only knows top/bottom
`page_header`/`page_footer` (0.488) and low-confidence footnotes (0.282).
DocLayout-YOLO/DocStructBench has **no genuine footnote class** (only
`table_footnote`, which never fires here), so footnote F1 is 0 by construction,
capping it at 0.502. Fine-tuning on v4 lifts every one of these into the
0.90–0.94 band (see the "Ours" table): +0.23 for DocLayout-YOLO, +0.33 for heron,
and the footnote class in particular goes from ≤0.68 to 0.78–0.89. So the bulk of
our numbers is Tibetan adaptation, not the pretrained backbone.

## DocLayNet-aligned shared-class mAP + domain transfer

Full write-up + tables: [`DOCLAYNET_SHARED.md`](DOCLAYNET_SHARED.md). All 13
systems re-scored in the DocLayNet-aligned shared space (page-header /
page-footer / footnote separate, text-area excluded), same COCO protocol. Our
fine-tunes cluster at **0.59–0.65** shared mAP@50:95 (RT-DETR-l 0.650 on top),
≥3× above every off-the-shelf / commercial / VLM system (≤0.20). Per-system JSON
under `shared_map/`.

**Domain transfer.** The DocLayNet-trained Docling layout-heron, off-the-shelf on
our books, drops from **0.655** shared mAP (on DocLayNet's own test) to **0.075**
on Tibetan pecha — but the IoU-sweep diagnostic shows it *finds* our
header/footer/footnote at IoU≥0.1 (recall 0.99–1.00) and only misaligns the boxes
(page-footer recall 0.99→0.39 at IoU≥0.5; box-convention mismatch, not misses).
Symmetrically, our production RT-DETR-l on the DocLayNet v1.2 test (4,999 pages)
scores shared mAP@50:95 **0.0045** — but here the failure is *genuine misses*
(it overlaps only 53% of DocLayNet page-headers / 35% of footnotes even at
IoU≥0.1), the complementary transfer failure. Metrics under
`doclaynet_transfer/` (and `s3://.../tdlav4/eval/doclaynet_transfer/`).

Raw per-split composition (Table 1) is in
[`DOCLAYNET_SHARED.md`](DOCLAYNET_SHARED.md) §G4 / `composition_v4.json` (corpus
25,460 boxes; footnote the rarest at 367).

## Hidden Trespass + COTe (area-based failure analysis)

`evaluation/run_literature_eval.py` emits, per system, the area-based
**Hidden Trespass** `HT_c` (undetected class-*c* GT area that bleeds into the OCR
crop), its complement `R_c` (detected-but-inside, removable pre-OCR), the total
`HT_c + R_c`, plus the library **COTe** decomposition (C/O/T/E via `cotescore`
0.2.0) and the class-resolved **text-area→peripheral** COTe-Trespass. It reports
Spearman ρ(HT, COTe-Trespass) and ρ(HT, LED-Merge) across systems. Count-based
contamination is retained as a secondary column.

`evaluation/run_v4_lit_eval.py` runs that same metric code over the **v4 system
set** (all 13 systems above, including the PP-DocLayout-L fine-tune) on
the leak-free 833-page test, each at its own best-mean-F1 operating point. Full
per-system JSON + tables: [`literature/RESULTS.md`](literature/RESULTS.md); raw
dump under `s3://.../tdlav4/eval/literature/`.

**Hidden Trespass, canonical (micro-averaged over the 833-page test):**

| system | hf HT | hf total | fn HT | fn total |
|---|---:|---:|---:|---:|
| PP-DocLayout-L (ours) | 0.003 | 0.005 | 0.037 | 0.037 |
| RT-DETR-l (ours, seed0) | 0.008 | 0.010 | 0.037 | 0.092 |
| Docling layout-heron (ours) | 0.002 | 0.005 | 0.074 | 0.074 |
| DocLayout-YOLO (ours) | 0.004 | 0.005 | 0.106 | 0.106 |
| RF-DETR-Large (ours) | 0.020 | 0.022 | 0.216 | 0.226 |
| Surya 2 VLM | 0.020 | 0.022 | 0.135 | 0.135 |
| Azure DI | 0.181 | 0.186 | 0.184 | 0.184 |
| Chandra 2 | 0.019 | 0.019 | 0.020 | 0.020 |
| DocLayout-YOLO DocStructBench (OTS) | 0.120 | 0.195 | 0.207 | 0.207 |
| Docling layout-heron (OTS) | 0.078 | 0.211 | 0.090 | 0.122 |
| PP-DocLayout-L (OTS) | 0.171 | 0.233 | 0.277 | 0.346 |
| AWS Textract | 0.160 | 0.168 | 0.644 | 0.644 |
| Google DocAI | 0.438 | 0.443 | 0.929 | 0.929 |

`HT` = the *hidden* (undetected) area bleeding into the OCR crop; `total` =
`HT + R` (whole text-area→class overlap). The best fine-tune, PP-DocLayout-L,
keeps both header-footer and footnote HT at ≤0.04 (footnote total 0.037 — it
detects essentially every footnote it bleeds into); our other fine-tunes stay at
≤0.02 header-footer / ≤0.11 footnote (RF-DETR's 0.22 footnote HT is the outlier
and matches its weaker footnote F1). The off-the-shelf and commercial systems
bleed an order of magnitude more — Google DocAI silently drops **93%** of
footnote area into the body crop, Textract **64%**.

**Cross-check (Spearman, all 13 systems):** ρ(HT, COTe-Trespass
text-area→peripheral) = **0.984**, so the area-based Hidden Trespass tracks the
independent library COTe-Trespass almost perfectly; ρ(HT, LED-Merge) = **−0.071**,
i.e. HT is orthogonal to same-class over-merging (LED-Merge measures a different
failure). The legacy count-based contamination gives ρ = **0.907** / **−0.033**
respectively — the area-based metric is the tighter proxy for COTe-Trespass.
