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

> **Unified scorer (Aug 2026).** F1, canonical COCO mAP, Hidden-Trespass / COTe /
> LED are now all produced by a single path — `run_v4_lit_eval.py` +
> `literature_metrics.py` — at each system's own best-mean-F1 operating point, so
> the headline table and the HT/COTe pass report one F1 per system. The earlier
> `canon_sweep_preds.py` headline diverged from it for two models: the text-area
> envelope is now rebuilt from only the boxes above the operating confidence
> (was: one envelope from all boxes gated by max-conf, which RF-DETR's spurious
> low-confidence body boxes inflated → text-area F1 0.954 vs 0.996), and the
> operating point is chosen on a 0.01 grid (was 0.05, which pinned heron's
> footnote at 0.809 instead of 0.833). See
> [`UNIFIED_SCORING.md`](UNIFIED_SCORING.md).

### Ours (fine-tuned on v4)

Columns: canonical per-class F1 + mean at the best-mean-F1 conf, canonical COCO
mAP (mean over the 3 classes), and the DocLayNet-aligned **shared mAP@50:95**
(page-header / page-footer / footnote, text-area excluded; see
[`DOCLAYNET_SHARED.md`](DOCLAYNET_SHARED.md)) — a stricter, cross-corpus-comparable
number on the same predictions.

| model | mean F1 | header-footer | text-area | footnote | conf | mean AP50 | mean AP50-95 | shared mAP@50:95 |
|---|---|---|---|---|---|---:|---:|---:|
| RT-DETR-l (`rtdetr_tdlav4_tam2col`, seed0) | **0.959** | 0.952 | 0.999 | **0.925** | 0.74 | 0.974 | 0.786 | **0.650** |
| PP-DocLayout-L (`pp_doclayout_tdlav4_tam2col`) | 0.958 | 0.951 | 0.997 | **0.925** | 0.68 | 0.959 | 0.781 | 0.641 |
| RF-DETR-Large (`rfdetr_tdlav4_tam2col`) | 0.927 | 0.949 | 0.996 | 0.835 | 0.26 | 0.925 | 0.667 | 0.604 |
| Docling layout-heron (`docling_heron_tdlav4_tam2col`) | 0.925 | 0.944 | 0.998 | 0.833 | 0.08 | 0.912 | 0.735 | 0.593 |
| DocLayout-YOLO (`doclayout_tdlav4_tam2col`) | 0.897 | 0.931 | 0.980 | 0.779 | 0.29 | 0.919 | 0.720 | 0.586 |

RT-DETR-l (seed0) and PP-DocLayout-L top the table at **0.959 / 0.958** — a
dead heat — both with the best footnote class of any system (0.925) and
near-perfect text-area. Under the unified scorer RF-DETR (0.927) and heron
(0.925) are also a tie (the old two-driver split had flipped their order). All
five fine-tunes sit in a tight 0.90–0.96 band; header-footer and text-area are
stable across the whole sweep, footnote is the volatile class that sets the
operating point.

> The PP-DocLayout-L fine-tune trained to epoch 75 (val COCO mAP 0.832) but its
> auto-exported PIR inference graph — built by the training box's newer paddle —
> would not load under the eval box's paddle 3.0.0 (`strides is not right`), which
> is what crashed the on-box self-eval. The graph was re-exported from
> `best_model.pdparams` with paddle 3.0.0 / PaddleDetection on the eval box; the
> re-exported preds + sweep are what the numbers above and the literature pass use.

The RT-DETR-l row above is the **seed0** run scored through the unified path
(0.959 @ conf 0.74), the same dump used everywhere else and the tam2col row of
the curriculum ablation ([`CURRICULUM.md`](CURRICULUM.md)).

**RT-DETR-l seed variance** (5 independent seeds 0–4, identical `tam2col` recipe,
leak-free v4 833-page test), **re-scored under the unified scorer** (each seed at
its own best-mean-F1 conf) so the band matches the headline: canonical mean-F1
**0.961 ± 0.009** (sample sd; min 0.947 / max 0.970), header-footer
**0.947 ± 0.004**, text-area **0.998 ± 0.001**, footnote **0.937 ± 0.024**
(footnote is the volatile class), canonical mAP@0.50:0.95 **0.773 ± 0.009**
(conf-independent — unchanged from the old band), shared mAP@0.50:0.95
**0.643 ± 0.006**. Full table + per-seed rows:
[`THRESHOLDS_AND_SEED_VARIANCE.md`](THRESHOLDS_AND_SEED_VARIANCE.md);
JSON `seed_variance_unified.json`. The mean (0.961) is the recommended headline;
seed0 (0.959, the served checkpoint, in the table above) sits just below it.
This ± band is the reference spread for reading the single-run numbers above: the
two lead rows (RT-DETR-l 0.959, PP-DocLayout-L 0.958) are within noise of each
other, RF-DETR (0.927) and heron (0.925) are a second tie ~1.5 sd below, and
DocLayout-YOLO (0.897) trails — so the ordering RT-DETR ≈ PP-DocLayout-L >
RF-DETR ≈ heron > DocLayout-YOLO is real, but any two rows within ~1 sd of each
other should be treated as a tie.

**Served per-class thresholds** (h/f ≈ 0.60, text-area ≈ 0.55, footnote ≈ 0.25),
re-established leak-free on the served seed0 model: full P/R/F1 sweep per class
(canonical + native text-area) in
[`THRESHOLDS_AND_SEED_VARIANCE.md`](THRESHOLDS_AND_SEED_VARIANCE.md) (JSON
`served_thresholds.json`). Raising h/f 0.25→0.60 buys +0.028 precision for −0.014
recall; text-area 0.25→0.55 (native) buys +0.007 precision at no recall cost; the
global best-mean-F1 conf is 0.74 and one global knob costs 0.0085 mean F1 vs
per-class tuning. Footnote is the exception on the leak-free test (only 38 GT
boxes): F1-optimal is ~0.87, but the served 0.25 is a recall-safe operational
choice.

### Label-scheme ablation (curriculum), leak-free on the final architecture

The five label variants (baseline / tam / tam2col / 3cls / 3cls_tam) retrained on
RT-DETR-l with the exact v4 recipe and scored on the v4 833-page test with the
unified evaluator. Full tables + verdict: [`CURRICULUM.md`](CURRICULUM.md);
JSON `curriculum/metrics.json`.

| variant | mean F1 | mean AP50-95 | native text-area AP50-95 (own / common-GT) |
|---|---:|---:|---:|
| baseline | 0.958 | 0.707 | 0.717 / 0.717 |
| tam | 0.946 | 0.785 | 0.962 / 0.410 |
| **tam2col** (production) | **0.959** | **0.786** | 0.964 / 0.422 |
| 3cls | 0.945 | 0.707 | 0.744 / 0.744 |
| 3cls_tam | 0.938 | 0.760 | 0.965 / 0.407 |

All three original conclusions hold on the leak-free split: keep header/footer
separate (merging drops footnote F1 to 0.894/0.864, the two lowest); merged
text-area lifts canonical text-area AP50-95 (0.742→0.935→0.958); `tam2col` is the
best production choice (top mean F1 / AP50-95). The "merged box inflates per-class
AP" **artifact reproduces and is stronger** — against a common multi-box GT the
envelope predictors collapse to 0.41–0.42 vs baseline's 0.717 (gap widened from
0.22 on the old dev split to 0.31 leak-free).

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
mean-F1 below our fine-tuned detectors (0.96). Chandra 2 has no layout-only mode —
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
| Docling layout-heron (`docling-layout-heron`) | `docling_heron_ots` | 0.590 | 0.488 | 0.989 | 0.292 | 0.58 |
| DocLayout-YOLO (DocStructBench) | `doclayout_docstruct_ots` | 0.503 | 0.605 | 0.904 | 0.000 | 0.08 |

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
