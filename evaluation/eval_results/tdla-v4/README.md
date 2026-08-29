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

| model | mean F1 | header-footer | text-area | footnote | conf |
|---|---|---|---|---|---|
| RT-DETR-l (`rtdetr` 5-seed mean ± sd) | 0.944 ± 0.010 | 0.951 | 0.990 | 0.891 | 0.50 |
| Docling layout-heron (`docling_heron_tdlav4_tam2col`) | **0.917** | 0.944 | 0.998 | 0.809 | 0.00 |
| DocLayout-YOLO (`doclayout_tdlav4_tam2col`) | 0.897 | 0.931 | 0.980 | 0.779 | 0.30 |
| PP-DocLayout-L (`pp_doclayout_tdlav4_tam2col`) | _training_ | | | | |

heron footnote confidence is systematically low, so the best-mean-F1 point keeps
all boxes; header-footer and text-area are stable across the whole sweep.

**RT-DETR-l seed variance** (`seed-variance-tdlav4/seed_variance.json`, 5
independent seeds 0–4, identical `tam2col` recipe, scored on the leak-free v4
833-page test, conf 0.50): canonical mean-F1 **0.944 ± 0.010** (sample sd;
min 0.935 / max 0.959), header-footer **0.951 ± 0.006**, text-area
**0.990 ± 0.003**, footnote **0.891 ± 0.031** (footnote is the volatile class),
canonical mAP@0.50:0.95 **0.773 ± 0.009**. The single best seed (seed 4) reaches
mean-F1 0.959. This ± band is the reference spread for reading the single-run
heron / DocLayout-YOLO / PP numbers above: the ~0.02–0.05 gaps to heron and
DocLayout-YOLO are several seed-sigmas, so they are real, but the architecture
ranking within ~1 sigma should be treated as a tie.

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

> **Test-set caveat.** These three rows were scored on the **v2 860-page test
> split**, not the leak-free v4 833-page test (see `evaluation/Surya_Chandra.md`;
> its title says "tdlav4" but the body pins the 860-image v2 split). They are the
> same canonical pipeline (`canon_sweep_preds.py`, remap `0:0,1:1,2:2,3:0`,
> IoU≥0.5) so the *shape* of the result carries over, but the numbers are **not
> strictly comparable** to the v4 rows above and are pending a v4 re-run. On that
> v2 split, our `tam2col` RT-DETR-l scores 0.946 (≈ the 0.944 v4 seed mean), so
> the relative gaps are indicative.

| model (v2 860-page test) | mean F1 | header-footer | text-area | footnote |
|---|---|---|---|---|
| Surya 2 layout VLM (`surya-ocr-2`) | 0.776 | 0.878 | 0.988 | 0.463 |
| Surya fast layout (RF-DETR, `surya_layout2`) | 0.774 | 0.895 | 0.989 | 0.439 |
| Chandra 2 (`chandra-ocr-2`) | 0.551 | 0.702 | 0.670 | 0.281 |

Surya's 650 M VLM is a dead heat with the tiny RF-DETR fast-layout model and both
sit ~0.17 mean-F1 below our fine-tuned detector, mostly on footnotes (≤0.46 vs our
0.89). Chandra 2 has no layout-only mode — it fully OCRs the page and derives
blocks from the transcription, so on Tibetan (which it cannot read) it loops to
the token cap and collapses recall. Raw predictions:
`s3://.../off-the-shelf-eval/{surya_vlm,chandra_vlm}/`.

## Hidden Trespass + COTe (area-based failure analysis)

`evaluation/run_literature_eval.py` now emits, per system, the area-based
**Hidden Trespass** `HT_c` (undetected class-*c* GT area that bleeds into the OCR
crop), its complement `R_c` (detected-but-inside, removable pre-OCR), the total
`HT_c + R_c`, plus the library **COTe** decomposition (C/O/T/E via `cotescore`
0.2.0) and the class-resolved **text-area→peripheral** COTe-Trespass. It reports
Spearman ρ(HT, COTe-Trespass) and ρ(HT, LED-Merge) across systems. Count-based
contamination is retained as a secondary column. This scorer is dataset-agnostic;
running it over the full v4 system set is pending assembly of every v4 prediction
dir (PP-DocLayout + RF-DETR still training).
