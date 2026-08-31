# Reproducibility appendix (draft)

This documents how every number in
`evaluation/eval_results/literature/` was produced. It is meant to be pasted
into the paper in a later integration pass. Do **not** treat it as a claim that
training was repeated with multiple seeds in this evaluation — it was not.

## What this pass actually ran

| item | value |
|---|---|
| Date | 2026-08-28 |
| Scoring host | local CPU (`pvenvs/1`, Python 3.11.2) |
| Scoring wall-clock | 290.9 s for 11 systems × 860 pages (COCO + sweep + contamination + LED + COTe) |
| Scoring hardware | no GPU (predictions were already on disk) |
| GPU used this pass | `i-0e77b5c8625c2c75a` (`torch2-clone-rtdetr`, g5.xlarge, NVIDIA A10G 24 GB) for DocLayNet transfer only |
| Libraries | `pycocotools==2.0.11`, `cotescore==0.2.0` |
| Randomness in scoring | none — greedy IoU matching and pycocotools COCOeval are deterministic given the dumps |

Failed runs this pass: one COTe import mismatch against `cotescore` 0.2.0
(`compute_canvas` was renamed to `eval_shape` in the installed wheel); fixed
and re-run. DocLayNet transfer: `ds4sd/DocLayNet` dataset scripts rejected;
unauthenticated Hub stream hung on teardown (fixed with `os._exit(0)`). The
authenticated streaming run then finished 4999 pages. No training jobs
were launched.

## Predictions (not re-inferred)

All Task 1 / Task 3 numbers, and the DocLayNet-detector → our-test half of
Task 2, score **archived YOLO dumps** under
`/home/eroux/azure_di_eval/`:

| system id | dump | paper operating conf |
|---|---|---|
| `rtdetr_tam2col` | `tam2col_pred/labels` | 0.50 |
| `rfdetr_tam2col` | `ft_preds/rfdetr/rfdetr_tam2col_pred/labels` | 0.30 |
| `doclayout_yolo_ft` | `ft_preds/dl_yolo/dl_yolo_ft_pred/labels` | 0.30 |
| `pp_doclayout_ft` | `ft_preds/pp_doclayout/pp_doclayout_ft_pred/labels` | 0.75 |
| `docling_heron_ft` | `ft_preds/docling_heron/docling_heron_ft_pred/labels` | 0.05 |
| `surya_ots` | `surya_pred/labels` | 0.30 |
| `docling_heron_ots` | `docling_heron_ots_pred/docling_heron_pred/labels` | 0.50 |
| `pp_doclayout_ots` | `pp_doclayout_pred/labels` | 0.30 |
| `doclayout_yolo_ots` | `doclayout_pred/labels` | 0.20 |
| `azure_di` | `azure_pred/labels` (no score column) | n/a (score:=1) |
| `aws_textract` | `aws_pred/labels` | 0.00 |
| `google_docai` | `/home/eroux/gdocai_eval/labels` (no score column) | n/a (score:=1) |

Ground truth: `/home/eroux/azure_di_eval/testset/labels/test` (860 txt) with
images in `.../images/test`. Pixel sizes are cached in
`evaluation/eval_results/literature/test_image_sizes.json`.

## Exact eval commands

```bash
source /home/eroux/pvenvs/1/bin/activate   # or: PYTHON=/home/eroux/pvenvs/1/bin/python
pip install pycocotools==2.0.11 cotescore==0.2.0   # once

bash evaluation/run_literature_eval.sh
# equivalent:
python evaluation/run_literature_eval.py \
    --gt-dir /home/eroux/azure_di_eval/testset/labels/test \
    --img-dir /home/eroux/azure_di_eval/testset/images/test \
    --out-dir evaluation/eval_results/literature
```

Outputs: `metrics.json`, `RESULTS.md`, per-system `{id}.json`.

### Ours → DocLayNet v1.2 test (shared classes)

Needs a GPU and Hugging Face **read** access to `docling-project/DocLayNet-v1.2`.
Images are **not** materialised: one temp PNG per page. Set `HF_TOKEN` from
`WRITY_HF_TOKEN` in `/home/eroux/BUDA/softs/ocr-evaluation-benchmark/env.sh`
(read-only Hub token). The script only calls `load_dataset` (download). It
never `push_to_hub`, never prints the token, and logs only
`Hugging Face auth: read token from env (download only)`.

This pass (2026-08-28 UTC):

| item | value |
|---|---|
| GPU host | `i-0e77b5c8625c2c75a` (`torch2-clone-rtdetr`, g5.xlarge, 1× A10G 24 GB) |
| Python | `/opt/pytorch/bin/python` (torch 2.11+cu130, ultralytics 8.4.90, datasets 4.8.4) |
| Dataset | `docling-project/DocLayNet-v1.2` split=`test`, streaming |
| Pages | **4999** pred + GT label files, 0 missing (unique stems = 4999) |
| Weights | `/home/ubuntu/tibetan_book_layout.pt` (HF `tibetan_book_layout.pt`) |
| Infer hyps | `imgsz=1024`, dump `conf=0.05`, device `0` |
| Score hyps | operating `conf=0.50` (paper tam2col point) |
| Infer wall-clock | 05:21:09–05:35:21 UTC (**~14.2 min** for pages 21–4999; pages 1–20 were a prior unauthenticated smoke) |
| Score wall-clock | 05:35:51–05:37:52 UTC (**~2 min**) on the same box |
| Shared mAP@0.50 / @0.50:0.95 | **0.029 / 0.006** |
| HF token use | `load_dataset(..., token=HF_TOKEN)` only |

Failed attempts this pass: `ds4sd/DocLayNet` (`Dataset scripts are no longer supported`); full PNG export aborted (root disk 97% full, ~3.8 G free); unauthenticated Hub stream was Hub-bound (~12 s/page) and hung on IterableDataset teardown (fixed with `os._exit(0)` + nohup).

```bash
# on a g5/g6 with ultralytics + datasets
set -a && source /path/to/ocr-evaluation-benchmark/env.sh && set +a
export HF_TOKEN="$WRITY_HF_TOKEN"   # read-only; do not echo
python evaluation/rtdetr_predict_doclaynet_stream.py \
    --weights tibetan_book_layout.pt \
    --out /path/to/doclaynet_transfer \
    --conf 0.05 --imgsz 1024 --device 0

python evaluation/score_ours_on_doclaynet.py \
    --pred-dir /path/to/doclaynet_transfer/pred_labels \
    --gt-dir   /path/to/doclaynet_transfer/gt_labels \
    --index    /path/to/doclaynet_transfer/index.jsonl \
    --out      /path/to/doclaynet_transfer/metrics_ours_on_doclaynet.json \
    --conf 0.50
```

Copy only `metrics_ours_on_doclaynet.json` back next to
`evaluation/eval_results/literature/` (do not copy the 4999-page dump; local
disk is tight). `run_literature_eval.py` picks that file up on the next
scoring pass.

### Docling layout-heron → DocLayNet v1.2 test (shared-3, closes the 2×2)

Same streaming recipe, heron instead of our RT-DETR
(`evaluation/heron_predict_doclaynet_stream.py`, `transformers`
`RTDetrV2ForObjectDetection`, heron labels mapped to our 4-class; heron's
`picture`/`checkbox_*` classes dropped). Scored with
`score_ours_on_doclaynet.py`.

| item | value |
|---|---|
| Pages | 4999 (own GT + index written from the same stream) |
| Infer hyps | dump `conf=0.05`, batch 1 |
| Score hyps | operating `conf=0.50` |
| Shared mAP@0.50 / @0.50:0.95 | **0.900 / 0.655** |
| Published 11-class reference | 0.699 |

This replaces the earlier not-apples-to-apples published 0.699 in the 2×2
bottom-right cell. Output: `metrics_heron_on_doclaynet.json`.

## Curriculum ablation re-scored under COCO (paper Tables 3–4)

The five v5 RT-DETR-l curricula (`baseline`, `tam`, `tam2col`, `3cls`,
`3cls_tam`) were **re-inferred** on the 860-page test and scored under COCO,
because no prediction dumps for the non-tam2col curricula existed on disk.

| item | value |
|---|---|
| Checkpoints | `s3://bec.bdrc.io/models/hff-detection/weights/rtdetr_v5_{baseline,tam,3cls,3cls_tam}_best.pt`; tam2col = `tibetan_book_layout.pt` |
| Inference | `evaluation/predict_testset_dump.py`, `imgsz=1024`, device 0, A10G |
| Class heads | 4-class {0 header,1 text-area,2 footnote,3 footer} for baseline/tam/tam2col; 3-class {0 header-footer,1 text-area,2 footnote} for 3cls/3cls_tam (read from each checkpoint's `model.names`, saved as `model_names.json`) |
| Scoring | `evaluation/score_curriculum.py` → `curriculum_coco.json`; schema (a) canonical for all five, schema (b) DocLayNet-aligned for the 4-class heads only |
| Per-variant infer wall-clock | ~90 s / 860 images |

**Confidence-floor consistency (important).** RT-DETR's decoder is NMS-free and
emits ~300 queries per image; at a very low floor (`conf=0.001`) the extra
near-duplicate low-score boxes become duplicate false positives and *depress*
COCO AP (tam2col canonical AP50 read 0.941 at 0.001 vs 0.977 at 0.05). The
archived finals were dumped at `conf=0.05`, so the curriculum dumps were
regenerated at `conf=0.05`; tam2col then reproduces its Task-1 numbers exactly
(canonical AP50 0.977, AP50-95 0.803). All curriculum numbers quoted use the
`conf=0.05` dumps.

Public DocLayNet detector used on *our* test: **Docling layout-heron**
(`docling-project/docling-layout-heron`, RT-DETRv2-R50). Its published
11-class COCO mAP on DocLayNet v1 is **0.699** (no post-processing); the
shared-3 re-run above gives **0.655** on DocLayNet test.

### Pre-canonical (native, no eval-merge) text-area AP

For the paper's "merged curricula look spectacular" claim. These are a
**re-score** (pycocotools, same protocol as `curriculum_coco.json`, `conf=0.05`
dumps on the 860-page test), **not** Ultralytics training-log val — scored with
the schema-(b) / DocLayNet path, i.e. the canonical envelope-merge is **not**
applied at eval; each model's native `text-area` boxes are matched against the
native (un-merged) test GT.

| curriculum | native text-area (no eval-merge) AP50 / AP50-95 | canonical (envelope-merged) AP50 / AP50-95 |
|---|---:|---:|
| `baseline` (multi-box native text-area) | 0.916 / **0.857** | 0.968 / 0.898 |
| `tam` (merged single-box native text-area) | 0.671 / **0.634** | 0.980 / 0.924 |

Context: on this test set the GT `text-area` is **86% single-box** (737/853
pages have exactly one box; mean 1.32/page), so the un-merged GT is already
close to an envelope.

Reading of the numbers:

- Under a **single consistent, un-merged** scoring, the merged-trained `tam`
  model is **worse** at native text-area (0.634) than the multi-box `baseline`
  (0.857): one page-sized box cannot tightly match the multi-box pages (14% of
  the set) and fits the single-box pages only loosely across the 0.50:0.95 IoU
  sweep.
- Applying the canonical envelope-merge collapses both GT and prediction to one
  box per page, which **rescues** `tam` (0.634 → 0.924) far more than it helps
  `baseline` (0.857 → 0.898). That +0.29 jump for `tam`, purely from the
  evaluation-time merge, is the measurement artifact.
- The paper's figure "0.86 → 0.98" is **not** reproducible as a single
  quantity. `0.86` matches `baseline`'s native no-merge AP50-95 (0.857); `0.98`
  matches `tam`'s **merged** AP**50** (0.980), whose AP50-95 is 0.924. The
  sentence therefore mixes two curricula, two schemas, and two IoU ranges.

Corrected framings (pick one):

- **Within `tam`, before/after the merge (AP50-95):** native text-area
  **0.63 → 0.92** once the page is scored as one envelope.
- **Cross-curriculum, each in its own native granularity (AP50-95):** `baseline`
  no-merge **0.86** vs `tam` merged **0.92** (not 0.98). If AP50 is intended,
  it is **0.92 → 0.98** (`baseline` no-merge AP50 0.916 vs `tam` merged AP50
  0.980).

> **Superseded for the paper by the leak-free v4 re-run** (below). The 860-page
> numbers above are the old development split; the paper reports the v4 833-page
> figures.

### Leak-free v4 curriculum re-run (final architecture)

The label-scheme ablation was re-run on the **served architecture RT-DETR-l** and
the **leak-free v4 split**, scored with the **unified evaluator**. Full tables +
verdict: [`tdla-v4/CURRICULUM.md`](tdla-v4/CURRICULUM.md); JSON
`tdla-v4/curriculum/metrics.json`; scorer `evaluation/score_curricula.py`.

Training (one `g5.xlarge` per non-tam2col variant, A10G, `/opt/pytorch` python,
ultralytics 8.4.135):

```
RTDETR("rtdetr-l.pt").train(data=<curriculum>/data.yaml, imgsz=1024, epochs=100,
    batch=8, patience=20, device=0, seed=0, deterministic=True, amp=True)
# then dump test preds at conf=0.05 (xywhn + per-box conf), upload to
#   s3://bec.bdrc.io/models/hff-detection/tdlav4/eval/rtdetr_tdlav4_<mode>/preds/
# weights + native_ap.json + train.log -> tdlav4/curricula/<mode>/
```

`baseline`/`tam`/`3cls`/`3cls_tam` retrained (early-stopped at epochs ~53–61);
`tam2col` = the production seed0 fine-tune dump (identical recipe), reused so the
ablation's `tam2col` row equals the headline RT-DETR-l row. Curriculum labels for
each variant are built by `data/build_curricula.py` from the `canonical-git@v4`
base and archived at `s3://.../tdlav4/curricula/<mode>_labels.tar.gz`.

Native text-area AP is emitted two ways (both un-merged): **(a)** vs each
variant's own scheme, **(b)** vs a common multi-box GT (baseline's raw labels).
Definition (b) is the artifact yardstick and reproduces the old note's finding
**more strongly** — the baseline−`tam` gap (native AP50-95) widens from 0.223 (860
dev) to 0.307 (v4 leak-free). All three original conclusions hold on the leak-free
split (see `CURRICULUM.md`).

### Unified F1 / mAP scoring (one driver)

The v4 canonical F1/mAP for all 13 systems is produced by a single path —
`run_v4_lit_eval.py` + `literature_metrics.py` — at each system's own
best-mean-F1 operating point, so the headline table and the HT/COTe pass report
one F1 per system. The fix: the canonical text-area envelope is rebuilt from only
the boxes above the operating confidence (`apply_schema(..., conf_floor)`), on a
0.01 grid. Diagnosis + per-system deltas: [`tdla-v4/UNIFIED_SCORING.md`](tdla-v4/UNIFIED_SCORING.md).
The same operating point drives that system's Hidden-Trespass / COTe / LED values.

## Google Document AI Layout Parser (bounding-box bug, now testable)

The blog note said Document AI's Layout Parser tags Tibetan text correctly but
its "bounding-box output is currently broken (an open bug on Google's side)".
As of **2026-08-28** the bug is **version-specific**, so the processor *can* now
be scored against the IoU evaluator by pinning the stable version.

Findings on project `bdrcetextscorpus`, location `us`, processor
`b3c84f91b290e421` (display `tibetan-layout-eval`), `return_bounding_boxes=True`:

| Layout Parser version | boxes returned |
|---|---|
| `pretrained-layout-parser-v1.6-2026-01-13` (RC) | correct type tags, **empty bboxes** |
| `pretrained-layout-parser-v1.6-pro-2025-12-01` (RC) | **empty bboxes** |
| `pretrained-layout-parser-v1.5-2025-08-25` (RC) | **empty bboxes** |
| `pretrained-layout-parser-v1.5-pro-2025-08-25` (RC) | **empty bboxes** |
| `pretrained-layout-parser-v1.0-2024-06-03` (stable) | **populated** normalized boxes |
| `pretrained` (default alias) | **populated** normalized boxes |

So the bug still affects the v1.5/v1.6 release-candidate line; the stable v1.0
returns real boxes. Two other gotchas: Layout Parser **rejects `image/*`**
uploads ("Unsupported mime type for content layout parser"), so each page image
is wrapped in a single-page PDF (`PIL … save(format="PDF")`, mime
`application/pdf`); and Layout Parser has **no footnote type** (footnote →
text-area, so footnote AP/F1 are 0 by construction).

This pass (2026-08-28 UTC):

| item | value |
|---|---|
| Processor version | `pretrained-layout-parser-v1.0-2024-06-03` (pinned) |
| Pages | **860** analyzed, 0 failed (5 pages returned no boxes) |
| Detections | 7520 boxes (468 header, 6921 text-area, 131 footer, 0 footnote) |
| Input | each JPEG wrapped as a single-page PDF |
| Auth | `GOOGLE_APPLICATION_CREDENTIALS` from `ocr-evaluation-benchmark/env.sh`; read-only Document AI `process` calls |
| Cost | ~860 Layout Parser pages (Document AI billing on that GCP project) |

```bash
set -a && source /home/eroux/BUDA/softs/ocr-evaluation-benchmark/env.sh && set +a
python evaluation/google_docai_predict.py \
    --source /home/eroux/azure_di_eval/testset/images/test \
    --out /home/eroux/gdocai_eval \
    --project bdrcetextscorpus --location us \
    --processor-id b3c84f91b290e421 --workers 8
# then run_literature_eval.py picks up google_docai automatically
```

Scored result (schema (a) canonical / (b) DocLayNet-aligned): text-area is
strong (F1 **0.974**, canonical AP50-95 **0.728**) — its many paragraph boxes
collapse cleanly into the body envelope — but header/footer is weak (hf F1
**0.246**, recall 0.17; 57% of header-footer GT absorbed into the body) and
footnote is 0 (no such type). See `google_docai.json`, the row in `RESULTS.md`,
and `figures/google_docai_vs_gt.png` for a GT-vs-prediction overlay.

## Training hyperparameters (original checkpoints)

Each released / reported checkpoint is **one training run**. Ultralytics
default `seed=0`, `deterministic=True`. RF-DETR logged `seed: null`.
We did **not** retrain ≥3 seeds in this pass. The paper's 0.93–0.96
fine-tuned mean-F1 band is across **architectures** (Table 5), not seeds;
treat those gaps as smaller than an unmeasured seed-to-seed std.

### RT-DETR-l (all five v5 curricula)

Recipes: `training/recipes/run_v5_{baseline,tam,tam2col,3cls,3cls_tam}.sh`.
Hardware: **1× NVIDIA A10G 24 GB**. Framework: Ultralytics `RTDETR`.

| knob | value | source |
|---|---|---|
| base weights | `rtdetr-l.pt` | recipe |
| image size | 1024 | recipe |
| epochs | 100 | recipe |
| early-stop patience | 20 | recipe |
| batch | 8 | recipe |
| save-period | 10 | recipe |
| AMP | true (train.py default) | train.py |
| device | 0 | recipe |
| `tam2col` best epoch | 40 (stopped at 60) | model card |
| seed | 0 | Ultralytics `default.yaml` |
| deterministic | True | Ultralytics `default.yaml` |
| optimizer | auto → AdamW for RT-DETR | Ultralytics default |
| lr0 | 0.01 in global default.yaml; RT-DETR trainer uses AdamW-scale LR | not overridden in recipe |
| lrf | 0.01 | default.yaml |
| weight_decay | 0.0005 | default.yaml |
| warmup_epochs | 3.0 | default.yaml |
| momentum / β1 | 0.937 | default.yaml |
| mosaic | 1.0 (close_mosaic=10) | default.yaml |
| mixup / cutmix / copy_paste | 0 | default.yaml |
| hsv_h / hsv_s / hsv_v | 0.015 / 0.7 / 0.4 | default.yaml |
| translate / scale | 0.1 / 0.5 | default.yaml |
| degrees / shear / perspective | 0 | default.yaml |
| fliplr / flipud | 0.5 / 0.0 | default.yaml |
| box / cls / dfl loss gains | 7.5 / 0.5 / 1.5 | default.yaml |

Anything not in the recipe is an Ultralytics default at the version used on
the A10G training box. Recipes live in `training/recipes/`; the Python entry
point is `training/train.py`.

### RF-DETR-L tam2col

Full dump: `evaluation/eval_results/hyperparams/rfdetr_tam2col_training_config.json`
(copied from `s3://bec.bdrc.io/models/hff-detection/rfdetr-tam2col/training_config.json`).

| knob | value |
|---|---|
| model | RFDETRLarge, DINOv2 windowed-small encoder |
| pretrain | `rf-detr-large-2026.pth` (Apache-2.0) |
| resolution | 1008 |
| batch | 8 |
| lr | 1e-4 (encoder 1.5e-4) |
| epochs | 100, early_stopping patience 20, min_delta 0.001 |
| weight_decay | 1e-4 |
| EMA | yes (decay 0.993) |
| AMP | yes |
| seed | **null** (not fixed) |
| num_queries / num_select | 300 |
| GPU | 1× A10G |

### Other fine-tunes (same `tam2col` labels)

DocLayout-YOLO, PP-DocLayout-L, and Docling layout-heron were fine-tuned on
the same `dataset_v5_tam2col` split; prediction dumps and canonical sweeps are
under `s3://bec.bdrc.io/models/hff-detection/{doclayout-yolo,pp-doclayout,docling-heron}-tam2col/`.
They were **not** re-trained here.

## COCO protocol details (this eval)

- Library: pycocotools `COCOeval`, `iouType='bbox'`.
- IoU thresholds: 0.50:0.05:0.95 (10 values).
- Area: all. maxDets: 100. 101-point interpolation.
- Schema (a): remap header+footer → one class (boxes **not** envelope-merged);
  text-area → one envelope per page (max conf); footnote unchanged.
- Schema (b): four native classes, **no** envelope merge.
- Shared-class mean: page-header, page-footer, footnote only.
- Operating-point P/R/F1: greedy match at IoU ≥ 0.5, confidence floor = paper
  conf (same matching as `canon_eval.py`).
- COCO AP uses **all** detections (conf floor 0; Azure score-less boxes get 1.0).

This is slightly **stricter / different interpolation** than the VOC-style AP
in `canon_eval.py`. Canonical mean AP50 for tam2col moves 0.981 (VOC, paper
Table 3) → **0.977** (COCO). That is expected; do not mix the two columns.

## Seeds and the 0.93–0.96 band

The spread across fine-tuned architectures on the same labels and test set
(canonical mean F1 at each model's best mean-F1 conf, paper Table 5 / this
JSON `op_canonical.mean_F1`):

| architecture | mean F1 @ paper conf |
|---|---|
| RT-DETR-l | 0.957 |
| RF-DETR-L | 0.953 |
| PP-DocLayout-L | 0.950 |
| DocLayout-YOLO | 0.947 |
| layout-heron | 0.928 |

Range **0.928–0.957**.

### Measured training-seed noise floor (leak-free v4 split)

This band **is now anchored to a measured seed noise floor**. RT-DETR-l
`tam2col` was retrained **5×** (seeds 0–4), identical recipe, only the RNG seed
differs (`deterministic=True`), and each run's 833-page test dump was scored
with the same canonical schema at conf 0.50. Artefacts:
`evaluation/eval_results/literature/tdlav4/seed_variance.json`;
predictions + weights on
`s3://bec.bdrc.io/models/hff-detection/seed-variance-tdlav4/seed{0..4}/`.

Important: this ran on the **corrected v4 split**, not the original one. The
original series/page split had **book-level leakage** (pages of the same work
in both train and the eval split); the v4 tag rebuilds a **series-level
stratified** split with work-ids disjoint across train/val/test (verified:
all pairwise intersections = 0). So these seed numbers live on a slightly
harder, clean test set and are **not** directly comparable to the paper's
old-split single-run figures — they are the honest noise floor.

| metric (canonical, conf 0.50) | mean | std (pop.) | min–max |
|---|---|---|---|
| **mean-F1** (hf / text-area / footnote) | **0.944** | **0.009** | 0.935–0.959 |
| header-footer F1 | 0.951 | 0.006 | 0.940–0.956 |
| text-area (envelope) F1 | 0.990 | 0.003 | 0.986–0.994 |
| footnote F1 | 0.891 | 0.028 | 0.861–0.938 |
| mean AP50-95 | 0.774 | 0.008 | 0.765–0.786 |
| text-area AP50-95 | 0.932 | 0.013 | 0.920–0.958 |

(Native VOC-style test mAP50-95 per seed, sanity only: 0.732 / 0.721 / 0.722 /
0.726 / 0.725 — spread ≈ 0.004.)

**Consequence for the architecture claim.** The seed-to-seed 1σ on canonical
mean-F1 is ≈ **0.009**, and the min–max from *nothing but the seed* spans
**0.935–0.959** — i.e. a single architecture reproduces almost the entire
0.928–0.957 cross-architecture band just by changing the seed. So gaps like
"RF-DETR-L 0.953 vs RT-DETR-l 0.957" are **well inside seed noise** and are not
distinguishable; the architecture comparison should be framed as *a wash within
the measured noise band*. Footnote F1 is the least stable class (σ ≈ 0.028),
so footnote-specific rankings are the least trustworthy.

The same wash holds *across curricula* for RT-DETR-l: the five v5 curricula
span canonical mean AP50-95 ≈ 0.78–0.81 and best mean-F1 ≈ 0.95–0.96, gaps
within this seed-to-seed uncertainty. tam2col was chosen for two-column
text-area behaviour, not a significant header/footer/footnote AP lead.

Reproduce:
```bash
# per-seed dumps + weights already on S3 (seed jobs on 5 g5.xlarge, one seed each,
# recipe = rtdetr-l.pt, imgsz 1024, epochs 100, patience 20, batch 8, deterministic)
aws s3 sync s3://bec.bdrc.io/models/hff-detection/seed-variance-tdlav4 /home/eroux/seed_variance_tdlav4
aws s3 sync s3://bec.bdrc.io/models/hff-detection/datasets/tdlav4_testset/labels \
    /home/eroux/seed_variance_tdlav4/gt
python evaluation/score_seeds.py \
    --gt-dir /home/eroux/seed_variance_tdlav4/gt \
    --img-dir /home/eroux/seed_variance_tdlav4/gt \
    --dumps-root /home/eroux/seed_variance_tdlav4 \
    --out evaluation/eval_results/literature/tdlav4/seed_variance.json --conf 0.50
```

## Character Error Vector

Not computed. It needs OCR on predicted crops and character-level
ground truth, which the layout dumps do not contain.
