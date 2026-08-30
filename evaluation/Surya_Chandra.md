# Off-the-shelf Surya 2 & Chandra 2 on the tdlav4 test set

Evaluation of Datalab's current flagship document models — **Surya 2's layout
VLM** and **Chandra 2** — for header / footer / footnote / text-area layout
detection on Tibetan book pages, compared against our fine-tuned detector.

- **Test set:** the leak-free **`tdlav4` 833-page test split** (Hub tag `v4`,
  series-disjoint from train/val).
  - Images: `images/test` inside
    `s3://bec.bdrc.io/models/hff-detection/tdlav4/dataset/dataset_tdlav4_tam2col.tar.gz`
    (833 real test JPEGs; stem set md5-verified against the GT below).
  - Ground truth (YOLO labels): `/home/eroux/seed_variance_tdlav4/gt` (833),
    exposed to the scorer as `/home/eroux/azure_di_eval/tdlav4_testset/labels/test`.
- **Date:** 2026-08-29/30
- **Hardware:** a private-subnet `g5.xlarge` (NVIDIA A10G, 24 GB), instance
  `i-027853f8ad991d5a0` (`surya-vlm-eval`, `us-east-1`), driven over AWS SSM. Now
  **stopped**.

> Both models were run on all **833** v4 test images (not the older v2 860-page
> split). Only 116 of the 833 v4 test pages overlap the v2 test set, so this is a
> genuine re-inference, not a re-score.

## Results

Same canonical pipeline as every other system (`evaluation/eval_pred_files.py`
via `canon_sweep_preds.py`): IoU ≥ 0.5, text-area merged into one per-page
envelope, header + footer combined into a single `header-footer` class (matched
individually), remap `0:0,1:1,2:2,3:0`, best-mean-F1 operating point.

| system (model actually run) | header-footer | text-area | footnote | **mean F1** |
|---|---|---|---|---|
| **Ours** — RT-DETR-l `tdlav4_tam2col` (5-seed mean) | 0.951 | 0.990 | 0.891 | **0.944** |
| Docling layout-heron `tdlav4_tam2col` (ours, fine-tuned) | 0.944 | 0.998 | 0.809 | 0.917 |
| **Surya 2 VLM** — `datalab-to/surya-ocr-2` | 0.865 | 0.993 | 0.474 | **0.777** |
| Azure DI prebuilt-layout | 0.611 | 0.990 | 0.252 | 0.618 |
| **Chandra 2** — `datalab-to/chandra-ocr-2` | 0.699 | 0.695 | 0.385 | **0.593** |
| Google DocAI Layout Parser | 0.151 | 0.967 | 0.000 | 0.373 |
| AWS Textract Layout | 0.232 | 0.835 | 0.000 | 0.356 |

(Ours / commercial rows are the v4 numbers from
`evaluation/eval_results/tdla-v4/README.md`, on the identical 833-page test.)

### Surya 2 VLM per-class detail (best-mean-F1 @ conf 0.00, IoU ≥ 0.5)

| class | P | R | F1 | meanIoU | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| header-footer | 0.871 | 0.859 | 0.865 | 0.687 | 1326 | 196 | 217 |
| text-area | 0.993 | 0.993 | 0.993 | 0.925 | 823 | 6 | 6 |
| footnote | 0.390 | 0.605 | 0.474 | 0.797 | 23 | 36 | 15 |

### Chandra 2 per-class detail (un-thresholded, IoU ≥ 0.5)

| class | P | R | F1 | meanIoU | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| header-footer | 0.896 | 0.572 | 0.699 | 0.682 | 883 | 102 | 660 |
| text-area | 0.762 | 0.638 | 0.695 | 0.897 | 529 | 165 | 300 |
| footnote | 0.714 | 0.263 | 0.385 | 0.839 | 10 | 4 | 28 |

## Key findings

- **Surya 2's large VLM (≈650 M) lands at 0.777**, well below our fine-tuned
  detector (0.944) and even our fine-tuned heron (0.917). It reads header/footer
  and text-area well (0.87 / 0.99) but only 0.47 on footnotes — the hardest and
  most valuable class for this corpus. It does beat the commercial layout APIs
  overall, largely on header/footer geometry.
- **Chandra 2 is markedly worse (0.593)** for a structural reason: it has no
  layout-only mode. The `ocr_layout` prompt makes it fully OCR the page and derive
  blocks from the transcription. On Tibetan — which it effectively cannot read —
  it loops until it hits the decode-token cap (avg **3,196 / max 4,000 tokens per
  page**), which truncates the bottom of the page and collapses recall
  (header-footer R 0.572, text-area R 0.638, footnote R 0.263). It sits between
  Google/Textract and Azure.
- **Neither Datalab flagship closes the footnote gap** (ours 0.891 vs ≤ 0.47).

For reference, the earlier **v2 860-page** run gave nearly identical figures —
Surya 2 VLM **0.776** (0.878 / 0.988 / 0.463), Chandra 2 **0.551** (0.702 / 0.670
/ 0.281) — so the split choice does not change the story; these v4 numbers are the
canonical, leak-free ones.

## How each was run

### Surya 2 VLM (`surya-ocr-2`)
- Package `surya-ocr==0.22.1`; `LayoutPredictor` + `SuryaInferenceManager`
  (auto-spawns the vLLM backend).
- Backend `vllm/vllm-openai:v0.20.1`, bfloat16, MTP speculative decoding
  (`VLLM_ENABLE_MTP=True`, 2 tokens), `VLLM_MAX_MODEL_LEN=18000`,
  `SURYA_GUIDED_LAYOUT=True`, `SURYA_MAX_TOKENS_LAYOUT=3072`.
- Deps as imported: torch `2.13.0+cu130`, transformers `5.16.1`,
  huggingface-hub `1.29.0`, openai `2.54.0`; device `NVIDIA A10G`.
- 833/833 pages, **0 layout errors**; dropped labels not in our schema:
  `Picture` ×87, `Table` ×2.
- **Confidence is page-level** (mean decode probability), so it is not
  meaningfully thresholdable — the reported point is un-thresholded, like Azure DI.
- Script: `evaluation/surya_vlm_predict.py` (maps Surya VLM labels →
  `0 header / 1 text-area / 2 footnote / 3 footer`, writes `cls cx cy w h conf`).

### Chandra 2 (`chandra-ocr-2`)
- Package `chandra-ocr==0.2.0`; `InferenceManager(method="vllm")`, prompt
  `ocr_layout`, blocks parsed from the returned HTML (`data-label` / `data-bbox`).
- Backend `vllm/vllm-openai:v0.17.0` launched via `chandra_vllm --gpu a10`
  (arch `Qwen3_5ForConditionalGeneration`, hybrid mamba, bfloat16,
  `max-model-len 18000`, prefix caching, `max-num-seqs 16`, MTP off).
- Two Tibetan-specific adjustments in our runner:
  - **Retries disabled** (`--max-retries 0`): Chandra's repeat-token detector
    false-fires on Tibetan script and otherwise forces up to 6 full re-decodes
    per page.
  - **Decode capped at 4,000 tokens** (`--max-output-tokens 4000`) with
    per-image concurrent writes, to bound the runaway/looping pages.
- 833/833 pages, **0 errors**; dropped labels not in our schema: `Image` ×84,
  `Table` ×1. No per-block confidence → rows written without a confidence column
  (single un-thresholded operating point). Full run ≈ 2 h on the A10G.
- Script: `evaluation/chandra_vlm_predict.py`.

Both prediction scripts write a `run_meta.json` recording the *actually imported*
package version, checkpoint id, backend and settings — do not trust the pip pin
without it.

## Artifacts

| system | S3 | local |
|---|---|---|
| Surya 2 VLM (v4) | `s3://bec.bdrc.io/models/hff-detection/off-the-shelf-eval-tdlav4/surya_vlm/` | `/home/eroux/azure_di_eval/surya_vlm_v4_pred/` |
| Chandra 2 (v4) | `s3://bec.bdrc.io/models/hff-detection/off-the-shelf-eval-tdlav4/chandra_vlm/` | `/home/eroux/azure_di_eval/chandra_vlm_v4_pred/` |

Each S3 prefix contains `labels/` (833 YOLO `.txt` predictions) and
`run_meta.json`. Local dirs additionally contain the confidence sweep
(`*_v4_sweep.txt`). Prediction scripts are staged at
`s3://bec.bdrc.io/models/hff-detection/scripts/{surya_vlm_predict.py,chandra_vlm_predict.py}`.
The superseded v2 860-page dumps live under
`s3://bec.bdrc.io/models/hff-detection/off-the-shelf-eval/{surya_vlm,chandra_vlm}/`.

## Reproduce the scoring

```bash
cd evaluation
# GT dataset dir: labels/test -> the verified 833-page v4 GT
#   ln -sfn /home/eroux/seed_variance_tdlav4/gt \
#           /home/eroux/azure_di_eval/tdlav4_testset/labels/test

# Surya 2 VLM
python canon_sweep_preds.py \
  /home/eroux/azure_di_eval/surya_vlm_v4_pred/labels \
  /home/eroux/azure_di_eval/tdlav4_testset \
  /home/eroux/azure_di_eval/surya_vlm_v4_pred/surya_vlm_v4_sweep.txt \
  "0:0,1:1,2:2,3:0" 0.5

# Chandra 2
python canon_sweep_preds.py \
  /home/eroux/azure_di_eval/chandra_vlm_v4_pred/labels \
  /home/eroux/azure_di_eval/tdlav4_testset \
  /home/eroux/azure_di_eval/chandra_vlm_v4_pred/chandra_vlm_v4_sweep.txt \
  "0:0,1:1,2:2,3:0" 0.5
```

## Not done / notes

- The **hosted Datalab API "accurate" mode** (the fully closed flagship) was
  **not** run — it needs a `DATALAB_API_KEY` and ~$4–9 for 833 pages. The local
  Chandra result above is a reasonable floor; the API would likely score somewhat
  higher but hits the same "OCRs Tibetan it can't read" wall.
- `run_literature_eval.py` scores every system on the older **v2 860-page** GT, so
  its `surya_vlm_ots` / `chandra_ots` rows are the v2 numbers (0.776 / 0.551),
  internally consistent with the other v2 rows there. The **v4** numbers in this
  file and in `eval_results/tdla-v4/README.md` are the canonical ones.
- The evaluation `g5.xlarge` is **stopped, not terminated** (reusable). Its
  instance-store NVMe scratch is gone on stop, but all predictions are on S3.
