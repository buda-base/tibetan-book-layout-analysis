# Off-the-shelf Surya 2 & Chandra 2 on the tdlav4 test set

Evaluation of Datalab's current flagship document models — **Surya 2's layout
VLM** and **Chandra 2** — for header / footer / footnote / text-area layout
detection on Tibetan book pages, compared against our fine-tuned detector.

- **Test set:** `tdlav4` (860 Tibetan book-page images, the v2 test split).
  - Images: `s3://bec.bdrc.io/models/hff-detection/eval/v2-testset/images/test`
  - Ground truth (YOLO labels): `/home/eroux/azure_di_eval/testset/labels/test`
- **Date:** 2026-08-29
- **Hardware:** a freshly launched private-subnet `g5.xlarge` (NVIDIA A10G, 24 GB),
  instance `i-027853f8ad991d5a0` (`surya-vlm-eval`, `us-east-1b`), driven over AWS
  SSM. Now **stopped**.

## Results

All systems scored with the **same canonical pipeline** (`evaluation/eval_pred_files.py`
via `canon_sweep_preds.py`): IoU ≥ 0.5, text-area merged into one per-page
envelope, header + footer combined into a single `header-footer` class, remap
`0:0,1:1,2:2,3:0`.

| system (model actually run) | header-footer | text-area | footnote | **mean F1** |
|---|---|---|---|---|
| **Ours** — `tam2col` RT-DETR-L | 0.894 | 0.988 | 0.957 | **0.946** |
| Surya fast layout — RF-DETR (`datalab-to/surya_layout2`) | 0.895 | 0.989 | 0.439 | 0.774 |
| **Surya 2 VLM** — `datalab-to/surya-ocr-2` | 0.878 | 0.988 | 0.463 | **0.776** |
| **Chandra 2** — `datalab-to/chandra-ocr-2` | 0.702 | 0.670 | 0.281 | **0.551** |

### Chandra 2 per-class detail (un-thresholded, IoU ≥ 0.5)

| class | P | R | F1 | meanIoU | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| header-footer | 0.899 | 0.576 | 0.702 | 0.691 | 860 | 97 | 633 |
| text-area | 0.740 | 0.612 | 0.670 | 0.894 | 522 | 183 | 331 |
| footnote | 0.667 | 0.178 | 0.281 | 0.788 | 8 | 4 | 37 |

## Key findings

- **Surya 2's large VLM (≈650 M) is a dead heat with the tiny RF-DETR
  fast-layout model** (0.776 vs 0.774): marginally better footnotes, marginally
  worse headers/footers. Both sit ~0.17 mean-F1 below our fine-tuned detector and
  both miss the majority of footnotes.
- **Chandra 2 is markedly worse (0.551)** for a structural reason: it has no
  layout-only mode. The `ocr_layout` prompt makes it fully OCR the page and derive
  blocks from the transcription. On Tibetan — which it effectively cannot read —
  it loops until it hits the decode-token cap (avg **3,300 / max 4,000 tokens per
  page**), which truncates the bottom of the page and collapses recall across the
  board (header-footer R 0.576, text-area R 0.612, footnote R 0.178 → only 8 TP).
- **Neither closed/flagship model closes the footnote gap** (ours 0.957 vs
  ≤ 0.46), which is the hardest and most valuable class for this corpus.

## How each was run

### Surya 2 VLM (`surya-ocr-2`)
- Package `surya-ocr==0.22.1`; `LayoutPredictor` + `SuryaInferenceManager`
  (auto-spawns the vLLM backend).
- Backend `vllm/vllm-openai:v0.20.1`, bfloat16, MTP speculative decoding
  (`VLLM_ENABLE_MTP=True`, 2 tokens), `VLLM_MAX_MODEL_LEN=18000`,
  `SURYA_GUIDED_LAYOUT=True`, `SURYA_MAX_TOKENS_LAYOUT=3072`.
- Deps as imported: torch `2.13.0+cu130`, transformers `5.16.1`,
  huggingface-hub `1.29.0`, openai `2.54.0`; device `NVIDIA A10G`.
- 857 pages processed (+3 smoke) = 860/860, **0 layout errors**; dropped labels
  not in our schema: `Picture` ×81, `Table` ×11.
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
- 850 pages processed (+10 smoke) = 860/860, **0 errors**; dropped labels not in
  our schema: `Image` ×55, `Table` ×4. No per-block confidence → rows written
  without a confidence column (single un-thresholded operating point). Full run
  ≈ 2 h on the A10G.
- Script: `evaluation/chandra_vlm_predict.py`.

Both prediction scripts write a `run_meta.json` recording the *actually imported*
package version, checkpoint id, backend and settings — do not trust the pip pin
without it.

## Artifacts

| system | S3 | local |
|---|---|---|
| Surya 2 VLM | `s3://bec.bdrc.io/models/hff-detection/off-the-shelf-eval/surya_vlm/` | `/home/eroux/azure_di_eval/surya_vlm_pred/` |
| Chandra 2 | `s3://bec.bdrc.io/models/hff-detection/off-the-shelf-eval/chandra_vlm/` | `/home/eroux/azure_di_eval/chandra_vlm_pred/` |

Each S3 prefix contains `labels/` (860 YOLO `.txt` predictions) and
`run_meta.json`. Local dirs additionally contain the confidence sweep
(`*_sweep.txt`). Prediction scripts are staged at
`s3://bec.bdrc.io/models/hff-detection/scripts/{surya_vlm_predict.py,chandra_vlm_predict.py}`.

## Reproduce the scoring

```bash
cd evaluation

# Surya 2 VLM
python canon_sweep_preds.py \
  /home/eroux/azure_di_eval/surya_vlm_pred/labels \
  /home/eroux/azure_di_eval/testset \
  /home/eroux/azure_di_eval/surya_vlm_pred/surya_vlm_sweep.txt \
  "0:0,1:1,2:2,3:0" 0.5

# Chandra 2
python canon_sweep_preds.py \
  /home/eroux/azure_di_eval/chandra_vlm_pred/labels \
  /home/eroux/azure_di_eval/testset \
  /home/eroux/azure_di_eval/chandra_vlm_pred/chandra_vlm_sweep.txt \
  "0:0,1:1,2:2,3:0" 0.5
```

## Not done / notes

- The **hosted Datalab API "accurate" mode** (the fully closed flagship) was
  **not** run — it needs a `DATALAB_API_KEY` and ~$4–9 for 860 pages. The local
  Chandra result above is a reasonable floor; the API would likely score somewhat
  higher but hits the same "OCRs Tibetan it can't read" wall.
- These two rows are **not yet wired into** `run_literature_eval.py` (no
  `surya_vlm_ots` / `chandra_ots` entries), so they don't appear in the generated
  results tables/figures yet.
- The evaluation `g5.xlarge` is **stopped, not terminated** (reusable). Its
  instance-store NVMe scratch is gone on stop, but all predictions are on S3.
