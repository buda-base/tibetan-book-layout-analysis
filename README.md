# Tibetan Book Layout Analysis

Training code, data pipeline, evaluation, and reproducible recipes for a
**document-layout detector for modern Tibetan books**. The model finds four
structural regions on every page image — **header**, **text-area**,
**footnote**, and **footer** — as a preprocessing step for OCR and etext
production.

- **Model (weights + usage):** [BDRC/Tibetan-Modern-Book-Layout-Detection](https://huggingface.co/BDRC/Tibetan-Modern-Book-Layout-Detection) on the Hugging Face Hub
- **Dataset:** [BDRC/TDLA-Training-Dataset-v2](https://huggingface.co/datasets/BDRC/TDLA-Training-Dataset-v2) (gated, fair-use)
- **Write-up:** [`BLOGPOST.md`](BLOGPOST.md) — the full story: data cleaning, the model bake-off, and the evaluation lessons

## The result in one paragraph

We compared several detectors (YOLO26s/m, DocLayout-YOLO, RT-DETR-l) and several
label formulations, evaluating everything in a **canonical 3-class space** so the
comparison is fair regardless of how each model was trained. The winner is
**`tam2col`**: an RT-DETR-l trained on 4 classes, with text-area boxes merged into
one envelope per page *except* on genuine two-column pages (where it keeps one box
per column). It reaches **canonical mean AP50 0.981**, best-in-class text-area and
footnote localization, and handles two-column layouts correctly.

One serving detail matters: header/footer should be thresholded higher than the
other classes (**conf ≈ 0.60** vs ≈ 0.25), which lifts their precision from ~0.83
to ~0.95 at almost no recall cost. See the model card and `inference/infer.py`.

## Repository layout

```
data/         dataset construction & QC
  convert_ndjson.py    Ultralytics-Platform NDJSON export -> YOLO folders (downloads pixels)
  build_dataset.py     pool batches, dedup, leakage-free volume-level splits, class alignment
  build_curricula.py   derive tam / 3cls / 3cls_tam / tam2col label variants (incl. 2-column heuristic)
  audit_data.py        geometric/logical consistency audit (bad boxes, impossible orderings, ...)
  analyze_splits.py    split statistics & leakage checks
  build_release.py     package the gated Hugging Face dataset (coords clamped to [0,1])
  measure_skew.py      empirical page-skew measurement (AABB vs deskew decision)

training/
  train.py             single entry point (RT-DETR / YOLO / DocLayout), --save-period, patience
  recipes/             the exact commands for every v5 run (baseline, tam, 3cls, 3cls_tam, tam2col)
  setup_doclayout.sh   installs DocLayout-YOLO in an isolated venv

evaluation/
  canon_eval.py        canonical per-class AP (fair cross-curriculum comparison)
  canon_sweep.py       confidence sweep -> per-class P/R/F1 CSVs (used to pick thresholds)
  eval_ckpts.py        native per-class mAP / P / R / F1 on the test split
  merged_eval.py       region-level (merged-box) scoring
  eval_worst.py        surface the worst confusions for manual review
  eval_results/        the actual v5 numbers + PR-curve plot behind the blog post

inference/
  infer.py             run the model with the recommended PER-CLASS thresholds
  predict.py           batch pre-annotation -> YOLO labels for re-import into the platform
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Inference

```bash
# download the weights from the Hugging Face model repo, then:
python inference/infer.py --weights tibetan_book_layout.pt --source page.jpg
```

### Reproduce training

The dataset is built from Ultralytics-Platform NDJSON exports, then trained on a
single A10G (24 GB) GPU. The winning recipe:

```bash
# 1. build the base 4-class dataset (see data/ scripts + BLOGPOST.md)
# 2. derive the tam2col label variant (text-area merged, two-column pages kept)
python data/build_curricula.py dataset tam2col dataset_tam2col
# 3. train RT-DETR-l
bash training/recipes/run_v5_tam2col.sh
```

All five v5 recipes live in `training/recipes/`; the canonical comparison that
selected the winner is produced by `evaluation/canon_eval.py` and
`evaluation/canon_sweep.py`.

## Classes

| id | name | notes |
|----|------|-------|
| 0 | header | running title / marginal text at top or side |
| 1 | text-area | main body text (one box per column) |
| 2 | footnote | notes below the text area |
| 3 | footer | folio numbers / marginal text at bottom or side |

Header-vs-footer confusion is irrelevant downstream, so the headline metrics
combine them into a single `header-footer` class as loss-free post-processing.

## License

Code in this repository is released under the [MIT License](LICENSE). The page
images in the dataset are distributed on a fair-use basis with **no content
license** — you are responsible for your own rights analysis (see the
[dataset card](https://huggingface.co/datasets/BDRC/TDLA-Training-Dataset-v2)).

## Acknowledgements

Developed by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io)
for the BDRC Etext Corpus. Annotations were produced and consolidated on the
Ultralytics platform.
