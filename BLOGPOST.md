# Detecting headers, footers, footnotes and text areas in Tibetan books

*A practical write-up of how we built a document-layout detector for scans of
modern Tibetan books — the data pipeline, the model bake-off, and a few
counter-intuitive lessons about evaluation.*

## The problem

This work targets **modern Tibetan books** (not traditional pecha). For
downstream OCR and etext production we need to know, for every page image, where
the four structural regions are:

- **text-area** — the main body text (can be split into several blocks, and on
  some pages laid out in **two columns**);
- **header** — running title / marginal text at the top or side;
- **footer** — folio numbers and marginal text at the bottom or side;
- **footnote** — notes below the main text area.

The goal: a robust object detector that finds these four regions, trained on a
clean, releasable dataset.

## The data

Annotations were produced on the [Ultralytics platform](https://platform.ultralytics.com/)
across several batches. A key constraint surfaced immediately: the platform only
supports **axis-aligned bounding boxes** (AABB). Slightly rotated scans therefore
get slightly loose boxes. We decided this was acceptable (the looseness is small
and consistent) rather than switching to oriented boxes.

Building a trustworthy dataset took more work than training the models:

1. **Pooling & dedup.** Several NDJSON exports (`tdla-v10`, `tdlabatch3reviewed`,
   `batch4`, an augmented `batch-11`) were pooled. We ignored the platform's own
   train/val/test splits and deduplicated by pixel MD5.
2. **Leakage-free splitting.** We split at the **volume** level (all pages of a
   book/volume stay in one split), stratified so footnote-bearing volumes are
   spread across train/val/test. **Augmented images were forced into train only**
   so the val/test sets stay clean and representative.
3. **Consistency audit.** A geometric/logical audit flagged likely annotation
   mistakes: near-duplicate boxes (IoU > 0.9), conflicting classes on the same
   region, impossible orderings (footer above header, header below the text
   centre, footnote not below the text area), out-of-bounds and degenerate boxes,
   plus unusually tiny or corner-placed headers/footers. Flagged pages (≈260)
   were exported back to the platform, manually corrected, and re-imported.
4. **Release.** The cleaned dataset was published as a gated dataset on the
   Hugging Face Hub with a fair-use disclaimer (all coordinates clamped to
   `[0,1]` to avoid rounding issues on import).

Final dataset: **8,325 images** (6,751 train / 714 val / 860 test), ~25,500 boxes.

## Round 1 — which architecture?

We trained several detectors at 1024 px and evaluated on the held-out test split:
YOLO26s/m, **DocLayout-YOLO** (a YOLOv10 fork pre-trained on document structure),
and **RT-DETR-l**.

| model | test mAP50 | test mAP50-95 |
|---|---|---|
| DocLayout-YOLO (best) | 0.936 | 0.745 |
| **RT-DETR-l (best)** | **0.954** | **0.754** |

**RT-DETR-l won** and became our reference model.

### A useful detour: which checkpoint?

We compared the "elbow" checkpoint (just after the loss stops dropping quickly)
against the converged, validation-selected `best.pt`. Two clean findings:

- DocLayout kept *improving on the test set* all the way to its val-best — no
  marginal-tail overfitting; the elbow checkpoint underfit (footnote AP 0.76 →
  0.95 with more training).
- RT-DETR, trained a few epochs *past* its val-best, **regressed on test**
  (footer mAP50-95 0.63 → 0.52).

The literature (early stopping — Prechelt 1998; deep double descent — Nakkiran
et al. 2019; SWA — Izmailov et al. 2018) says neither the elbow nor the last
epoch is reliably best; the robust choice is the **validation-selected checkpoint
with weight averaging (EMA)**, which is exactly what the Ultralytics stack saves.
So we simply keep `best.pt`.

## Round 2 — does relabelling help? (and a lesson in fair evaluation)

We then asked whether changing the *label formulation* helps. Three RT-DETR
variants, all on the same test split:

- **baseline** — 4 classes, text-area left as-is (possibly several boxes/page).
- **tam** — 4 classes, but all text-area boxes **merged** into one envelope/page.
- **3cls** — header and footer **merged** into a single `header-footer` class.
- **3cls_tam** — both of the above.

Taken at face value, the curricula looked great (tam text-area mAP50-95 jumped
0.86 → 0.98!). But that is a **measurement artifact**: a single merged box is
trivial to localize, which inflates per-class mAP. To compare fairly we built a
**canonical evaluator** that maps *every* model into one common space, applying
the merges as post-processing so the metric is identical for all:

- **header+footer combined** into one class, with boxes matched *individually*
  (relabelling — NOT enveloping, since a header at the top and a footer at the
  bottom would merge into a page-sized box);
- **text-area merged** into one envelope per page (done as post-processing for
  models not trained that way);
- **footnote** unchanged.

In this fair, apples-to-apples space (all five variants retrained on the final
`v5` dataset, 860 test images):

| model | header-footer | text-area | footnote | mean AP50 | mean AP50-95 |
|---|---|---|---|---|---|
| baseline | 0.969 / 0.690 | 0.975 / 0.902 | 0.970 / 0.818 | 0.971 | 0.803 |
| tam | 0.960 / 0.687 | 0.985 / 0.929 | 0.970 / 0.807 | 0.972 | 0.808 |
| **tam2col** | 0.965 / 0.683 | **0.988** / 0.910 | **0.991** / 0.824 | **0.981** | 0.806 |
| 3cls | 0.959 / 0.676 | 0.967 / 0.869 | 0.984 / 0.808 | 0.970 | 0.784 |
| 3cls_tam | 0.964 / 0.705 | 0.981 / 0.929 | 0.968 / 0.796 | 0.971 | 0.810 |

*(each cell is AP50 / AP50-95)*

**The headline result: on the merged canonical metric the curricula are all within
noise of each other on AP50-95 (0.78–0.81).** Three genuine signals survive the
fair comparison:

- **Keep header and footer as *separate* training classes.** Models trained with
  them distinct score as high or higher on the combined class than the model
  trained with them pre-merged — distinct supervision plus lossless post-hoc
  combining wins, and never hurts.
- **Training on merged text-area genuinely helps text-area** specifically
  (0.902 → 0.929 AP50-95), at the cost of needing a higher confidence threshold.
- **`tam2col` has the best AP50 and the best text-area/footnote localization**,
  and — crucially — the canonical metric *understates* it, because it merges the
  two columns of a two-column page back into a single envelope (see below).

### Precision / recall (the 3 classes we actually care about)

Header-vs-footer confusion is irrelevant downstream, so we evaluate the three
classes **text-area, header+footer, footnote** and sweep the confidence
threshold. Best-F1 operating points (F1 @conf, canonical space):

| class | baseline | tam | **tam2col** | 3cls | 3cls_tam |
|---|---|---|---|---|---|
| header-footer | 0.954 @.57 | 0.954 @.67 | 0.950 @.65 | 0.943 @.56 | 0.952 @.67 |
| text-area | 0.922 @.67 | 0.938 @.95 | **0.952 @.89** | 0.900 @.80 | 0.892 @.95 |
| footnote | 0.957 @.63 | 0.946 @.25 | **0.957 @.23** | 0.954 @.70 | 0.946 @.46 |

Recall is uniformly high (0.90–1.00); **precision is the differentiator**.

![Canonical 3-class PR curves](evaluation/eval_results/pr_canonical.png)

### Setting the confidence threshold for header/footer

The one operating-point trap worth calling out: at the naive default of
`conf=0.25`, header/footer **precision collapses to ~0.83** because the detector
happily over-predicts small marginal boxes. Sweeping the threshold on `tam2col`
shows how sharp the fix is:

| conf | P | R | F1 |
|---|---|---|---|
| 0.25 (default) | 0.83 | 0.97 | 0.894 |
| 0.45 | 0.93 | 0.96 | 0.941 |
| 0.55 | 0.95 | 0.95 | 0.947 |
| **0.60** | **0.95** | **0.95** | **0.950** |
| 0.65 | 0.96 | 0.94 | 0.950 |

Raising the header/footer threshold from 0.25 to **~0.60** buys +0.12 precision
for only −0.02 recall — F1 0.894 → 0.950. Text-area (a single merged envelope) is
essentially threshold-insensitive, and footnote is best left *low* (~0.25, where
recall is 1.00). So the practical recipe is **per-class thresholds**
(header/footer ≈ 0.60, text-area ≈ 0.25, footnote ≈ 0.25); if a single global
value is required, **0.45** is the best compromise.

## Two-column pages

Some pages lay the body text out in two columns, and merging those into one box
is wrong. We added a **heuristic** that keeps two text-area boxes when a page is a
genuine two-column layout: the text-area boxes must split into a left/right group
that are **horizontally disjoint** (overlap < 20% of the smaller column's width)
and **vertically co-extensive** (share ≥ 30% of the smaller column's height).
On the full dataset this flags ~175 pages (~2%). We trained a `tam2col` variant
(text-area merged *except* on two-column pages), which turned out to be the
strongest model overall.

The payoff is only visible on the model's **own** label schema, because the
canonical evaluator re-merges the two columns into one envelope and cannot reward
the split. On the native 4-class test set, keeping the columns lifts text-area
localization dramatically:

| model | text-area mAP50 | text-area mAP50-95 |
|---|---|---|
| baseline (raw multi-box) | 0.923 | 0.864 |
| **tam2col** | **0.994** | **0.980** |

That is the real reason to prefer `tam2col`: it gets two-column pages right —
returning one clean box per column — without regressing anything else.

## Recommendation & reproducibility

Our production choice is **`tam2col`** — a 4-class RT-DETR-l that keeps header and
footer as separate training classes (combined losslessly in post-processing) and
merges text-area into one envelope *except* on genuine two-column pages, where it
returns one box per column. It has the best canonical AP50 (0.981) and the best
text-area and footnote scores, ties everything else on header/footer, and is the
only variant that handles two-column layouts correctly. Serve it with per-class
confidence thresholds (**header/footer ≈ 0.60**, text-area ≈ 0.25,
footnote ≈ 0.25), or a single global **0.45** if simplicity is preferred.

**Artifacts.** The trained model, with usage and the per-class thresholds, is on
the Hugging Face Hub at
[BDRC/Tibetan-Modern-Book-Layout-Detection](https://huggingface.co/BDRC/Tibetan-Modern-Book-Layout-Detection);
the cleaned, audited dataset is released (gated) at
[BDRC/TDLA-Training-Dataset-v2](https://huggingface.co/datasets/BDRC/TDLA-Training-Dataset-v2);
and all training/evaluation code and recipes live in the
[tibetan-book-layout-analysis](https://github.com/buda-base/tibetan-book-layout-analysis)
GitHub repository (raw weights + metrics are also mirrored to
`s3://bec.bdrc.io/models/hff-detection/`).

*Tooling note:* every metric in this post is reproducible with three small,
self-contained scripts — `canon_eval.py` (fair per-class AP), `canon_sweep.py`
(confidence sweep / PR curves), and `merged_eval.py` (region-level merged boxes) —
all schema-aware so they work for the 3-class and 4-class models alike.
