# tdla-v4 — label-scheme (curriculum) ablation on the final architecture

The label-scheme ablation, re-run on the **served production architecture**
(RT-DETR-l) and the **leak-free v4 split**, so it is no longer a development-split
table. All five variants share the exact v4 fine-tune recipe (Ultralytics
`RTDETR`, `rtdetr-l.pt`, imgsz 1024, 100 epochs / patience 20, batch 8, seed 0,
deterministic) and are scored on the **v4 833-page test** with the **unified
evaluator** (`literature_metrics`, per-conf text-area envelope, 0.01 grid — the
same path as the 13-system table; see [`UNIFIED_SCORING.md`](UNIFIED_SCORING.md)).

`tam2col` = the production RT-DETR-l seed0 fine-tune (identical recipe), reused so
the ablation and the headline table agree on that row. The other four
(`baseline`, `tam`, `3cls`, `3cls_tam`) were retrained from scratch for this
ablation. Data: `score_curricula.py`; per-variant JSON in
[`curriculum/metrics.json`](curriculum/metrics.json).

The five label variants:

- **baseline** — 4 classes, text-area left as-is (possibly several boxes/page).
- **tam** — text-area boxes merged into one envelope/page.
- **tam2col** — like `tam`, but genuine two-column pages keep two boxes (~2% of pages).
- **3cls** — header + footer merged into a single `header-footer` training class.
- **3cls_tam** — both merges.

## Canonical AP (v4 833-page test, AP50 / AP50-95)

| variant | header-footer | text-area | footnote | mean AP50 | mean AP50-95 |
|---|---:|---:|---:|---:|---:|
| baseline | 0.954 / 0.567 | 0.945 / 0.742 | 0.956 / 0.814 | 0.952 | 0.707 |
| tam | 0.953 / 0.569 | 0.990 / 0.935 | 0.992 / 0.850 | 0.978 | 0.785 |
| **tam2col** (production) | 0.945 / 0.568 | 0.990 / **0.958** | 0.987 / 0.832 | 0.974 | **0.786** |
| 3cls | 0.949 / 0.549 | 0.932 / 0.758 | 0.979 / 0.814 | 0.953 | 0.707 |
| 3cls_tam | 0.954 / 0.573 | 0.979 / 0.915 | 0.948 / 0.793 | 0.960 | 0.760 |

## Best-F1 operating points (canonical, single global best-mean-F1 conf)

| variant | conf | header-footer F1 | text-area F1 | footnote F1 | mean F1 |
|---|---:|---:|---:|---:|---:|
| baseline | 0.75 | 0.947 | 0.990 | 0.937 | 0.958 |
| tam | 0.78 | 0.935 | 0.998 | 0.905 | 0.946 |
| **tam2col** (production) | 0.74 | 0.952 | 0.999 | 0.925 | **0.959** |
| 3cls | 0.57 | 0.943 | 0.999 | 0.894 | 0.945 |
| 3cls_tam | 0.71 | 0.951 | 0.999 | 0.864 | 0.938 |

Recall is uniformly high across all variants (0.90–1.00); precision sets the
operating point. `tam2col` has the highest mean F1 (0.959) and the highest
footnote F1 among the merged-header variants; the two header+footer-merged
variants (`3cls`, `3cls_tam`) have the two lowest footnote F1 (0.894 / 0.864).

## Native (no eval-merge) text-area AP — the measurement artifact

Two definitions of "native" text-area AP (both un-merged; AP50 / AP50-95):

| variant | (a) vs OWN scheme | (b) vs COMMON multi-box GT | canonical (envelope-merged) |
|---|---:|---:|---:|
| baseline | 0.827 / 0.717 | 0.827 / **0.717** | 0.945 / 0.742 |
| tam | 1.000 / 0.962 | 0.468 / **0.410** | 0.990 / 0.935 |
| tam2col | 0.990 / 0.964 | 0.479 / **0.422** | 0.990 / 0.958 |
| 3cls | 0.867 / 0.744 | 0.867 / 0.744 | 0.932 / 0.758 |
| 3cls_tam | 0.990 / 0.965 | 0.471 / 0.407 | 0.979 / 0.915 |

- **(a) vs own scheme** measures how well a model reproduces its *own* training
  target. Envelope-trained variants (`tam`/`tam2col`/`3cls_tam`) trivially hit
  ~0.96 — a single big box is easy to place. This is the inflation the canonical
  evaluator is designed to neutralise.
- **(b) vs a common multi-box GT** (baseline's raw un-merged labels) is the
  honest cross-curriculum yardstick — every model's raw text-area boxes scored
  against the *same* native GT. Here the envelope predictors collapse to
  **0.41–0.42** AP50-95 (a single page box cannot tightly match multi-box pages
  across the 0.50:0.95 IoU sweep), well below the multi-box `baseline`/`3cls`
  (**0.72 / 0.74**).
- Applying the canonical envelope-merge then **rescues** the envelope models —
  `tam` 0.410 → 0.935, `tam2col` 0.422 → 0.958 — far more than it moves
  `baseline` (0.717 → 0.742). That jump, purely from the eval-time merge, **is**
  the measurement artifact.

Leak-free vs the old dev-split note (`REPRODUCIBILITY.md`, 860-page split:
baseline 0.857 / tam 0.634): the artifact **reproduces and is stronger** — the
baseline−`tam` native gap (common-GT AP50-95) widens from **0.223** to
**0.307** on the leak-free split.

## Do the three original conclusions hold on the leak-free split?

**Yes — all three hold; two are stronger, one has a thinner margin.**

1. **Keep header and footer as separate training classes — HOLDS.** Merging them
   (`3cls`, `3cls_tam`) gives no gain on the combined class (header-footer AP50-95
   0.549 / 0.573 vs 0.567–0.573 for the 4-class variants) and drags footnote down
   to the two lowest values of the whole table (F1 0.894 / 0.864 vs 0.905–0.937).
   Richer supervision + a loss-free post-hoc merge still wins.
2. **Training on merged text-area genuinely helps canonical text-area — HOLDS,
   stronger.** Canonical text-area AP50-95 rises baseline 0.742 → tam 0.935 →
   tam2col 0.958 (+0.19 / +0.22), a larger lift than the old dev split (+0.03),
   because the leak-free baseline is lower. The core caveat also holds: the lift
   is a scoring artifact of collapsing to one box, not better line detection
   (see native table above).
3. **`tam2col` is the best production choice; two-column benefit is invisible to
   canonical scoring — HOLDS, thinner margin.** `tam2col` has the top mean F1
   (0.959), top mean AP50-95 (0.786), and top canonical text-area localization
   (0.958). It no longer has the clear mean-AP50 lead it showed on the dev split
   (tam 0.978 ≥ tam2col 0.974): with only ~2% two-column pages and a canonical
   metric that cannot see column reading-order, `tam2col ≈ tam` canonically by
   construction — exactly the caveat stated originally. `tam2col` remains the
   served scheme because its advantage is in reading-order correctness, which the
   canonical metric does not reward.

## Reproduce

```
# retrain the 4 non-tam2col variants (per box), then:
evaluation/.venv_eval/bin/python evaluation/score_curricula.py --pull
```

`--pull` fetches the four dumps from `s3://.../tdlav4/eval/rtdetr_tdlav4_<mode>/`;
`tam2col` reuses the production seed0 dump. Weights + `native_ap.json` +
`train.log` per variant under `s3://.../tdlav4/curricula/<mode>/`. See
[`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) for the training-box recipe.
