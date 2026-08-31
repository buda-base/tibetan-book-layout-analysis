# Served RT-DETR-l tam2col — per-class thresholds + unified seed variance (v4)

Both items below are on the **served model = RT-DETR-l tam2col**, leak-free v4
833-page test, canonical space, unified scorer (`literature_metrics`; per-conf
text-area envelope, IoU>=0.5 greedy matching, COCO mAP 0.50:0.05:0.95).
Scripts: `evaluation/sweep_served_thresholds.py` (Task A),
`evaluation/score_seed_variance_unified.py` (Task B).
JSON: `served_thresholds.json`, `seed_variance_unified.json`.

---

## Task A — per-class confidence sweep (threshold recipe, re-established leak-free)

seed 0 checkpoint. Grid swept 0.01..0.95; readable points below. For **text-area**
the sweep is shown BOTH natively (per predicted box, no merge) and in the
canonical envelope space; the paper's "0.955 -> 0.98 @0.55" was the **native** sweep.

**header-footer (canonical):**

| conf | P | R | F1 | tp/fp/fn |
|--:|--:|--:|--:|:--|
| 0.10 | 0.839 | 0.962 | 0.897 | 1485/284/58 |
| 0.25 (default) | 0.936 | 0.957 | 0.946 | 1477/101/66 |
| 0.40 | 0.956 | 0.953 | 0.954 | 1470/68/73 |
| **0.60 (served)** | **0.964** | **0.944** | **0.954** | 1456/55/87 |
| 0.80 | 0.971 | 0.922 | 0.946 | 1423/43/120 |
| 0.90 | 0.980 | 0.564 | 0.716 | 870/18/673 |

Best-F1 = 0.9548 @ conf 0.41. **Raising h/f from 0.25 to 0.60 buys +0.028
precision for −0.014 recall** (F1 0.946 → 0.954). F1 is flat (0.950–0.955) across
0.30–0.70, so 0.60 is a near-optimal, precision-favouring choice. (Note: the
leak-free tam2col model is better calibrated than the old dev-split number — at
0.25 h/f precision is 0.936, not the ~0.83 "collapse"; that collapse now only
appears below ~0.15.)

**text-area — native (per-box, no merge):**

| conf | P | R | F1 | tp/fp/fn |
|--:|--:|--:|--:|:--|
| 0.25 (default) | 0.987 | 0.994 | 0.990 | 832/11/5 |
| **0.55 (served)** | **0.994** | 0.994 | 0.994 | 832/5/5 |
| 0.80 | 0.996 | 0.992 | 0.994 | 830/3/7 |

Best-F1 = 0.9946 @ conf 0.75. **Raising text-area from 0.25 to 0.55 buys +0.007
precision (0.987 → 0.994) at zero recall cost**, dropping ~6 spurious boxes. Same
direction as the old "0.955 → 0.98" finding but a smaller gain, because the
tam2col model's native precision is already high at 0.25.

**text-area — canonical envelope:** flat at **0.999** F1 across 0.10–0.80 —
threshold-insensitive, because the envelope inherits the max confidence of its
boxes (confirms the original observation; the merge hides the native precision
gain, which is why the served threshold is tuned on the native sweep).

**footnote (canonical):**

| conf | P | R | F1 | tp/fp/fn |
|--:|--:|--:|--:|:--|
| 0.10 | 0.731 | 1.000 | 0.844 | 38/14/0 |
| **0.25 (served)** | 0.804 | **0.974** | 0.881 | 37/9/1 |
| 0.40 | 0.841 | 0.974 | 0.902 | 37/7/1 |
| 0.80 | 0.878 | 0.947 | 0.911 | 36/5/2 |
| 0.87 (best-F1) | 0.947 | 0.947 | 0.947 | 36/2/2 |
| 0.90 | 1.000 | 0.816 | 0.899 | 31/0/7 |

**Caveat — footnote differs from the old recipe.** The v4 test has only **38
footnote GT boxes**, so F1 is dominated by a handful of thresholdable false
positives: best-F1 is 0.947 @ conf 0.87 (not the served 0.25). But footnote
**recall** is 1.00 at conf ≤0.10 and stays 0.974 through 0.70, then falls; F1 is
otherwise flat (0.88–0.91) over a wide range. The served **0.25 favours recall**
(catch essentially every footnote, accept a few FPs) — an operational choice, not
the F1-optimum. If the paper wants an F1-optimal footnote threshold on this test
it is ~0.85–0.90; if it wants the served recall-safe recipe it is 0.25. State
which, and note the 38-box sample makes the footnote curve noisy.

**Single global vs per-class thresholds.** The global best-mean-F1 conf is
**0.74** (mean F1 **0.9585**), matching `tab:oppoints`. Per-class tuning
(h/f 0.41, text-area 0.15, footnote 0.87) gives mean F1 **0.9670** — so using one
global knob costs **0.0085 mean F1** vs per-class tuning. The served recipe
(0.60 / 0.55 / 0.25) is a precision/recall-operational compromise, not the F1
argmax.

---

## Task B — 5-seed variance under the UNIFIED scorer (supersedes the old 0.50-conf band)

The old band (canonical mean F1 **0.944 ± 0.010**, footnote **0.891 ± 0.031**,
mAP **0.773 ± 0.009**) was computed at a fixed conf 0.50, so it was **not**
comparable to the unified headline (0.959). Re-scored all five existing seed
dumps with the unified scorer (each seed at its own best-mean-F1 conf); no
retraining.

| seed | conf | mean F1 | h/f F1 | text-area F1 | footnote F1 | canon mAP@[.5:.95] | shared mAP@[.5:.95] |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 0.74 | 0.9585 | 0.9518 | 0.9988 | 0.9250 | 0.7859 | 0.6500 |
| 1 | 0.76 | 0.9468 | 0.9429 | 0.9976 | 0.9000 | 0.7660 | 0.6344 |
| 2 | 0.81 | 0.9625 | 0.9431 | 0.9958 | 0.9487 | 0.7645 | 0.6378 |
| 3 | 0.78 | 0.9648 | 0.9493 | 0.9976 | 0.9474 | 0.7728 | 0.6438 |
| 4 | 0.73 | 0.9704 | 0.9497 | 0.9994 | 0.9620 | 0.7781 | 0.6464 |

**Unified band (mean ± sample sd, min / max, n=5):**

| metric | mean ± sd | min | max | (old fixed-0.50 band) |
|---|---|--:|--:|---|
| canonical mean F1 | **0.9606 ± 0.0088** | 0.9468 | 0.9704 | 0.944 ± 0.010 |
| header-footer F1 | 0.9474 ± 0.0041 | 0.9429 | 0.9518 | 0.951 ± 0.006 |
| text-area F1 | 0.9978 ± 0.0014 | 0.9958 | 0.9994 | 0.990 ± 0.003 |
| footnote F1 | 0.9366 ± 0.0244 | 0.9000 | 0.9620 | 0.891 ± 0.031 |
| canonical mAP@[.5:.95] | 0.7735 ± 0.0089 | 0.7645 | 0.7859 | 0.773 ± 0.009 |
| shared mAP@[.5:.95] | 0.6425 ± 0.0064 | 0.6344 | 0.6500 | (not in old band) |

**Recommendation for the paper.** Make the headline the **mean, 0.961 ± 0.009**
(unified), not seed 0's 0.959 — seed 0 sits just below the mean, and the band now
matches the scorer that produces every other number. canonical mAP@[.5:.95] is
conf-independent, so its mean (0.7735) is unchanged from the old band (good
sanity check). Footnote remains the volatile class (sd 0.024).
