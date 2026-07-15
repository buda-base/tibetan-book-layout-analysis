# Evaluation

All models are scored in the **same canonical 3-class space** so the numbers are
directly comparable, regardless of each model's native label set:

| canonical class | how it is built |
| --- | --- |
| `header-footer` | header + footer, matched **individually** (we don't care about header-vs-footer confusion) |
| `text-area` | **all** text boxes on a page merged into **one envelope** (GT and prediction alike) |
| `footnote` | unchanged |

Matching is greedy at **IoU ≥ 0.5**. Merging text-area into a single envelope is
what makes the comparison apples-to-apples: a model that emits one big text block
and a model that emits many paragraph blocks are scored identically.

## Our model vs. off-the-shelf systems (860-image modern-Tibetan test split)

Off-the-shelf models are swept over confidence and reported at their **best
mean-F1 operating point**. Azure DI returns no confidence, so it has a single,
un-thresholdable operating point.

| Model | header-footer F1 | text-area F1 | footnote F1 | mean F1 |
| --- | --- | --- | --- | --- |
| **Ours — `tam2col` (fine-tuned RT-DETR-L)** @0.25 | 0.894 | 0.988 | **0.957** | **0.946** |
| Surya fast layout (RF-DETR, `surya_layout2`) @0.30 | 0.895 | 0.989 | 0.439 | 0.774 |
| Azure AI Document Intelligence (`prebuilt-layout`) | 0.625 | 0.989 | 0.404 | 0.673 |
| DocLayout-YOLO (DocStructBench) @0.20 | 0.657 | 0.886 | 0.000 | 0.515 |

Headline findings:

- **Text area is easy.** Surya and Azure both match our fine-tuned model (~0.99).
- **Headers/footers.** Surya's off-the-shelf layout detector is genuinely strong
  (0.895, on par with our fine-tuned model). Azure and DocLayout are much weaker.
- **Footnotes are the differentiator.** Only the fine-tuned model handles them
  (0.957). Nothing off-the-shelf clears 0.44, and DocLayout scores **zero** —
  DocStructBench has no page-footnote class. Footnotes are exactly the
  OCR-critical structure generic document models don't understand for Tibetan
  books, which is the core justification for fine-tuning.

Per-class precision / recall / meanIoU and the full confidence sweeps are in
[`eval_results/`](eval_results/) (`*_canon_eval.txt`, `*_sweep.txt`).

## RF-DETR vs. RT-DETR (they are not the same model)

The names collide but the architectures don't:

| | **RT-DETR** (what we fine-tuned) | **RF-DETR** (what Surya's fast layout uses) |
| --- | --- | --- |
| Author | Baidu (2023) | Roboflow (2025) |
| Backbone | CNN (ResNet / HGNetv2) | DINOv2 self-supervised ViT |
| Design | hybrid CNN encoder + IoU-aware query selection | single-scale deformable DETR, NAS-tuned on top of DINOv2 |
| In this repo | `rtdetr-l` via Ultralytics, fine-tuned on our data | `datalab-to/surya_layout2`, used off-the-shelf |

So Surya is **not** running the model we trained — it runs a different detector
(RF-DETR) that Datalab already fine-tuned on document layout (DocLayNet-style, 20
classes). We report Surya on that exact model (`surya_layout2`); there is no
separate "vanilla" model to benchmark.

**Could we fine-tune Surya's layout detector?** Yes — `surya_layout2` is a
byte-for-byte vendored copy of Roboflow's open-source RF-DETR, so Roboflow's
`rfdetr` package (`RFDETRLarge().train(dataset_dir=..., ...)`, COCO **or** YOLO
format) can fine-tune it, and the result loads back into
`surya.fast_layout.FastLayoutPredictor(checkpoint=...)`. Caveat: Surya *code* is
Apache-2.0 but the *weights* are modified AI Pubs OpenRAIL-M (free for
research / nonprofits / <$5M orgs; a derivative inherits those use
restrictions). A cleanly-licensed release would fine-tune from Roboflow's own
Apache-2.0 RF-DETR base instead.

## Reproducing

```bash
# Surya fast layout (rf-detr) — pure torch, no vLLM/docker
SURYA_INFERENCE_AUTOSTART=false python surya_predict.py \
    --source <images> --out surya_pred --batch 16

# DocLayout-YOLO DocStructBench (batched to stay memory-safe)
python doclayout_predict.py --weights <docstructbench.pt> \
    --source <images> --out doclayout_pred --batch 8 --imgsz 1024

# Azure AI Document Intelligence (needs AZURE_DI_ENDPOINT / AZURE_DI_KEY)
python azure_di_predict.py --source <images> --out azure_pred --workers 10

# Score any of the above in canonical space (text-area merged, header+footer combined)
#   args: <pred_labels_dir> <dataset_dir> [remap] [iou] [conf_floor]
python eval_pred_files.py surya_pred/labels <dataset> "0:0,1:1,2:2,3:0" 0.5 0.3
```

Prediction label dumps for all three baselines are archived at
`s3://bec.bdrc.io/models/hff-detection/off-the-shelf-eval/`.
