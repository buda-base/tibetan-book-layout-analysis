# We just wanted to delete the headers

*How a small "solved" preprocessing step for Tibetan OCR turned into a proper
model bake-off — and a few counter-intuitive lessons about how you measure
success.*

## The short version

We build OCR for scanned modern Tibetan books, and before any recognition
happens we need to strip running headers, footers, and footnotes off the page
so they don't get glued into the middle of a sentence. We assumed this was
solved — document layout detection is a mature field, with open-source
detectors, cloud APIs, and now vision-language models happy to draw boxes
around anything. It wasn't: every off-the-shelf and commercial system we
tested, including the ones with genuinely good aggregate scores, had a habit
of quietly stitching the clutter it missed straight into the body text —
which is worse for us than a low score. So we built our own benchmark,
trained our own detector, and are releasing both: a layout-detection model and
its training dataset for modern Tibetan books, open on the Hugging Face Hub,
with every number in this post reproducible from the code on GitHub.

## The problem

We are building a high-accuracy OCR system for scanned **modern Tibetan books**.
Our goal downstream is clean etext: the actual body of the work, with none of the
clutter that lives around it. On a Tibetan page that clutter is very real —
running headers march along the top or side margin, folio numbers and
marginal notes sit in the footer, and footnotes crowd the bottom of the block. If
you feed all of that to an OCR engine and concatenate the output, you get a text
stream where a chapter title interrupts a sentence and a page number lands in the
middle of a word. So before we OCR anything, we want to find and set aside the
headers and footers, and isolate the footnotes from the main text.

Four regions per page, in our vocabulary:

- **text-area** — the main body text (sometimes several blocks, and on some pages
  laid out in **two columns**);
- **header** — running title / marginal text at the top or side;
- **footer** — folio numbers and marginal text at the bottom or side;
- **footnote** — notes below the main text area.

Downstream we mostly care about three things: keep the text-area, drop
header+footer, and peel off footnotes. (Whether a marginal box is a "header" or a
"footer" barely matters to us — a distinction that will come back later.)

Simple enough, or so we thought. Our first instinct was *not* to train
anything. Document layout analysis is a mature field; there are open-source
detectors, cloud APIs, and now vision-language models that will happily draw
boxes around anything you ask. This felt like a solved problem that we could
solve with an API key.

Colleagues on the OpenPecha team had already taken a first pass
at answering that, benchmarking nine open-source and VLM systems against a
Tibetan layout test set
([write-up here](https://forum.openpecha.org/t/how-well-do-existing-layout-detection-models-handle-tibetan-books/610)).
It was a useful first signal, and not an encouraging one:

- The best model, [Surya](https://github.com/datalab-to/surya), needed a
  post-processing merge to reach 82% mAP@0.5 on headers/footers — its raw
  output scored 34%.
- Footnotes were close to a universal miss (0–1% AP for several systems).
- The general-purpose VLMs nailed the body text and then collapsed on the
  margins, hallucinating headers along the way.

That told us this was worth taking seriously, but it stopped short of two
things we actually needed: it didn't test the commercial document-AI
incumbents, and it scored aggregate boxes rather than asking how these
systems would behave once wired into a real OCR pipeline. So we picked up
that thread ourselves. This is the story of how we found out — and why the
interesting part turned out to be not the training, but the measuring.

## Building a benchmark we could trust

If we were going to answer this properly we needed a large, leakage-free,
audited Tibetan layout dataset of our own — and a full list of everything
worth testing against it, commercial incumbents included.

The dataset was the part that actually took the time. Annotations were
produced on the [Ultralytics platform](https://platform.ultralytics.com/)
across several batches. The first thing we hit was a modeling constraint: the
platform only supports **axis-aligned bounding boxes**. On a slightly rotated
scan, an axis-aligned box is a little loose. We decided to live with it — the
looseness is small and consistent — rather than jump to oriented boxes.

The rest was discipline:

- **Leakage-free splitting.** We split at the **volume** level — every page of a
  given book stays in one split — and stratified so that the rare footnote-bearing
  volumes are represented in train, val, and test. Augmented images were confined
  to **train only**, so validation and test are clean, original scans.
- **A consistency audit.** A geometric/logical pass flagged likely annotation
  mistakes: near-duplicate boxes (IoU > 0.9), conflicting classes on the same
  region, physically impossible orderings (a footer above a header, a header below
  the middle of the text, a footnote that isn't below the text), out-of-bounds and
  degenerate boxes, and suspiciously tiny or corner-stuck headers/footers. About
  260 flagged pages went back for manual correction.
- **Release.** The cleaned dataset is published (gated, fair-use) on the Hugging
  Face Hub, with every coordinate clamped to `[0,1]`.

Final tally: **8,325 images** (6,751 train / 714 val / 860 test), ~25,500 boxes.

While the dataset was taking shape, we also mapped the field of commercial
document-AI incumbents — the services whose entire pitch is structured layout
with named roles:

- **[Azure AI Document Intelligence](https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence)**
  is the strongest candidate. Its layout model assigns paragraph *roles* that
  include `pageHeader`, `pageFooter`, `pageNumber` and — unusually — an
  explicit **`footnote`** role. On paper it targets exactly our four regions.
- **[AWS Textract](https://aws.amazon.com/textract/)** (Layout) returns
  `HEADER`, `FOOTER`, and `PAGE_NUMBER` blocks, but has **no footnote class**.
- **[Google Document AI](https://cloud.google.com/document-ai)** and
  **[ABBYY FineReader](https://pdf.abbyy.com/finereader-pdf/)** both
  reconstruct document structure and separate running heads/footers, though
  footnote handling is weaker or geared toward reflowing Latin-script PDFs.

Our prior was that none of these would have actually seen anything like a
Tibetan book: they're trained overwhelmingly on Western business documents —
invoices, contracts, English books — and Tibetan uchen script, dense pointed
text, and margin-hugging running titles are wildly out of distribution. On top
of that, these are closed, per-page-priced services: for a corpus that will run
to millions of pages, and for a project whose output is meant to be an **open,
reproducible dataset and model**, we can neither afford them nor retune them to
Tibetan conventions.

But a prior isn't a measurement. So we settled on one canonical way to score
everything — our own models and the off-the-shelf/commercial candidates alike
— against the same 860-page held-out test set:

- **header + footer combined** into one class, with boxes matched
  *individually* (a relabelling, **not** an envelope — a header at the top and
  a footer at the bottom must not merge into a page-sized box);
- **text-area merged** into one envelope per page, applied as post-processing
  even for models that don't natively output one box per page;
- **footnote** left untouched;
- a match counts at **IoU ≥ 0.5**.

With that scoring in place, we could finally put the prior to the test.

## Off-the-shelf and commercial systems look promising

We tested six systems on our own 860-page test set, scored with the
canonical evaluator above:

- **Azure AI Document Intelligence** (`prebuilt-layout`) — the commercial
  incumbent with, on paper, the perfect schema. It returns no confidence
  scores, so it has a single, un-thresholdable operating point.
- **AWS Textract** (Layout) — returns block-level confidence, but sweeping it
  never beat simply keeping every box, so in practice it too has one
  operating point worth reporting.
- **Surya** — the best performer in the OpenPecha benchmark. Modern Surya
  routes layout through a 650M vision-language model, but it also ships a
  lightweight, pure-PyTorch layout detector (`surya_layout2`, an
  **[RF-DETR](https://github.com/roboflow/rf-detr)** — Roboflow's
  DINOv2-based detector, *not* to be confused with the Baidu
  **[RT-DETR](https://github.com/lyuwenyu/RT-DETR)** architecture we later
  fine-tuned ourselves). We benchmark that detector at its best-F1 confidence.
- **[DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)**
  (DocStructBench) — a strong open-source document detector, swept to its
  best-F1 confidence.
- **[PP-DocLayout-L](https://huggingface.co/PaddlePaddle/PP-DocLayout-L)**
  (PaddleOCR) — Baidu's document-layout RT-DETR variant, pretrained on
  Chinese document structure, swept to its best-F1 confidence.
- **[Docling layout-heron](https://huggingface.co/docling-project/docling-layout-heron)**
  (IBM) — an RT-DETRv2 detector with explicit `page_header`, `page_footer`,
  and `footnote` classes, swept to its best-F1 confidence.

Everyone gets the easy question right and diverges on the hard ones (best-F1
operating points, canonical 3-class F1):

| system | header-footer | text-area | footnote | **mean F1** |
|---|---|---|---|---|
| Surya fast layout (RF-DETR, off-the-shelf) | 0.895 | 0.989 | 0.439 | 0.774 |
| Azure AI Document Intelligence | 0.625 | 0.989 | 0.404 | 0.673 |
| PP-DocLayout-L (off-the-shelf) | 0.485 | 0.865 | 0.667 | 0.672 |
| Docling layout-heron (off-the-shelf) | 0.481 | 0.992 | 0.397 | 0.624 |
| DocLayout-YOLO (DocStructBench) | 0.657 | 0.886 | 0.000 | 0.515 |
| AWS Textract | 0.186 | 0.893 | 0.000 | 0.360 |

A few things jump out, and most of them are encouraging. **Text-area looks like
a solved problem** — Azure, Surya, Docling heron, PP-DocLayout-L, and even AWS
Textract all tie or nearly tie at ~0.87–0.99, so if all you want is "where is
the body text," an API key or an off-the-shelf detector would do. The real
surprise is that **Surya's off-the-shelf RF-DETR is a genuinely good
header/footer detector** (0.895) — much better than the OpenPecha benchmark's
raw-Surya numbers suggested. And footnote support already exists in more
places than you'd guess — Azure, Docling heron, and PP-DocLayout-L all ship an
explicit footnote class, and PP-DocLayout-L's off-the-shelf footnote F1
(0.667) is the best score in the table. AWS Textract is the outlier that
matches its documented schema exactly: no footnote class, and (unlike its
peers) it also just doesn't find most headers and footers on a Tibetan page
(0.186). Still, on this table as a whole, an argument for "just use what's out
there" looks almost plausible.

## Until we look at contamination

An F1 number hides *how* a model fails, and for our pipeline the *how* is
everything. We crop the predicted text-area and send it to OCR, so there are two
very different ways to miss a header:

1. **Absorb it** — the text-area box grows to swallow the header, and its text
   gets OCR'd as body text. This silently corrupts the etext. **This is the
   failure we cannot tolerate.**
2. **Drop it cleanly** — no header box is emitted, but the text-area stays tight,
   so the body text is still clean. We lose the header, but we don't poison the
   text.

So we measured, for every ground-truth header/footer and footnote, whether a
model that *failed to detect it* had **folded that region into its text-area
envelope** (≥ 50% of the region inside the predicted body block):

| system | header/footer detected | …folded into text-area | footnote detected | …folded into text-area |
|---|---|---|---|---|
| Surya fast layout | 91% | 1.2% | 56% | 16% |
| Azure Document Intelligence | 58% | **12%** | 44% | **22%** |
| DocLayout-YOLO | 67% | 6% | 0% | 20% |
| AWS Textract | 15% | **17%** | 0% | **56%** |
| Docling layout-heron (off-the-shelf) | 53% | 3% | 58% | 9% |
| PP-DocLayout-L (off-the-shelf) | 44% | 12% | 53% | **42%** |

*("folded into text-area" = share of **all** ground-truth regions of that type
that the system both missed **and** buried inside its OCR text block.)*

This is the twist. **Azure would fold roughly one in eight running heads /
folio numbers, and more than one in five footnotes, straight into the body
text.** On a corpus headed for millions of pages, that is systematic, silent
contamination of exactly the etext we are trying to keep clean — precisely the
problem we set out to solve. Its headline text-area F1 of 0.989 looks
reassuring right up until you notice *what* that text block contains. AWS
Textract is worse on both counts at once: it finds only 15% of headers/footers
in the first place, and 17% of *all* of them end up folded into the OCR
text — plus more than half of all footnotes, since it has no footnote class
to catch them with.

Docling layout-heron and PP-DocLayout-L round out the picture, and PP-DocLayout-L
delivers the cleanest illustration of the whole point of this section.
Recall it had the *best* off-the-shelf footnote F1 in our first table (0.667) —
on paper, the strongest footnote detector we tested. Look at what it does with
the footnotes it misses, though: of the 47% of footnotes it doesn't detect,
almost all of them (42% of *all* footnotes) get folded straight into the body
text. A good aggregate footnote score and heavy contamination on the footnotes
it doesn't catch turn out to be entirely compatible facts about the same model.

Surya is the one genuine bright spot: as a header/footer detector it barely
contaminates at all (1.2% absorbed), so for header/footer removal alone it
would be a defensible off-the-shelf choice. But it still buries **16% of
footnotes** in the body text, and its strongest configuration is the
heavyweight VLM stack; the weights are also OpenRAIL-M-licensed (free for
research and nonprofits, paid above a revenue threshold), which complicates an
open release.

So: **could we have used Azure, or AWS Textract? No.** Azure can find the
text, but hides the clutter *inside* it when it misses; AWS Textract simply
doesn't find most of the clutter in the first place. Surya could plausibly
handle header/footer stripping, but not footnotes, and not under a license
we'd want for an open release. The one-sentence answer to the question we
started with is: the off-the-shelf systems are good enough to find your text
and not good enough to protect it.

Which meant training our own — and the two questions that turned out to
matter most were not the ones we expected.

## Finding the right training path

We picked **RT-DETR-l** as our starting architecture — the same detector
family behind PP-DocLayout-L above, which had already impressed us in the
off-the-shelf comparison. The first real question wasn't architecture, though
— it was *how you frame the labels*. We trained five RT-DETR variants on the
same split:

- **baseline** — 4 classes, text-area left as-is (possibly several boxes/page).
- **tam** — 4 classes, but all text-area boxes **merged** into one envelope/page.
- **tam2col** — like `tam`, except on genuine two-column pages, where two
  boxes are kept instead of one. A single merged envelope reads fine for one
  column, but on a two-column page it makes the OCR read straight across
  both columns instead of finishing the left one first; a simple
  disjointness heuristic flags the ~175 pages (~2% of the dataset) where
  that applies.
- **3cls** — header and footer **merged** into a single `header-footer` class.
- **3cls_tam** — both `tam` and `3cls`.

Taken at face value, the merged curricula looked spectacular — `tam`'s
text-area mAP50-95 leapt from 0.86 to 0.98! But that is a **measurement
artifact**: a single big merged box is trivial to localize, so the per-class
mAP inflates for free. If we'd stopped there we'd have fooled ourselves — and
it's exactly this trap that the canonical evaluator defined earlier (merges
applied as post-processing, never as training-time relabelling) is built to
avoid: it forces every model, ours included, into the same evaluation space.

In this apples-to-apples space (all five variants retrained on the final dataset,
860 test images):

| model | header-footer | text-area | footnote | mean AP50 | mean AP50-95 |
|---|---|---|---|---|---|
| baseline | 0.969 / 0.690 | 0.975 / 0.902 | 0.970 / 0.818 | 0.971 | 0.803 |
| tam | 0.960 / 0.687 | 0.985 / 0.929 | 0.970 / 0.807 | 0.972 | 0.808 |
| **tam2col** | 0.965 / 0.683 | **0.988** / 0.910 | **0.991** / 0.824 | **0.981** | 0.806 |
| 3cls | 0.959 / 0.676 | 0.967 / 0.869 | 0.984 / 0.808 | 0.970 | 0.784 |
| 3cls_tam | 0.964 / 0.705 | 0.981 / 0.929 | 0.968 / 0.796 | 0.971 | 0.810 |

*(each cell is AP50 / AP50-95)*

Once the playing field is level, the dramatic gaps evaporate: on AP50-95 the
curricula are all within noise of each other (0.78–0.81). But three real signals
survive:

- **Keep header and footer as *separate* training classes.** Models given the
  distinction score as high or higher on the *combined* class than the model
  trained on pre-merged labels. Richer supervision plus a loss-free post-hoc merge
  beats throwing the distinction away up front.
- **Training on merged text-area genuinely helps text-area** (0.902 → 0.929
  AP50-95) — but it needs a higher confidence threshold to pay off.
- The canonical metric, by construction, **can't see** one thing that turned out
  to matter a lot: two-column pages. `tam2col` — the only variant that keeps
  two boxes there — is the strongest model of the whole lot (best mean AP50
  above, 0.981), which was a genuinely good surprise given we mostly added it
  just to check whether the two-column case was worth the bother.

### Precision and recall — the three classes we actually care about

Since header-vs-footer confusion is irrelevant to us, we evaluate the three
meaningful classes — **text-area, header+footer, footnote** — and sweep the
confidence threshold. Best-F1 operating points (canonical space):

| class | baseline | tam | **tam2col** | 3cls | 3cls_tam |
|---|---|---|---|---|---|
| header-footer | 0.954 @.57 | 0.954 @.67 | 0.950 @.65 | 0.943 @.56 | 0.952 @.67 |
| text-area | 0.922 @.67 | 0.938 @.95 | **0.952 @.89** | 0.900 @.80 | 0.892 @.95 |
| footnote | 0.957 @.63 | 0.946 @.25 | **0.957 @.23** | 0.954 @.70 | 0.946 @.46 |

Recall is uniformly high (0.90–1.00); **precision is the differentiator**.

### The confidence threshold is not one number

The single most useful practical lesson: **the right confidence threshold differs
by class**, and the Ultralytics default of 0.25 is wrong for most of them.

At `conf=0.25`, header/footer **precision collapses to ~0.83** — the detector
cheerfully over-predicts small marginal boxes. Sweeping the threshold shows how
cheap the fix is:

| conf | P | R | F1 |
|---|---|---|---|
| 0.25 (default) | 0.83 | 0.97 | 0.894 |
| 0.45 | 0.93 | 0.96 | 0.941 |
| 0.55 | 0.95 | 0.95 | 0.947 |
| **0.60** | **0.95** | **0.95** | **0.950** |
| 0.65 | 0.96 | 0.94 | 0.950 |

Nudging header/footer from 0.25 to **~0.60** buys +0.12 precision for a −0.02
recall cost — F1 0.894 → 0.950.

Text-area is subtler, and nearly tricked us. In the *canonical* (merged-envelope)
space it looks threshold-insensitive, because the envelope inherits the highest
confidence among its boxes. But a **native per-class sweep** (scoring each
text-area box on its own) reveals a real, cheap precision gain: raising text-area
from 0.25 to **~0.55** lifts precision 0.955 → 0.98 for a 0.002 recall cost,
quietly dropping ~23 low-confidence spurious boxes. Footnote, meanwhile, is best
left *low* (~0.25, recall 1.00) — its few false positives are high-confidence and
can't be thresholded away without losing genuine footnotes.

The practical recipe: **per-class thresholds** — header/footer ≈ 0.60, text-area
≈ 0.55, footnote ≈ 0.25 — or a single global **0.50** if you need one knob.

## Finding the right architecture

`tam2col`'s labelling scheme settled the training path. Now that we had a
training recipe that actually worked, the natural next question was: how much
does the underlying architecture matter? We'd started with RT-DETR-l as a
reasonable default — a close relative, PP-DocLayout-L, had already impressed
us in the off-the-shelf comparison — but the honest way to test that choice
was to fine-tune the strongest other candidates from that comparison on the
exact same `tam2col` recipe and labels, and see who could keep up.

Surya's fast layout detector *is* Roboflow's RF-DETR (DINOv2 backbone,
Apache-2.0 weights) — a different architecture from Baidu's RT-DETR, but
fine-tunable with the same `dataset_v5_tam2col` labels. We fine-tuned that,
plus Docling layout-heron (RT-DETRv2, IBM), PP-DocLayout-L, and
DocLayout-YOLO (DocStructBench base) — all on the same recipe.

All were scored on the same 860-page test split with the canonical evaluator
(confidence swept per model; best mean-F1 operating point):

| system | header-footer | text-area | footnote | **mean F1** | best conf |
|---|---|---|---|---|---|
| **RT-DETR-l fine-tuned (Ours)** | 0.949 | 0.998 | 0.933 | **0.960** | 0.50 |
| **RF-DETR-L fine-tuned** (Roboflow) | 0.963 | 0.994 | 0.923 | **0.960** | 0.30 |
| Docling heron fine-tuned | 0.940 | 0.934 | 0.923 | 0.932 | 0.05 |
| PP-DocLayout-L fine-tuned | 0.954 | 0.995 | 0.920 | **0.956** | 0.75 |
| DocLayout-YOLO fine-tuned | 0.948 | 0.996 | 0.897 | 0.947 | 0.30 |

The headline: **fine-tuning RF-DETR (Roboflow) on our Tibetan data matches
our RT-DETR-l fine-tuned baseline exactly** — mean F1 0.960, with footnote F1
0.923. That is a useful licensing datapoint: Roboflow's Apache-2.0 RF-DETR base
is a viable alternative to Ultralytics RT-DETR-l for this task, at least on our
benchmark.

Docling heron fine-tuning is a clear step up from off-the-shelf (0.624 → 0.932) and
gets footnotes to 0.923, but it lands just below our RT-DETR-l fine-tuned baseline
on text-area and header-footer. PP-DocLayout-L fine-tuning reaches **0.956 mean
F1** (0.672 off-the-shelf → 0.956), essentially matching that baseline; its
first run was interrupted by disk exhaustion, then restarted with checkpoint
pruning and early-stopped at epoch 25. **DocLayout-YOLO** (DocStructBench base,
same recipe) early-stopped at epoch 61 (best @ epoch 41) and reaches **0.947 mean
F1** — strong on text-area (0.996) but slightly below on footnotes (0.897).

The RF-DETR-L and DocLayout-YOLO checkpoints are published alongside the
primary RT-DETR-l release — see [Our models](#our-models) below. Checkpoints,
prediction dumps, and full confidence sweeps for the Docling and PP-DocLayout
runs aren't published alongside this post, but are available on request.

The convergence holds on contamination too, which is the check that actually
matters for this project. Run the same failure-mode analysis from the
contamination section on all five fine-tuned models, at each one's best-F1
confidence:

| system | header/footer detected | …folded into text-area | footnote detected | …folded into text-area |
|---|---|---|---|---|
| **RT-DETR-l fine-tuned (Ours)** | 96% | 0.6% | 93% | 2% |
| RF-DETR-L fine-tuned | 97% | 0.1% | 93% | 7% |
| Docling heron fine-tuned | 95% | 0.1% | 93% | 2% |
| PP-DocLayout-L fine-tuned | 93% | 0.2% | 89% | 7% |
| DocLayout-YOLO fine-tuned | 96% | 0.1% | 87% | 0% |

Where the off-the-shelf systems spanned a 40-point range on header/footer
contamination alone (1.2% to 56%), every fine-tuned model lands under 1%. All
five still leave some footnote contamination on the table (0–7%), which
tracks with footnote being the class with the least training data — but
that's an order of magnitude better than any off-the-shelf system managed.
Notably, our RT-DETR-l fine-tuned model's footnote F1 is the best of the five
in the architecture table above, yet its footnote contamination (2%) is merely
mid-pack here —
another small reminder that F1 and contamination are related but not
interchangeable measurements.

So architecture, in the end, mattered less than we expected: several
architectures, fine-tuned on the same audited Tibetan dataset, converge to
roughly the same ceiling, on the failure mode we actually care about as well
as on F1. What mattered was having that dataset, the canonical evaluator, and
the labelling lessons from the previous section — the model family was almost
a free choice once those were right.

## Our models

Our production model is **RT-DETR-l fine-tuned** on the `tam2col`-labelled data:
a 4-class RT-DETR-l that keeps header and footer as separate training classes
(combined losslessly afterward), merges text-area into a single envelope
*except* on genuine two-column pages, and is served with per-class thresholds
(header/footer ≈ 0.60, text-area ≈ 0.55, footnote ≈ 0.25). It has the best
canonical AP50 (0.981), the best text-area and footnote localization, and is
the only variant that handles two columns correctly.

It also closes the loop on the question that started this post. On the same
contamination test that showed Azure folding 12% of headers/footers and 22% of
footnotes into the body text, it folds in **0.6%** and **2%** — an order of
magnitude cleaner, and the number that actually matters for a downstream OCR
pipeline that will run across millions of pages.

The bigger takeaways, though, are the ones we didn't expect going in:

1. **"Solved problem" is a trap.** For a low-resource script, the mature tooling —
   open-source *and* commercial — mostly isn't built for you, and a good aggregate
   score can hide a much worse failure mode. Fine-tuning a strong off-the-shelf
   detector (RF-DETR) can match a from-scratch RT-DETR-l training run, but nothing
   off-the-shelf, fine-tuned or not, gets there without first having the audited
   Tibetan dataset to fine-tune on.
2. **The metric will lie to you if you let it.** Half of our "wins" were
   measurement artifacts until we forced every model into a common evaluation
   space, and Azure's reassuring 0.989 text-area F1 hid the very failure mode we
   cared most about.
3. **Thresholds are a per-class decision**, and getting them right was worth as
   much as any architecture change.

Everything is open. The primary trained model (with usage and thresholds) is
on the Hugging Face Hub at
[BDRC/Tibetan-Modern-Book-Layout-Detection-RTDETR](https://huggingface.co/BDRC/Tibetan-Modern-Book-Layout-Detection-RTDETR)
(MIT); the two runner-up architectures from the bake-off above are published
alongside it, at
[BDRC/Tibetan-Modern-Book-Layout-Detection-RFDETR](https://huggingface.co/BDRC/Tibetan-Modern-Book-Layout-Detection-RFDETR)
(Apache-2.0) and
[BDRC/Tibetan-Modern-Book-Layout-Detection-DocLayout-YOLO](https://huggingface.co/BDRC/Tibetan-Modern-Book-Layout-Detection-DocLayout-YOLO)
(AGPL-3.0, inherited from DocLayout-YOLO's own codebase — check that license
fits your use case before adopting it over the other two). The cleaned,
audited dataset is at
[BDRC/TDLA-Training-Dataset-v2](https://huggingface.co/datasets/BDRC/TDLA-Training-Dataset-v2);
and all training, evaluation, and threshold-sweep code lives in the
[tibetan-book-layout-analysis](https://github.com/buda-base/tibetan-book-layout-analysis)
repository. Every number in this post is reproducible from the scripts there.

## Limitations

A few things we didn't test, in the interest of not overclaiming:

- **We ran Azure, Surya, and AWS Textract end-to-end; Google Document AI and
  ABBYY FineReader we didn't.** For Google Document AI we got far enough to
  confirm its Layout Parser correctly tags Tibetan text as header/paragraph/footer
  — but its bounding-box output is currently broken (an open bug on Google's
  side, unrelated to Tibetan specifically), so we couldn't score it against our
  IoU-based evaluator. ABBYY FineReader was assessed from its documented schema
  and our prior about its training data only. It's possible one of them would
  do better than we expect — we just haven't measured it.
- **We only tested vision-based, pre-OCR layout detection.** A different
  approach entirely is to run OCR first and strip headers, footers, and
  footnotes afterward using textual signals — a line that repeats
  near-identically every few pages is probably a running header, a lone
  short number in a consistent position is probably a folio number. We
  didn't build or benchmark that pipeline. It could be a reasonable
  complement to layout detection (or catch contamination that geometric
  detection misses), but it has its own weaknesses — it needs the OCR to
  already be roughly right, and a header that changes wording, like a
  chapter title, is harder to catch by repetition alone.
- **The close scores across fine-tuned architectures (0.93–0.96 mean F1) are
  suggestive, not conclusive.** The reading we find most likely is the
  mundane one: modern detector architectures tend to converge once you fine-tune
  them on enough clean, well-annotated data, and that's consistent with the
  rest of this post — the dataset and the labelling scheme did more work
  than the architecture. But we can't fully rule out a less comfortable
  reading: that part of our ~0.95 ceiling is an artifact of the benchmark
  itself — ambiguous cases, small inconsistencies in where a box edge should
  fall — rather than genuine headroom being exhausted. We haven't measured
  inter-annotator agreement on our own boxes, so we'd treat a future model
  that claims a small improvement over our fine-tuned baselines on this exact
  benchmark with some skepticism until it's been checked against that noise
  floor.

## Acknowledgements

This work is part of the BDRC Etext Corpus project, funded by the Khyentse
Foundation. Thanks to the OpenPecha/Dharmaduta team, whose earlier benchmark
on this exact problem is what got this investigation started, and to
everyone on the BDRC team who annotated, audited, and corrected boxes across
the dataset's several rounds.
