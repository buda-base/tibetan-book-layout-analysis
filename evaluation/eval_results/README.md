# Dataset versions and where numbers live

Hub dataset: [BDRC/TiBLAD](https://huggingface.co/datasets/BDRC/TiBLAD)
(paired with the models `BDRC/TiBLA-RTDETR`, `BDRC/TiBLA-PP-DocLayout-L`,
`BDRC/TiBLA-RFDETR`). **Always name the git tag**, never "the TDLA test set".

| Tag | Images (train / val / test) | Split unit | What it is |
| --- | --- | --- | --- |
| `v1` | 2,794 | volume | Original release |
| `v2` | 8,325 (6,751 / 714 / **860**) | volume (`i_id`) | Expanded corpus; **series (`w_id`) leakage** across splits |
| `v3` | 8,093 (6,751 / 615 / **727**) | volume, then drop leaking val/test pages | v2 with series leakage removed; train unchanged |
| `v4` | 8,325 (6,743 / 749 / **833**) | series (`w_id`), stratified | Full v2 corpus, re-split; classes + side vs top/bottom headers/footers balanced |

## Scores and prediction dumps

Put every metric JSON, sweep CSV, and YOLO dump under a folder whose name **is**
the dataset tag:

```
evaluation/eval_results/tdla-v2/   # alias: literature/ is the v2 paper comparison (860-page test)
evaluation/eval_results/tdla-v3/   # tag v3, 727-page test
evaluation/eval_results/tdla-v4/   # tag v4, 833-page test
```

Each of those folders has a `DATASET` file (tag, test size, Hub revision).
Do not write new scores into `literature/` — that directory is frozen as **v2**.

When logging metrics, set `"dataset_tag": "v3"` (or `v4`) in the JSON.

## Split lists (this repo)

Exact image paths for each release:

```
data/splits/v3/{train,val,test}.txt
data/splits/v4/{train,val,test}.txt
```

These match `train.txt` / `val.txt` / `test.txt` on the Hub at the same tag.
