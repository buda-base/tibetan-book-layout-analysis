# Split lists by dataset tag

These files are copies of `train.txt` / `val.txt` / `test.txt` from
[BDRC/TDLA-Training-Dataset](https://huggingface.co/datasets/BDRC/TDLA-Training-Dataset)
at the matching git tag. Paths are YOLO-style (`./images/<split>/<stem>.jpg`).

| Directory | Tag | Images | Test |
| --- | --- | ---: | ---: |
| `v3/` | `v3` | 8,093 | 727 |
| `v4/` | `v4` | 8,325 | 833 |

How to tell versions apart when scoring: see
[`evaluation/eval_results/README.md`](../../evaluation/eval_results/README.md).
