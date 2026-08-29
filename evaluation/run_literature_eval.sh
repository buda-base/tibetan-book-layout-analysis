#!/usr/bin/env bash
# Re-score archived test-set predictions under COCO / COTe / LED.
# Does not re-run detectors. Requires the venv with pycocotools + cotescore.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVAL="$ROOT/evaluation"
PY="${PYTHON:-/home/eroux/pvenvs/1/bin/python}"
GT="${GT_DIR:-/home/eroux/azure_di_eval/testset/labels/test}"
IMG="${IMG_DIR:-/home/eroux/azure_di_eval/testset/images/test}"
OUT="${OUT_DIR:-$EVAL/eval_results/literature}"

"$PY" "$EVAL/run_literature_eval.py" \
    --gt-dir "$GT" \
    --img-dir "$IMG" \
    --out-dir "$OUT" \
    "$@"
echo "results: $OUT/RESULTS.md"
echo "json:    $OUT/metrics.json"
