#!/usr/bin/env bash
# GPU-side recipe for Task 2, ours -> DocLayNet v1.2 test (shared classes).
# Intended to run on a g5/g6 with /opt/pytorch or a venv that has ultralytics
# + datasets + pycocotools + cotescore. Streaming export does not keep PDFs.
#
# Example (on the GPU box, after scp of this repo + tibetan_book_layout.pt):
#   bash evaluation/run_doclaynet_transfer.sh \
#       --weights /home/ubuntu/tibetan_book_layout.pt \
#       --out /home/ubuntu/doclaynet_transfer
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVAL="$ROOT/evaluation"
PY="${PYTHON:-python3}"
WEIGHTS=""
OUT="/tmp/doclaynet_transfer"
DEVICE="${DEVICE:-0}"
IMGSZ="${IMGSZ:-1024}"
BATCH="${BATCH:-4}"
MAX_PAGES="${MAX_PAGES:-0}"
SKIP_IMAGES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --weights) WEIGHTS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --max-pages) MAX_PAGES="$2"; shift 2 ;;
    --skip-images) SKIP_IMAGES=1; shift ;;
    --python) PY="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
[[ -n "$WEIGHTS" ]] || { echo "--weights is required" >&2; exit 2; }

mkdir -p "$OUT"
EXPORT="$OUT/doclaynet_test"
PRED="$OUT/ours_pred"

echo "[1/3] export DocLayNet v1.2 test -> $EXPORT"
EXPORT_FLAGS=(--out "$EXPORT")
if [[ "$SKIP_IMAGES" -eq 1 ]]; then
  echo "ERROR: inference needs images; not skipping" >&2
  exit 2
fi
if [[ "$MAX_PAGES" -gt 0 ]]; then
  EXPORT_FLAGS+=(--max-pages "$MAX_PAGES")
fi
"$PY" "$EVAL/export_doclaynet_test.py" "${EXPORT_FLAGS[@]}"

echo "[2/3] RT-DETR-l tam2col inference -> $PRED"
"$PY" "$EVAL/rtdetr_predict.py" \
    --weights "$WEIGHTS" \
    --source "$EXPORT/images" \
    --out "$PRED" \
    --conf 0.05 --imgsz "$IMGSZ" --device "$DEVICE" --batch "$BATCH"

echo "[3/3] score shared classes"
"$PY" "$EVAL/score_ours_on_doclaynet.py" \
    --pred-dir "$PRED/labels" \
    --gt-dir "$EXPORT/labels" \
    --index "$EXPORT/index.jsonl" \
    --out "$OUT/metrics_ours_on_doclaynet.json" \
    --conf 0.50

echo "done: $OUT/metrics_ours_on_doclaynet.json"
