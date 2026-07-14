#!/usr/bin/env bash
# Final baseline retrain: RT-DETR-l, 4-class (unmerged), on the cleaner
# dataset_v5 (latest label fixes). This is the chosen production curriculum.
set -uo pipefail
cd ~/bec-orchestration/hff_training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[v5] START rtdetr_v5_baseline @ $(date -u)"
/opt/pytorch/bin/python train.py --data dataset_v5/data.yaml --framework rtdetr \
    --model rtdetr-l.pt --imgsz 1024 --epochs 100 \
    --device 0 --patience 20 --name rtdetr_v5_baseline --batch 8 --save-period 10 \
    && echo "[v5] DONE rtdetr_v5_baseline" || echo "[v5] FAIL rtdetr_v5_baseline"
echo "V5_BASELINE_COMPLETE @ $(date -u)"
