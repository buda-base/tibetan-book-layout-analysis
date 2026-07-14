#!/usr/bin/env bash
# tam variant: RT-DETR-l, 4-class, text-area merged EXCEPT two-column pages keep
# two text-area boxes (one per column). On dataset_v5_tam2col.
set -uo pipefail
cd ~/bec-orchestration/hff_training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[v5] START rtdetr_v5_tam2col @ $(date -u)"
/opt/pytorch/bin/python train.py --data dataset_v5_tam2col/data.yaml --framework rtdetr \
    --model rtdetr-l.pt --imgsz 1024 --epochs 100 \
    --device 0 --patience 20 --name rtdetr_v5_tam2col --batch 8 --save-period 10 \
    && echo "[v5] DONE rtdetr_v5_tam2col" || echo "[v5] FAIL rtdetr_v5_tam2col"
echo "V5_TAM2COL_COMPLETE @ $(date -u)"
