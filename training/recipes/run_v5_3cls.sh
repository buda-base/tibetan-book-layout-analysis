#!/usr/bin/env bash
# Round-2 curriculum: RT-DETR-l, 3-class (header+footer merged -> "header-footer").
# Same test split as baseline. save-period keeps periodic checkpoints.
set -uo pipefail
cd ~/bec-orchestration/hff_training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[v5] START rtdetr_v5_3cls @ $(date -u)"
/opt/pytorch/bin/python train.py --data dataset_v5_3cls/data.yaml --framework rtdetr \
    --model rtdetr-l.pt --imgsz 1024 --epochs 100 \
    --device 0 --patience 20 --name rtdetr_v5_3cls --batch 8 --save-period 10 \
    && echo "[v5] DONE rtdetr_v5_3cls" || echo "[v5] FAIL rtdetr_v5_3cls"
echo "V5_3CLS_COMPLETE @ $(date -u)"
