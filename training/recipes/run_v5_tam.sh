#!/usr/bin/env bash
# Round-2 curriculum: RT-DETR-l, 4-class, text-area boxes merged per page.
# Same test split as baseline. save-period keeps periodic checkpoints.
set -uo pipefail
cd ~/bec-orchestration/hff_training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[v5] START rtdetr_v5_tam @ $(date -u)"
/opt/pytorch/bin/python train.py --data dataset_v5_tam/data.yaml --framework rtdetr \
    --model rtdetr-l.pt --imgsz 1024 --epochs 100 \
    --device 0 --patience 20 --name rtdetr_v5_tam --batch 8 --save-period 10 \
    && echo "[v5] DONE rtdetr_v5_tam" || echo "[v5] FAIL rtdetr_v5_tam"
echo "V5_TAM_COMPLETE @ $(date -u)"
