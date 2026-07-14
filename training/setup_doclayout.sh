#!/usr/bin/env bash
# Create a dedicated venv for DocLayout-YOLO.
#
# DocLayout-YOLO ships as a renamed fork of ultralytics ("doclayout_yolo") and
# pins its own dependency set, so it is kept out of the mainline /opt/pytorch
# environment to avoid clobbering the YOLO26 install.
#
# Usage:  bash setup_doclayout.sh
# Then:   ~/hff_training/.venv_doclayout/bin/python train.py \
#             --framework doclayout --model <doclayout_base.pt> ...
set -euo pipefail

VENV="$HOME/bec-orchestration/hff_training/.venv_doclayout"
WEIGHTS_DIR="$HOME/bec-orchestration/hff_training/weights"

if ! python3 -m venv --help >/dev/null 2>&1; then
    sudo apt-get update -qq && sudo apt-get install -y -qq python3.12-venv
fi

python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -U pip wheel
# torch (CUDA) + the DocLayout-YOLO fork + helpers.
"$VENV/bin/pip" install -q torch torchvision
"$VENV/bin/pip" install -q doclayout-yolo huggingface-hub opencv-python-headless pyyaml pandas

echo "DocLayout-YOLO venv ready: $VENV"
"$VENV/bin/python" -c "from doclayout_yolo import YOLOv10; import torch; print('doclayout OK, cuda', torch.cuda.is_available())"

# Base checkpoint (DocStructBench) to fine-tune from.
mkdir -p "$WEIGHTS_DIR"
"$VENV/bin/python" - <<PY
from huggingface_hub import hf_hub_download
p = hf_hub_download("juliozhao/DocLayout-YOLO-DocStructBench",
                    "doclayout_yolo_docstructbench_imgsz1024.pt",
                    local_dir="$WEIGHTS_DIR")
print("base weights:", p)
PY

cat <<'NOTE'

To fine-tune DocLayout-YOLO on the merged dataset, download a base checkpoint
(DocStructBench) once, then train from it:

  from huggingface_hub import hf_hub_download
  hf_hub_download("juliozhao/DocLayout-YOLO-DocStructBench",
                  "doclayout_yolo_docstructbench_imgsz1024.pt",
                  local_dir="weights")

  ~/hff_training/.venv_doclayout/bin/python train.py --framework doclayout \
      --model weights/doclayout_yolo_docstructbench_imgsz1024.pt \
      --data dataset/data.yaml --imgsz 1280 --epochs 100 --name doclayout_1280
NOTE
