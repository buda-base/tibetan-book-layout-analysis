#!/usr/bin/env python3
"""Fine-tune Docling layout-heron (RT-DETRv2) on our COCO-format layout dataset.

Mirrors the tam2col recipe where the HF Trainer API allows: 100 epochs, batch 8,
lr 1e-4, early stopping patience 20. Starts from docling-project/docling-layout-heron
and retrains the detection head for our 4 classes.

Expects COCO layout from yolo2coco.py:
  <dataset>/{train,valid,test}/_annotations.coco.json + images alongside.

Usage:
  python train_docling_heron.py --dataset <coco_dir> --out <run_dir>
                                [--epochs 100] [--batch 8] [--lr 1e-4]
                                [--patience 20]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    RTDetrImageProcessor,
    RTDetrV2ForObjectDetection,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)


class CocoLayoutDataset(Dataset):
    def __init__(self, split_dir: Path, processor: RTDetrImageProcessor):
        ann_path = split_dir / "_annotations.coco.json"
        coco = json.loads(ann_path.read_text())
        self.processor = processor
        self.images = {im["id"]: im for im in coco["images"]}
        self.categories = {c["id"]: c["name"] for c in coco["categories"] if c["id"] > 0}
        # RF-DETR convention: YOLO class c -> category id c+1
        self.cat_ids = sorted(self.categories)
        by_img: dict[int, list] = {i: [] for i in self.images}
        for ann in coco["annotations"]:
            by_img[ann["image_id"]].append(ann)
        self.items = []
        for img_id, im in sorted(self.images.items()):
            anns = by_img.get(img_id, [])
            coco_anns = []
            for a in anns:
                cid = a["category_id"]
                if cid not in self.categories:
                    continue
                x, y, w, h = a["bbox"]
                coco_anns.append({
                    "image_id": img_id,
                    "category_id": self.cat_ids.index(cid),
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })
            self.items.append((img_id, split_dir / im["file_name"], coco_anns))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_id, path, coco_anns = self.items[idx]
        image = Image.open(path).convert("RGB")
        target = {"image_id": img_id, "annotations": coco_anns}
        enc = self.processor(images=image, annotations=target, return_tensors="pt")
        return {"pixel_values": enc["pixel_values"].squeeze(0), "labels": enc["labels"][0]}


def collate(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = [b["labels"] for b in batch]
    return {"pixel_values": pixel_values, "labels": labels}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="docling-project/docling-layout-heron")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()

    ds_root = Path(args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    processor = RTDetrImageProcessor.from_pretrained(args.base)
    model = RTDetrV2ForObjectDetection.from_pretrained(
        args.base,
        num_labels=4,
        ignore_mismatched_sizes=True,
    )

    train_ds = CocoLayoutDataset(ds_root / "train", processor)
    val_ds = CocoLayoutDataset(ds_root / "valid", processor)
    id2label = {i: train_ds.categories[cid] for i, cid in enumerate(train_ds.cat_ids)}
    label2id = {v: k for k, v in id2label.items()}
    model.config.id2label = {str(k): v for k, v in id2label.items()}
    model.config.label2id = label2id

    eval_epochs = max(1, args.epochs // 20)
    targs = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()
    trainer.save_model(str(out / "best"))
    processor.save_pretrained(str(out / "best"))
    print("DOCLING_HERON_TRAIN_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
