#!/usr/bin/env python3
"""Fine-tune PP-DocLayout-L via PaddleX on our YOLO layout dataset.

Converts dataset_v5_tam2col to PaddleX COCO layout, then runs PaddleX train with
the PP-DocLayout-L config. Requires a PaddleX checkout (cloned on first run).

Usage:
  python train_pp_doclayout.py --yolo-dataset <yolo_dir> --out <run_dir>
                                 [--epochs 100] [--batch 4] [--lr 1e-4]
                                 [--paddlex-dir ~/PaddleX]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
_AP_RE = re.compile(r"Best test bbox ap is ([\d]+\.[\d]+)")
_CKPT_RE = re.compile(r"Save checkpoint: .*/(\d+)\s*$")


def yolo_to_paddlex_coco(yolo_dir: Path, out: Path) -> None:
    names = {}
    in_names = False
    for ln in (yolo_dir / "data.yaml").read_text().splitlines():
        if ln.strip().startswith("names:"):
            in_names = True
            continue
        if in_names:
            s = ln.strip()
            if not s or not s[0].isdigit():
                break
            k, v = s.split(":", 1)
            names[int(k)] = v.strip()

    for split, px_split in [("train", "train"), ("val", "val")]:
        img_src = yolo_dir / "images" / split
        lbl_src = yolo_dir / "labels" / split
        if not img_src.is_dir():
            continue
        # PaddleX expects a flat images/ dir shared by train+val COCO jsons
        img_dst = out / "images"
        img_dst.mkdir(parents=True, exist_ok=True)
        images, annotations, ann_id = [], [], 1
        categories = [{"id": i + 1, "name": names[i], "supercategory": "none"}
                      for i in sorted(names)]
        imgs = sorted(p for p in img_src.iterdir() if p.suffix.lower() in IMG_EXT)
        for img_id, ip in enumerate(imgs, 1):
            dst = img_dst / ip.name
            if not dst.exists():
                dst.symlink_to(ip.resolve())
            from PIL import Image
            with Image.open(ip) as im:
                W, H = im.size
            images.append({"id": img_id, "file_name": ip.name, "width": W, "height": H})
            lp = lbl_src / f"{ip.stem}.txt"
            if lp.exists():
                for ln in lp.read_text().splitlines():
                    p = ln.split()
                    if len(p) < 5:
                        continue
                    c = int(p[0])
                    cx, cy, w, h = (float(x) for x in p[1:5])
                    bw, bh = w * W, h * H
                    bx, by = (cx * W) - bw / 2, (cy * H) - bh / 2
                    annotations.append({
                        "id": ann_id, "image_id": img_id, "category_id": c + 1,
                        "bbox": [bx, by, bw, bh], "area": bw * bh, "iscrowd": 0,
                        "segmentation": [],
                    })
                    ann_id += 1
        ann_name = "instance_train.json" if px_split == "train" else "instance_val.json"
        (out / "annotations").mkdir(parents=True, exist_ok=True)
        (out / "annotations" / ann_name).write_text(json.dumps({
            "images": images, "annotations": annotations, "categories": categories,
        }))


def ensure_paddledetection(paddlex_dir: Path) -> Path:
    """Register PaddleDetection models with PaddleX (plugin install may stop at ext_op)."""
    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "setuptools<81"],
            check=True,
        )
    pd_repo = paddlex_dir / "paddlex/repo_manager/repos/PaddleDetection"
    if not pd_repo.is_dir():
        raise SystemExit(
            f"PaddleDetection repo missing at {pd_repo}. "
            "Run: paddlex --install PaddleDetection -y"
        )
    (pd_repo / ".installed").touch()
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["PADDLE_PDX_PADDLEDETECTION_PATH"] = str(pd_repo.resolve())
    # Import registers PP-DocLayout-L with PaddleX
    import paddlex.repo_apis.PaddleDetection_api.object_det.register  # noqa: F401
    from paddlex.repo_apis.base.register import get_registered_model_info
    get_registered_model_info("PP-DocLayout-L")
    patch_best_model_strict_improvement(pd_repo)
    return pd_repo


def patch_best_model_strict_improvement(pd_repo: Path) -> None:
    """Keep earliest best_model on val ties (PaddleDetection defaults to >=)."""
    callbacks = pd_repo / "ppdet/engine/callbacks.py"
    if not callbacks.is_file():
        return
    text = callbacks.read_text()
    replacements = [
        ("if epoch_ap >= self.best_ap:", "if epoch_ap > self.best_ap:"),
        ("if map_res[key][0] >= self.best_ap:", "if map_res[key][0] > self.best_ap:"),
    ]
    changed = False
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            changed = True
    if changed:
        callbacks.write_text(text)
        print("Patched PaddleDetection callbacks: best_model uses strict >", flush=True)


def prune_epoch_checkpoints(out_train: Path, keep: int = 2) -> None:
    """Drop old per-epoch Paddle saves; always keep best_model."""
    if not out_train.is_dir():
        return
    epoch_dirs = sorted(
        (p for p in out_train.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    for p in epoch_dirs[:-keep]:
        shutil.rmtree(p, ignore_errors=True)
    # Drop stray partial saves (e.g. failed epoch-end write)
    for p in out_train.iterdir():
        if p.is_file() and p.suffix in {".pdparams", ".pdopt", ".pdema"}:
            p.unlink(missing_ok=True)


def _parse_early_stop_state(log_path: Path) -> tuple[float, int, int]:
    """Return (best_ap, best_epoch, last_val_epoch) from a PaddleDetection train log."""
    best_ap, best_epoch, last_val_epoch = 0.0, -1, -1
    if not log_path.is_file():
        return best_ap, best_epoch, last_val_epoch
    pending_epoch = -1
    for line in log_path.read_text(errors="replace").splitlines():
        m = _CKPT_RE.search(line)
        if m:
            pending_epoch = int(m.group(1))
        m = _AP_RE.search(line)
        if m and pending_epoch >= 0:
            ap = float(m.group(1))
            last_val_epoch = pending_epoch
            if ap > best_ap:
                best_ap, best_epoch = ap, pending_epoch
            pending_epoch = -1
    return best_ap, best_epoch, last_val_epoch


def should_early_stop(
    best_ap: float,
    best_epoch: int,
    last_val_epoch: int,
    *,
    patience: int,
    target_ap: float,
    max_epoch_no_target: int,
) -> tuple[bool, str]:
    crossed = best_ap >= target_ap
    if not crossed and last_val_epoch >= max_epoch_no_target:
        return True, (
            f"best bbox ap {best_ap:.3f} never reached {target_ap:.3f}; "
            f"stopping after epoch {last_val_epoch}"
        )
    if crossed and best_epoch >= 0 and (last_val_epoch - best_epoch) >= patience:
        return True, (
            f"no val improvement for {patience} epochs "
            f"(best {best_ap:.3f} @ epoch {best_epoch})"
        )
    return False, ""


def start_early_stop_monitor(
    log_path: Path,
    proc: subprocess.Popen,
    *,
    patience: int = 20,
    target_ap: float = 0.810,
    max_epoch_no_target: int = 25,
    poll_interval: int = 60,
):
    def _loop():
        pos = 0
        best_ap, best_epoch, last_val_epoch = _parse_early_stop_state(log_path)
        pending_epoch = -1
        while proc.poll() is None:
            if log_path.is_file():
                text = log_path.read_text(errors="replace")
                new = text[pos:]
                pos = len(text)
                for line in new.splitlines():
                    m = _CKPT_RE.search(line)
                    if m:
                        pending_epoch = int(m.group(1))
                    m = _AP_RE.search(line)
                    if m and pending_epoch >= 0:
                        ap = float(m.group(1))
                        last_val_epoch = pending_epoch
                        if ap > best_ap:
                            best_ap, best_epoch = ap, pending_epoch
                        pending_epoch = -1
                        stop, reason = should_early_stop(
                            best_ap, best_epoch, last_val_epoch,
                            patience=patience,
                            target_ap=target_ap,
                            max_epoch_no_target=max_epoch_no_target,
                        )
                        if stop:
                            print(f"EARLY_STOP: {reason}", flush=True)
                            proc.send_signal(signal.SIGTERM)
                            return
            time.sleep(poll_interval)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


def start_checkpoint_pruner(out_train: Path, keep: int = 2, interval: int = 300):
    def _loop():
        while True:
            prune_epoch_checkpoints(out_train, keep=keep)
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


def run_paddlex(
    main_py: Path,
    paddlex_dir: Path,
    cfg: Path,
    opts: list[str],
    *,
    interruptible: bool = False,
) -> subprocess.Popen | None:
    env = os.environ.copy()
    subprocess.run(
        [sys.executable, "-c",
         "import paddlex.repo_apis.PaddleDetection_api.object_det.register"],
        check=True, cwd=str(paddlex_dir), env=env,
    )
    cmd = [sys.executable, str(main_py), "-c", str(cfg), *opts]
    print(" ".join(cmd), flush=True)
    if interruptible:
        return subprocess.Popen(cmd, cwd=str(paddlex_dir), env=env)
    subprocess.run(cmd, check=True, cwd=str(paddlex_dir), env=env)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-classes", type=int, default=4)
    ap.add_argument("--device", default="gpu:0")
    ap.add_argument("--paddlex-dir", default="~/PaddleX")
    ap.add_argument("--keep-checkpoints", type=int, default=2,
                    help="Max per-epoch checkpoint dirs to retain (plus best_model)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from train_output/best_model if present")
    ap.add_argument("--patience", type=int, default=20,
                    help="Stop after this many val epochs without improvement "
                         "(once target AP is reached)")
    ap.add_argument("--target-ap", type=float, default=0.810,
                    help="Val bbox AP threshold; below this, cap at --max-epoch-no-target")
    ap.add_argument("--max-epoch-no-target", type=int, default=25,
                    help="Last epoch (0-based) to run if --target-ap is never reached")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    ds = out / "paddlex_coco"
    if not (ds / "annotations" / "instance_train.json").exists():
        yolo_to_paddlex_coco(Path(args.yolo_dataset), ds)

    paddlex_dir = Path(args.paddlex_dir).expanduser()
    main_py = paddlex_dir / "main.py"
    if not main_py.exists():
        raise SystemExit(f"PaddleX not found at {paddlex_dir}")
    ensure_paddledetection(paddlex_dir)
    cfg = paddlex_dir / "paddlex/configs/modules/layout_detection/PP-DocLayout-L.yaml"
    out_train = out / "train_output"
    prune_epoch_checkpoints(out_train, keep=args.keep_checkpoints)
    start_checkpoint_pruner(out_train, keep=args.keep_checkpoints)

    common = [
        "-o", f"Global.dataset_dir={ds.resolve()}",
        "-o", f"Global.output={out_train.resolve()}",
        "-o", f"Global.device={args.device}",
    ]
    if args.resume and (out_train / "best_model.pdparams").exists():
        common.extend(["-o", f"Global.pretrained_model={out_train / 'best_model'}"])

    print("=== check_dataset ===", flush=True)
    run_paddlex(main_py, paddlex_dir, cfg, ["-o", "Global.mode=check_dataset", *common])

    print("=== train ===", flush=True)
    train_log = out_train / "train.log"
    proc = run_paddlex(main_py, paddlex_dir, cfg, [
        "-o", "Global.mode=train",
        *common,
        "-o", f"Train.epochs_iters={args.epochs}",
        "-o", f"Train.batch_size={args.batch}",
        "-o", f"Train.learning_rate={args.lr}",
        "-o", f"Train.num_classes={args.num_classes}",
    ], interruptible=True)
    assert proc is not None
    start_early_stop_monitor(
        train_log, proc,
        patience=args.patience,
        target_ap=args.target_ap,
        max_epoch_no_target=args.max_epoch_no_target,
    )
    rc = proc.wait()
    best_ap, best_epoch, last_val_epoch = _parse_early_stop_state(train_log)
    stop, reason = should_early_stop(
        best_ap, best_epoch, last_val_epoch,
        patience=args.patience,
        target_ap=args.target_ap,
        max_epoch_no_target=args.max_epoch_no_target,
    )
    if stop:
        print(f"PP_DOCLAYOUT_TRAIN_EARLY_STOP: {reason}", flush=True)
        return 0
    if rc != 0:
        raise SystemExit(rc)
    print("PP_DOCLAYOUT_TRAIN_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
