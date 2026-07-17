#!/usr/bin/env python3
"""Watch a running PP-DocLayout PaddleX train log and stop when early-stop rules fire.

Policy (matches train_pp_doclayout.py defaults):
  - If val bbox AP never reaches --target-ap, stop after --max-epoch-no-target.
  - Once --target-ap is reached, stop after --patience val epochs without improvement.
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

# Reuse logic from train_pp_doclayout when available.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_pp_doclayout import (  # noqa: E402
    _AP_RE,
    _CKPT_RE,
    _parse_early_stop_state,
    should_early_stop,
)


def find_train_pids() -> list[int]:
    out = subprocess.check_output(["pgrep", "-f", "train_pp_doclayout.py"], text=True)
    pids = [int(x) for x in out.split() if x.strip()]
    if not pids:
        out = subprocess.check_output(
            ["pgrep", "-f", "detmodel_PP-DocLayout-L.yml"], text=True
        )
        pids = [int(x) for x in out.split() if x.strip()]
    return pids


def kill_train_tree():
    for pid in find_train_pids():
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def tail_new_lines(path: Path, pos: int) -> tuple[list[str], int]:
    if not path.is_file():
        return [], pos
    with path.open(errors="replace") as f:
        f.seek(pos)
        chunk = f.read()
        pos = f.tell()
    return chunk.splitlines(), pos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--target-ap", type=float, default=0.810)
    ap.add_argument("--max-epoch-no-target", type=int, default=25)
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log_path = Path(args.log).expanduser()
    best_ap, best_epoch, last_val_epoch = _parse_early_stop_state(log_path)
    pending_epoch = -1
    pos = log_path.stat().st_size if log_path.is_file() else 0

    print(
        f"watching {log_path} | best={best_ap:.3f}@{best_epoch} "
        f"last_val={last_val_epoch} | target={args.target_ap} "
        f"cap_epoch={args.max_epoch_no_target} patience={args.patience}",
        flush=True,
    )

    while True:
        lines, pos = tail_new_lines(log_path, pos)
        for line in lines:
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
                print(
                    f"val epoch {last_val_epoch}: ap={ap:.3f} "
                    f"best={best_ap:.3f}@{best_epoch}",
                    flush=True,
                )
                stop, reason = should_early_stop(
                    best_ap, best_epoch, last_val_epoch,
                    patience=args.patience,
                    target_ap=args.target_ap,
                    max_epoch_no_target=args.max_epoch_no_target,
                )
                if stop:
                    print(f"EARLY_STOP: {reason}", flush=True)
                    if args.dry_run:
                        return 0
                    kill_train_tree()
                    return 0

        if not find_train_pids():
            print("train process gone; watcher exiting", flush=True)
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
