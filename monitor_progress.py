#!/usr/bin/env python3
"""
monitor_progress.py — Maintain a clean, human-readable progress log while
chemprop trains in the background.

It polls two sources, in order of preference:
  1. Lightning's CSVLogger output written by chemprop:
         <checkpoint-dir>/**/trainer_logs/version_*/metrics.csv
     Columns typically include: epoch, step, train_loss, val_loss, val/mae, val/mse
     (all metric values are on the ORIGINAL / physical scale, because chemprop's
      regression predictor un-normalizes predictions before computing metrics).
  2. Fallback: regex-scanning the raw training log produced by the tqdm
     progress bar (used only if tensorboard is installed so no metrics.csv exists).

For every completed validation epoch it appends ONE line like:

[2026-09-16 14:22:31] Epoch  34/500 | val_loss=0.0456 | val_MAE=0.0342 | val_MSE=0.0021 | train_loss=0.0123 | best_val_MAE=0.0330 @ epoch 31

When the trainer process (PID) exits, it writes a final summary block,
including the test-set metrics chemprop prints at the very end of training.

Usage (normally invoked by run_chemprop_bg.sh, not by hand):
    python3 monitor_progress.py \
        --property Polarizability \
        --epochs 500 \
        --pid 12345 \
        --run-log logs/Polarizability_20260916_140500.log \
        --out logs/Polarizability_progress.log \
        --checkpoint-dir checkpoints/Polarizability \
        --poll 20
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import os
import re
import signal
import sys
import time
from pathlib import Path


def now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def pid_alive(pid: int) -> bool:
    """True if the trainer process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except TypeError:
        return False


def latest_metrics_csv(checkpoint_dir: str) -> str | None:
    """Return the newest trainer_logs metrics.csv under the checkpoint dir."""
    pattern = os.path.join(checkpoint_dir, "**", "trainer_logs", "version_*", "metrics.csv")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def read_metrics_csv(path: str) -> dict[int, dict[str, float]]:
    """
    Parse a Lightning CSVLogger metrics.csv into {epoch: {metric: value}}.

    chemprop logs train_loss every few steps and val metrics once per epoch,
    so each epoch may span several rows; we keep the LAST non-null value of
    each metric per epoch. Robust to partially-written trailing lines.
    """
    per_epoch: dict[int, dict[str, float]] = {}
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_epoch = row.get("epoch")
                if raw_epoch is None or raw_epoch == "":
                    continue
                try:
                    epoch = int(float(raw_epoch))
                except ValueError:
                    continue
                slot = per_epoch.setdefault(epoch, {})
                for key, val in row.items():
                    if key in (None, "epoch", "step"):
                        continue
                    if val is None or val == "":
                        continue
                    try:
                        slot[key] = float(val)
                    except ValueError:
                        continue
    except (OSError, csv.Error):
        # File may be mid-write; keep whatever we parsed so far.
        pass
    return per_epoch


TQDM_EPOCH_RE = re.compile(r"Epoch (\d+):")
TQDM_METRIC_RE = re.compile(r"(val/mae|val/mse|val_loss|train_loss(?:_step)?)=([-+0-9.eE]+)")


def read_metrics_from_raw_log(path: str, max_bytes: int = 4_000_000) -> dict[int, dict[str, float]]:
    """
    Fallback parser: extract 'Epoch N:' + 'val/mae=...' from the tqdm output
    in the raw chemprop log. Only the tail of the file is scanned to keep
    this cheap. Returns {epoch: {metric: value}} (same shape as CSV parser).
    """
    per_epoch: dict[int, dict[str, float]] = {}
    try:
        size = os.path.getsize(path)
        with open(path, "r", errors="replace") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()  # skip the partial line we landed in
            for line in f:
                m = TQDM_EPOCH_RE.search(line)
                if not m:
                    continue
                epoch = int(m.group(1))
                slot = per_epoch.setdefault(epoch, {})
                for km, kv in TQDM_METRIC_RE.findall(line):
                    try:
                        slot[km] = float(kv)
                    except ValueError:
                        continue
    except OSError:
        pass
    return per_epoch


TEST_METRIC_RE = re.compile(r"test/(mae|rmse|mse):\s*([-+0-9.eE]+)")


def find_test_metrics(raw_log: str) -> dict[str, float]:
    """Scrape 'test/mae: 0.1234' style lines chemprop prints at the end."""
    results: dict[str, float] = {}
    try:
        with open(raw_log, "r", errors="replace") as f:
            for line in f:
                m = TEST_METRIC_RE.search(line)
                if m:
                    try:
                        results[f"test/{m.group(1)}"] = float(m.group(2))
                    except ValueError:
                        continue
    except OSError:
        pass
    return results


def fmt(v: float | None, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Maintain a clean chemprop progress log.")
    p.add_argument("--property", dest="prop", required=True)
    p.add_argument("--epochs", type=int, required=True, help="Total planned epochs")
    p.add_argument("--pid", type=int, required=True, help="PID of the chemprop trainer process")
    p.add_argument("--run-log", required=True, help="Raw chemprop stdout/stderr log")
    p.add_argument("--out", required=True, help="Clean progress log to append to")
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--poll", type=float, default=20.0, help="Seconds between polls")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Header (log is truncated by the launcher before the monitor starts)
    with open(out_path, "a") as f:
        f.write("=" * 100 + "\n")
        f.write(f" Chemprop training progress — {args.prop}\n")
        f.write(f" Started : {now_str()}\n")
        f.write(f" Target  : {args.epochs} epochs   |   Trainer PID: {args.pid}\n")
        f.write(f" Raw log : {args.run_log}\n")
        f.write(
            " Note    : val_MAE / val_MSE are on the original (physical) scale\n"
            "           of the property, computed by chemprop on the validation split.\n"
        )
        f.write("=" * 100 + "\n")
        f.write(f"[{now_str()}] Waiting for the first epoch to finish "
                f"(data loading + featurization can take a few minutes)...\n")
        f.flush()

    last_written = -1
    best_mae: float | None = None
    best_epoch: int | None = None
    start = time.time()
    heartbeat_written = False

    def poll_sources() -> dict[int, dict[str, float]]:
        csv_path = latest_metrics_csv(args.checkpoint_dir)
        if csv_path:
            data = read_metrics_csv(csv_path)
            if data:
                return data
        if os.path.exists(args.run_log):
            return read_metrics_from_raw_log(args.run_log)
        return {}

    def write_epoch_line(epoch: int, m: dict[str, float], final: bool = False) -> None:
        nonlocal best_mae, best_epoch
        val_mae = m.get("val/mae")
        if val_mae is not None and (best_mae is None or val_mae < best_mae):
            best_mae, best_epoch = val_mae, epoch
        best_txt = ""
        if best_mae is not None and best_epoch is not None:
            best_txt = f" | best_val_MAE={fmt(best_mae)} @ epoch {best_epoch + 1}"
        with open(out_path, "a") as f:
            f.write(
                f"[{now_str()}] Epoch {epoch + 1:>4}/{args.epochs}"
                f" | val_loss={fmt(m.get('val_loss'))}"
                f" | val_MAE={fmt(m.get('val/mae'))}"
                f" | val_MSE={fmt(m.get('val/mse'))}"
                f" | train_loss={fmt(m.get('train_loss'))}"
                f"{best_txt}\n"
            )
            f.flush()

    def graceful_exit(signum, _frame):
        with open(out_path, "a") as f:
            f.write(f"[{now_str()}] Monitor received signal {signum}; exiting "
                    f"(training process is NOT affected).\n")
        sys.exit(0)

    signal.signal(signal.SIGTERM, graceful_exit)
    signal.signal(signal.SIGINT, graceful_exit)

    while True:
        data = poll_sources()

        # Write any newly-completed epochs that carry validation results
        for epoch in sorted(data.keys()):
            if epoch <= last_written:
                continue
            m = data[epoch]
            has_val = any(k in m for k in ("val/mae", "val_loss", "val/mse"))
            if has_val:
                write_epoch_line(epoch, m)
                last_written = max(last_written, epoch)

        # Reassuring heartbeat while the very first epoch is still running
        if (last_written == -1 and not heartbeat_written
                and time.time() - start > 600):
            with open(out_path, "a") as f:
                f.write(f"[{now_str()}] Still no completed epoch after "
                        f"{int(time.time() - start)}s — trainer is alive; large "
                        f"datasets take a while before the first validation. "
                        f"See raw log for details.\n")
            heartbeat_written = True

        if not pid_alive(args.pid):
            time.sleep(3)  # let the trainer flush its last writes
            data = poll_sources()
            for epoch in sorted(data.keys()):
                if epoch > last_written:
                    m = data[epoch]
                    if any(k in m for k in ("val/mae", "val_loss", "val/mse")):
                        write_epoch_line(epoch, m)
                        last_written = max(last_written, epoch)
            break

        time.sleep(args.poll)

    # ---------------- Final summary ----------------
    elapsed = int(time.time() - start)
    hrs, rem = divmod(elapsed, 3600)
    mins, _ = divmod(rem, 60)

    test_metrics = find_test_metrics(args.run_log) if os.path.exists(args.run_log) else {}

    with open(out_path, "a") as f:
        f.write("-" * 100 + "\n")
        f.write(f"[{now_str()}] TRAINING RUN ENDED (monitor watched PID {args.pid}, "
                f"elapsed {hrs}h {mins:02d}m)\n")
        f.write(f"  Epochs with validation metrics recorded : {last_written + 1} "
                f"of {args.epochs} planned\n")
        if best_mae is not None and best_epoch is not None:
            f.write(f"  Best validation MAE (physical units)     : {best_mae:.6f} "
                    f"@ epoch {best_epoch + 1}\n")
        else:
            f.write("  No validation metrics were recorded — check the raw log / .err file.\n")
        if test_metrics:
            f.write("  Final TEST-set metrics (from chemprop, physical units):\n")
            for k in sorted(test_metrics):
                f.write(f"    {k}: {test_metrics[k]:.6f}\n")
            f.write("  -> Per-molecule test predictions: "
                    f"{args.checkpoint_dir}/model_0/test_predictions.csv\n")
        f.write("=" * 100 + "\n")


if __name__ == "__main__":
    main()
