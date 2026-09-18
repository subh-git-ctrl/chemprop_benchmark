# Chemprop QM40 VM Training & Execution Guide

Everything needed to set up, train, monitor, and evaluate **Chemprop** on the QM40 dataset for comparison against your QPred model — with **one command**.

---

## 1. Quick Start (One Command)

```bash
chmod +x run_chemprop_bg.sh run_property_bg.sh stop_chemprop.sh

# Train Polarizability for 100 epochs on the full QM40 split:
./run_chemprop_bg.sh Polarizability 100
```

That single command will:

1. **Build a dedicated, isolated environment** for chemprop — created at an **explicit prefix on your big data disk** (`/mnt/conda/envs/chemprop`, Python 3.11), or `.venv-chemprop/` in the repo if you don't use conda. It **never touches your `qpred` environment** (the old failure mode: chemprop needs Python ≥ 3.10 while qpred ships a different Python). Your own disk-mount script handles mounting the data disk; the launcher **verifies it is really mounted** before touching anything and aborts safely (nothing written to the small root disk) if it is not.
2. **Auto-install anything missing** in that env only: PyTorch (CUDA build on Linux via PyPI wheels) and `pip install -e ./chemprop`.
3. Verify / generate the train/val/test splits in `compare/` (seed 42, QPred-matching).
4. **Launch training in the background** (`nohup` + `disown`) so it keeps running after you close SSH.
5. **Launch a progress monitor** that writes a clean, continuously-updating log with one line per finished epoch, plus a final summary including TEST-set MAE/MSE.

Watch progress (clean, recommended):

```bash
tail -f logs/Polarizability_progress.log
```

You will see lines like:

```
[2026-09-16 14:22:31] Epoch  34/100 | val_loss=0.0456 | val_MAE=0.0342 (phys 2.5310) | val_MSE=0.0021 | train_loss=0.0123 | best_val_MAE=0.0330 (phys 2.4416) @ epoch 31
```

> **Units:** chemprop computes per-epoch validation metrics in the **normalized (z-score) space** of the training targets (that is how it selects checkpoints), so `val_MAE` / `val_MSE` / `val_loss` are normalized. The monitor scrapes the target scaler chemprop prints at startup (`Train data: mean = [...] | std = [...]`) and shows the **physical** value next to each: `phys = normalized x train-std`. The **final TEST metrics and `test_predictions.csv` ARE in physical units** (chemprop de-normalizes in eval mode), so end-of-run numbers are directly comparable with QPred's physical MAE.

Stop a training:

```bash
./stop_chemprop.sh Polarizability
```

---

## 2. Command Syntax

```bash
./run_chemprop_bg.sh <property_name> [num_epochs=100] [batch_size=64] [num_workers=2]

# equivalent alias (kept for compatibility):
./run_property_bg.sh <property_name> [num_epochs] [batch_size] [num_workers]
```

Examples:

```bash
./run_chemprop_bg.sh Polarizability 100          # default batch 64
./run_chemprop_bg.sh HOMO 500 64 4               # 500 epochs, batch 64, 4 dataloader workers
./run_chemprop_bg.sh "spatial extent" 100        # quotes needed for names with spaces
./run_chemprop_bg.sh "Internal_E(0K)" 100
```

Optional environment variables:

```bash
CHEMPROP_DATA_DISK=/mnt ./run_chemprop_bg.sh ...                    # data-disk mountpoint (default: /mnt)
CHEMPROP_ENV_DIR=/abs/path/envs/chemprop ./run_chemprop_bg.sh ...   # full env-path override (wins over the above)
CHEMPROP_ENV_NAME=chemprop ./run_chemprop_bg.sh ...                 # env name for the default location
CHEMPROP_PYTHON=/usr/bin/python3.11 ./run_chemprop_bg.sh ...        # python for venv creation
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./run_chemprop_bg.sh ...  # specific CUDA build
```

Multi-GPU selection:

```bash
CUDA_VISIBLE_DEVICES=0 ./run_chemprop_bg.sh Polarizability 100
```

> **Note on dataset size:** the script trains on the **entire split** (110,000 train / 10,000 val / 42,956 test molecules). `batch_size` is only the GPU mini-batch size, and `num_workers` only the dataloader parallelism.

---

## 3. Environment Notes (first run only)

The very first run downloads and installs PyTorch + chemprop into the dedicated env (one-time, ~10–20 min depending on bandwidth). Every later run starts in seconds — the env lives **on the big data disk** (`/mnt/conda/envs/chemprop` by default, ~60 GB free), and the pip cache is automatically moved to `/mnt/pip-cache` as well, so the small root disk stays untouched.

What the auto-bootstrap picks, in order:

| Situation | Environment used |
|---|---|
| `CHEMPROP_ENV_DIR` set | created/reused at exactly that path |
| `<data-disk>/conda/envs/chemprop` exists (default `/mnt`) | reused as-is |
| conda available | created at the **explicit prefix** `<data-disk>/conda/envs/chemprop` (Python 3.11) — conda's `envs_dirs` can never redirect it elsewhere |
| no conda | creates `.venv-chemprop` in the repo from newest system Python ≥ 3.10 |

> A conda env named `chemprop` that lives **anywhere other than the managed prefix** (e.g. `~/.conda/envs/chemprop` on the root disk) is **ignored** with a note — the launcher only uses the env at the prefix. To use that exact path anyway: `CHEMPROP_ENV_DIR=/that/path`.

> **Data-disk check:** before creating anything, the launcher verifies that `<data-disk>` is a genuinely mounted filesystem (and not just the root fs showing through an unmounted `/mnt` — the classic post-reboot trap). If the disk is not mounted it aborts with **nothing written**, telling you to run your disk-mount script first. Override the mountpoint with `CHEMPROP_DATA_DISK` if yours differs from `/mnt`.

Manual setup (if you ever prefer to do it yourself):

```bash
conda create -y -p /mnt/conda/envs/chemprop python=3.11
conda activate /mnt/conda/envs/chemprop
pip install torch                                   # or: pip install torch --index-url https://download.pytorch.org/whl/cu121
cd chemprop && pip install -e . && cd ..
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import chemprop; print('Chemprop', chemprop.__version__)"
```

> **Never** install chemprop into the `qpred` env: chemprop 2.x requires Python ≥ 3.10 and its pinned dependencies (lightning, torch ≥ 2.1, …) can break qpred's own environment. That's exactly why the bootstrap builds a separate one.

---

## 4. Dataset & Splits

Splits match QPred exactly (seed 42 via `np.random.permutation`, valid SMILES only, 110k train / 10k val / rest test). They are generated automatically if missing, or manually:

```bash
cd compare && python create_split.py && cd ..
```

This creates `compare/train.csv`, `compare/val.csv`, `compare/test.csv` (plus `*_ids.txt`).

---

## 5. Monitoring & Managing a Run

All artifacts live in `logs/`:

| File | Contents |
|---|---|
| `logs/<prop>_progress.log` | **Clean per-epoch progress + final summary (watch this)** |
| `logs/<prop>_latest.log` | Raw chemprop output (symlink to the timestamped log) |
| `logs/<prop>_<timestamp>.log` | Raw log of a specific run |
| `logs/<prop>_monitor.err` | Errors from the progress monitor (should stay empty) |
| `logs/<prop>.pid`, `logs/<prop>_monitor.pid` | PIDs of trainer / monitor |

```bash
tail -f logs/Polarizability_progress.log   # clean progress (one line per epoch)
tail -f logs/Polarizability_latest.log     # raw chemprop output
ps -p $(cat logs/Polarizability.pid)       # is it still running?
pgrep -fl "chemprop"                       # all chemprop processes
watch -n 1 nvidia-smi                      # GPU usage
./stop_chemprop.sh Polarizability          # stop cleanly (SIGTERM, saves last state)
```

The progress log ends with a summary block: total epochs completed, best validation MAE and its epoch, and the final **TEST-set MAE/MSE** that chemprop prints after training. Per-molecule test predictions are saved automatically to:

```
checkpoints/<prop>/model_0/test_predictions.csv
```

---

## 6. Supported QM40 Target Properties

All 13 quantum properties present in the split CSVs:

| # | Property Name | Physical Meaning | Units |
|---|-------------------------|------------------|-------|
| 1 | `Polarizability` | Polarizability (Benchmark Default) | Bohr³ |
| 2 | `dipol_mom` | Dipole Moment | Debye |
| 3 | `HOMO` | Highest Occupied Molecular Orbital | Hartree |
| 4 | `LUMO` | Lowest Unoccupied Molecular Orbital | Hartree |
| 5 | `HL_gap` | HOMO-LUMO Energy Gap | Hartree |
| 6 | `"spatial extent"` | Electronic Spatial Extent *(use quotes)* | Bohr² |
| 7 | `ZPE` | Zero-Point Vibrational Energy | kcal/mol |
| 8 | `"Internal_E(0K)"` | Internal Energy at 0 Kelvin | Hartree |
| 9 | `"Inter_E(298)"` | Internal Energy at 298.15 Kelvin | Hartree |
| 10 | `Enthalpy` | Enthalpy at 298.15 Kelvin | Hartree |
| 11 | `Free_E` | Gibbs Free Energy at 298.15 Kelvin | Hartree |
| 12 | `CV` | Heat Capacity at 298.15 Kelvin | cal/mol·K |
| 13 | `Entropy` | Entropy at 298.15 Kelvin | cal/mol·K |

---

## 7. Post-Training Evaluation & QPred Comparison

### Option A (fastest): use the predictions training already wrote

```bash
python compare/chemprop_qpred_style_mae.py \
    --train-path compare/train.csv \
    --test-path compare/test.csv \
    --predictions-path checkpoints/Polarizability/model_0/test_predictions.csv \
    --target-column Polarizability
```

The script auto-detects the prediction column and joins on `Zinc_id` when available or on the `smile` column otherwise. It reports MAE / RMSE in both the **QPred-normalized convention** (train-set mean/MAD) and **physical units**.

### Option B: run `chemprop predict` yourself (produces `pred_0` layout)

```bash
python -m chemprop.cli.main predict \
    -i compare/test.csv \
    --smiles-columns smile \
    --model-paths checkpoints/Polarizability/model_0/best.pt \
    -o compare/chemprop_predictions.csv
```

> **Important:** the correct module invocation is `python -m chemprop.cli.main predict` (or just the `chemprop` console script). The older `python -m chemprop.cli train` form crashes because chemprop has no `cli/__main__.py`.

Then compute the QPred-style MAE:

```bash
python compare/chemprop_qpred_style_mae.py \
    --train-path compare/train.csv \
    --test-path compare/test.csv \
    --predictions-path compare/chemprop_predictions.csv \
    --target-column Polarizability
```

### Side-by-side vs QPred

Put QPred's test predictions in `compare/qpred_predictions.csv` with columns `Zinc_id,actual,predicted`, then:

```bash
# Default (files in compare/, label Polarizability):
cd compare && python compare_predictions.py && cd ..

# Any other property — pass the label and (optionally) column overrides:
cd compare && python compare_predictions.py --target-property HOMO && cd ..
```

The script auto-detects the prediction columns (`predicted`/`pred_0`/target-name for QPred and Chemprop files respectively); see `--help` for overrides.

---

## 8. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| First run takes long before epoch 1 appears | Normal: torch/chemprop install + featurization of 110k molecules. Check `tail -f logs/<prop>_latest.log`. |
| `CUDA available: False` | CPU-only torch got installed. Reinstall with the right index: `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./run_chemprop_bg.sh ...` (after `./stop_chemprop.sh` + deleting the env). |
| CUDA OOM | Lower batch size: `./run_chemprop_bg.sh Polarizability 100 32`. |
| Run won't start: "already running" | A previous run for that property is alive: `./stop_chemprop.sh <prop>` first. |
| Wrong split files / want to re-split | Delete `compare/train.csv`, `compare/val.csv`, `compare/test.csv` and re-run (auto-regenerates with seed 42). |
| Want a fresh environment | `conda env remove -p /mnt/conda/envs/chemprop` (or `rm -rf .venv-chemprop`, or delete your `CHEMPROP_ENV_DIR` path) and re-run the launcher. |
| Monitor stopped but training alive | Monitor logs go to `logs/<prop>_monitor.err`; training is unaffected — check the raw log. |
| `[Errno 28] No space left on device` / low disk | The launcher pre-checks **~10 GiB** on the env filesystem and **~4 GiB** on the pip-cache filesystem *before* downloading anything, and aborts with instructions when either is short. Free up space (`pip cache purge`, `conda clean --all`, old checkpoints) or `export CHEMPROP_ENV_DIR=/path/with/room` (any disk you mounted, wherever you like) and re-run. Nothing is downloaded when the check fails. |

### Where the environment lives (disk policy)

The chemprop env is created at an **explicit prefix on your big data disk**: `/mnt/conda/envs/chemprop` by default (override the mountpoint with `CHEMPROP_DATA_DISK`, or the whole path with `CHEMPROP_ENV_DIR`). Because the path is explicit (`conda create -p`), conda settings such as `envs_dirs` can never redirect it to another disk, and the ~7 GB env plus the pip cache (`/mnt/pip-cache`, moved there automatically) live on the ~60 GB data disk instead of the small root disk.

Mounting is **your mount script's job** — the launcher never mounts, unmounts, or names any device. It only *verifies*: before creating anything it checks that `<data-disk>` really is a mounted filesystem and not the root fs showing through an unmounted mountpoint (the classic post-reboot trap). If the disk is not mounted, it aborts with **nothing written** and tells you to run your mount script.

Disk-space needs on the **first** run only:

| Filesystem | Free space required | When checked |
|---|---|---|
| Data disk (env + pip cache) | ~10 GiB | before any download |

If the check fails, the launcher aborts with instructions and downloads **nothing**. After a successful install it runs `pip cache purge` automatically, reclaiming ~2–3 GB.

After a VM reboot:

```bash
./your-mount-script.sh          # your script that mounts the data disk at /mnt
./run_chemprop_bg.sh Polarizability 100
```

The launcher finds the existing env at `/mnt/conda/envs/chemprop` (`torch: already installed`) and starts training with **zero re-downloads**. If you forgot the mount step, it stops safely instead of rebuilding on the root disk.

### Best practices

1. Always launch through the script (or `tmux`/`screen`) when training over SSH — the script already detaches with `nohup` + `disown`.
2. One training per property at a time; use different property names (or rename `checkpoints/<prop>`) for parallel runs on different GPUs.
3. `num_workers > 0` speeds up dataloading on Linux VMs; if you see worker hangs, pass `0` as the 4th argument.
