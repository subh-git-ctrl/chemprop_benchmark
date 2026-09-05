# Chemprop QM40 VM Training & Execution Guide

This document contains all the commands required to set up, train, monitor, and evaluate Chemprop on a VM for the QM40 dataset benchmark.

---

## 1. Quick Start (Background Training on VM)

Run training in the background with `nohup` and `disown` so it **keeps running to completion even after you close your terminal or disconnect from the VM**.

> **Note:** The script trains on the **ENTIRE QM40 dataset** (all **110,000** training molecules, **10,000** validation molecules, and **42,956** test molecules). The default `batch_size` (64 or 32) is simply the **GPU mini-batch size** (how many molecules are fed per gradient step), not the dataset size.

```bash
# Give execution permission
chmod +x run_chemprop_bg.sh run_property_bg.sh

# Train on Polarizability for 100 epochs on the entire dataset:
./run_chemprop_bg.sh Polarizability 100

# Or using the alias:
./run_property_bg.sh Polarizability 100
```

To view live training progress in real time:
```bash
tail -f logs/Polarizability_latest.log
```

---

## 2. VM Environment Setup

### Option A: Using Conda (Recommended)

```bash
# 1. Create a conda environment with Python 3.11
conda create -y -n qpred python=3.11
conda activate qpred

# 2. Install PyTorch with CUDA support (example for CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install Chemprop and its dependencies
cd chemprop
pip install -e .
pip install -r requirements.txt
pip install rdkit pandas numpy scikit-learn

# 4. Verify installation & GPU availability
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python3 -c "import chemprop; print('Chemprop version:', chemprop.__version__)"
```

### Option B: Using Python Virtual Environment (venv)

```bash
cd chemprop
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install -r requirements.txt
pip install rdkit pandas numpy scikit-learn
```

---

## 3. Dataset & Splits Preparation

The dataset splits match the QPred model exactly (Seed = 42, 80% train / 10% val / 10% test):

```bash
# Navigate to the compare folder and generate split files if not already generated
cd compare
python3 create_split.py
cd ..
```

This creates:
- `compare/train.csv` (110,000 molecules)
- `compare/val.csv` (10,000 molecules)
- `compare/test.csv` (remaining molecules)

---

## 4. Background Training Script Usage

The script `run_chemprop_bg.sh` automatically:
- Activates your conda environment (`qpred` or `chemprop`) or venv
- Checks and verifies the train/val/test split files
- Sets `PYTHONUNBUFFERED=1` so logs update in real-time
- Executes Chemprop in the background via `nohup`
- Disowns the process so closing the terminal / SSH will **not** kill training
- Saves process PID in `logs/<property>.pid` for easy management

### Syntax:
```bash
./run_chemprop_bg.sh <property_name> [num_epochs] [batch_size]
```

### Examples:
```bash
# 1. Train on Polarizability (100 epochs, batch size 32)
./run_chemprop_bg.sh Polarizability 100 32

# 2. Train on HOMO (150 epochs)
./run_chemprop_bg.sh HOMO 150

# 3. Train on spatial extent (use quotes for property names with spaces)
./run_chemprop_bg.sh "spatial extent" 100 32

# 4. Train on Internal Energy at 0K
./run_chemprop_bg.sh "Internal_E(0K)" 100 32
```

---

## 5. Monitoring & Managing Background Training

### Monitor Logs in Real-Time:
```bash
# Follow the latest log for Polarizability:
tail -f logs/Polarizability_latest.log

# Or view the last 50 lines:
tail -n 50 logs/Polarizability_latest.log

# Check error log:
cat logs/Polarizability_*.err
```

### Check if Training is Still Running:
```bash
# Check by saved PID:
ps -p $(cat logs/Polarizability.pid)

# Or check all active Chemprop training processes:
pgrep -fl "chemprop.cli train"
```

### Monitor GPU and Resource Usage:
```bash
# Real-time GPU monitoring (updates every second):
watch -n 1 nvidia-smi

# Check CPU and RAM:
htop
```

### Stop / Terminate Training:
```bash
# Stop using the stored PID:
kill $(cat logs/Polarizability.pid)

# Force stop if needed:
kill -9 $(cat logs/Polarizability.pid)
```

---

## 6. Supported QM40 Target Properties

All 13 quantum properties present in `train.csv`, `val.csv`, and `test.csv` can be trained directly:

| # | Property Name in Script | Physical Meaning | Units |
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

## 7. Direct CLI Training Command (Alternative to Script)

If you wish to run the Chemprop CLI command directly in your terminal:

```bash
python3 -m chemprop.cli train \
    -i compare/train.csv compare/val.csv compare/test.csv \
    --smiles-columns smile \
    --target-columns Polarizability \
    --task-type regression \
    --output-dir checkpoints/Polarizability \
    --epochs 100 \
    --batch-size 32 \
    --metrics mae mse \
    --accelerator auto
```

---

## 8. Post-Training Evaluation & Predictions

After training completes, the best model weights are saved at:
`checkpoints/<property_name>/model_0/best.pt`

### Step 1: Generate Test Predictions with Chemprop
```bash
python3 -m chemprop.cli predict \
    -i compare/test.csv \
    --smiles-columns smile \
    --model-paths checkpoints/Polarizability/model_0/best.pt \
    -o compare/chemprop_predictions.csv
```

### Step 2: Compute QPred-Style Normalized MAE & Physical MAE
```bash
python3 compare/chemprop_qpred_style_mae.py \
    --train-path compare/train.csv \
    --test-path compare/test.csv \
    --predictions-path compare/chemprop_predictions.csv \
    --target-column Polarizability \
    --id-column Zinc_id \
    --prediction-column prediction
```

### Step 3: Run Direct Side-by-Side Comparison against QPred
Ensure `compare/qpred_predictions.csv` is present with columns `Zinc_id,actual,predicted`:
```bash
cd compare
python3 compare_predictions.py
```

---

## 9. VM Best Practices & Tips

1. **Keep processes alive:** Always use `./run_chemprop_bg.sh` or `tmux`/`screen` when training over SSH.
2. **Log rotation:** Each training run creates a timestamped log file `logs/<property>_<timestamp>.log` and symlinks `logs/<property>_latest.log` for convenience.
3. **Multi-GPU selection:** If your VM has multiple GPUs, you can target a specific GPU by prefixing:
   ```bash
   CUDA_VISIBLE_DEVICES=0 ./run_chemprop_bg.sh Polarizability 100
   ```
4. **Resuming terminal after logout:**
   Simply SSH back into the VM and run:
   ```bash
   cd chemprop-benchmark
   tail -f logs/Polarizability_latest.log
   ```
