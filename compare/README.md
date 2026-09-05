# QM40 Chemprop Benchmark Setup

## Overview

This folder contains an isolated benchmark workspace for comparing your custom QPred QM40 model against Chemprop on the same dataset with identical train/val/test splits.

## Directory Structure

```
chemprop-benchmark/
├── data_qm40/              ← QM40 dataset (CSV files)
│   ├── main.csv            ← Molecules and quantum properties
│   ├── bond.csv            ← Bond information
│   └── xyz.csv             ← 3D coordinates
├── chemprop/               ← Cloned Chemprop repository
│   └── ... (chemprop source code)
└── compare/                ← Benchmark utilities and results
    ├── create_split.py     ← Data split script (seed=42, matching QPred)
    ├── train_ids.txt       ← Train set molecule IDs
    ├── val_ids.txt         ← Val set molecule IDs
    ├── test_ids.txt        ← Test set molecule IDs
    ├── train.csv           ← Train set CSV
    ├── val.csv             ← Val set CSV
    ├── test.csv            ← Test set CSV
    └── README.md           ← This file
```

## Key Setup Details

### Data Split (Matching QPred Model)

- **Random Seed:** 42 (using `np.random.permutation`)
- **Split Ratio:** 80% train / 10% val / 10% test
- **Capped Sizes:**
  - Max train: 110,000 molecules
  - Max val: 10,000 molecules
  - Rest goes to test
- **Validation:** All indices use valid (parseable) SMILES only

### Dataset Properties

The QM40 dataset contains 13 quantum mechanical properties:

1. `dipol_mom` - Dipole moment (Debye)
2. `Polarizability` - Polarizability (Bohr³) ← **Default target for benchmark**
3. `HOMO` - HOMO energy (Hartree)
4. `LUMO` - LUMO energy (Hartree)
5. `HL_gap` - HOMO-LUMO gap (Hartree)
6. `spatial extent` - Electronic spatial extent (Bohr²)
7. `ZPE` - Zero-point energy (kcal/mol)
8. `Internal_E(0K)` - Internal energy at 0K (Hartree)
9. `Inter_E(298)` - Internal energy at 298K (Hartree)
10. `Enthalpy` - Enthalpy at 298K (Hartree)
11. `Free_E` - Free energy at 298K (Hartree)
12. `CV` - Heat capacity at 298K (cal/mol·K)
13. `Entropy` - Entropy at 298K (cal/mol·K)

**Note:** If your QPred model uses normalized or scaled values, convert to raw units before comparison.

## Step-by-Step Benchmark Workflow

### 1. Create Train/Val/Test Split

```bash
cd chemprop-benchmark/compare
python create_split.py
```

This will create:

- `train_ids.txt`, `val_ids.txt`, `test_ids.txt` - Molecule IDs for each split
- `train.csv`, `val.csv`, `test.csv` - Full CSV files for each split

Sizes will match your QPred model exactly (same seed, same split method).

### 2. Set Up Chemprop Environment

```bash
cd chemprop-benchmark/chemprop

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows

# Install dependencies
pip install -e .
pip install -r requirements.txt
```

### 3. Prepare Chemprop Input Files

Convert the split CSVs to Chemprop format:

- Chemprop expects SMILES, target property, and split information
- Use the `compare/train.csv`, `compare/val.csv`, `compare/test.csv`

Create a Chemprop-compatible input CSV:

```csv
smiles,Polarizability,split
CC(C)C,...,train
CN(C)C(=O)CNC1,...,val
...
```

### 4. Train Chemprop on Polarizability

```bash
cd chemprop

python scripts/train.py \
  --data_path ../compare/chemprop_data.csv \
  --dataset_type regression \
  --task_names Polarizability \
  --save_dir checkpoints/qm40_polarizability \
  --seed 42 \
  --num_folds 1 \
  --batch_size 32 \
  --epochs 100 \
  --lr 1e-4
```

### 5. Generate Test Predictions

```bash
python scripts/predict.py \
  --test_path ../compare/test.csv \
  --checkpoint_path checkpoints/qm40_polarizability/model.pt \
  --output_path ../compare/chemprop_predictions.csv
```

### 6. Compare Results

Compare Chemprop MAE with your QPred model:

```python
import pandas as pd
import numpy as np

# Load predictions
qpred_preds = pd.read_csv('qpred_test_predictions.csv')
chemprop_preds = pd.read_csv('compare/chemprop_predictions.csv')

# Calculate MAE
qpred_mae = np.mean(np.abs(qpred_preds['predicted'] - qpred_preds['actual']))
chemprop_mae = np.mean(np.abs(chemprop_preds['predicted'] - chemprop_preds['actual']))

print(f"QPred MAE: {qpred_mae:.4f}")
print(f"Chemprop MAE: {chemprop_mae:.4f}")
print(f"Difference: {abs(qpred_mae - chemprop_mae):.4f}")
```

## Important Notes

### Unit Conversion

⚠️ **Critical:** Ensure target values are in the same units:

- QM40 raw units (Bohr³ for Polarizability)
- If your QPred model uses normalized values, denormalize before comparison
- Use training set statistics (mean, MAD) from `data.py` if needed

### Reproducibility Checklist

- [ ] Random seed = 42 (both models)
- [ ] Same train/val/test split (use files in `compare/`)
- [ ] Same target property and units
- [ ] Same test set
- [ ] Both models evaluated on raw (non-normalized) values

### Commands Quick Reference

```bash
# Setup
cd chemprop-benchmark/compare
python create_split.py

# Train Chemprop (example)
cd ../chemprop
python scripts/train.py --data_path ../compare/chemprop_data.csv \
  --dataset_type regression --task_names Polarizability \
  --seed 42 --num_folds 1

# Predict
python scripts/predict.py --test_path ../compare/test.csv \
  --checkpoint_path checkpoints/qm40_polarizability/model.pt
```

## Troubleshooting

| Issue                       | Solution                                                              |
| --------------------------- | --------------------------------------------------------------------- |
| SMILES parsing errors       | Ensure RDKit is installed: `pip install rdkit`                        |
| Split sizes mismatch        | Verify seed=42 and check valid SMILES count in create_split.py output |
| Unit mismatch in comparison | Check denormalization constants in QPred's data.py                    |
| Chemprop train fails        | Ensure data format matches: SMILES, target, split columns             |

## Results Documentation

After training, document:

1. **Chemprop Configuration**
   - Architecture (default or custom)
   - Hyperparameters (LR, batch size, epochs)
   - Random seed (should be 42)

2. **Performance Metrics**
   - Test set MAE (Chemprop)
   - Test set MAE (QPred)
   - Difference and % improvement

3. **Data Verification**
   - Train/val/test split sizes
   - Random seed used
   - Target property and units
   - Any data preprocessing steps

## References

- **Chemprop GitHub:** https://github.com/chemprop/chemprop
- **QM40 Dataset:** https://quantum-machine.org/datasets/
- **QPred Project:** Parent directory (`qpred-app/`)

---

**Note:** This benchmark folder is completely isolated from the QPred project. No QPred code is imported or modified here.
