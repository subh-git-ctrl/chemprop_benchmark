# Benchmark Execution Checklist

## Pre-Training Checklist

- [ ] **Verify Directory Structure**

  ```
  chemprop-benchmark/
  ├── data_qm40/           → QM40 CSV files (bond.csv, main.csv, xyz.csv)
  ├── chemprop/            → Chemprop repository (cloned)
  └── compare/             → Benchmark utilities
      ├── create_split.py
      ├── compare_predictions.py
      ├── README.md
      └── CHECKLIST.md (this file)
  ```

- [ ] **Verify Data Split Script**
  - Run: `cd compare && python create_split.py`
  - Check that three files are created:
    - `train.csv` - Training set
    - `val.csv` - Validation set
    - `test.csv` - Test set
  - Verify split sizes match expected (80/10/10)

- [ ] **Prepare Chemprop Input**
  - Run create_split.py to generate CSV files
  - Files created should be:
    - `compare/train.csv`
    - `compare/val.csv`
    - `compare/test.csv`

## Chemprop Training Checklist

- [ ] **Create and Activate Virtual Environment**

  ```bash
  cd chemprop
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```

- [ ] **Install Dependencies**

  ```bash
  pip install -e .
  pip install -r requirements.txt
  ```

- [ ] **Verify Installation**

  ```bash
  python -c "import chemprop; print(chemprop.__version__)"
  ```

- [ ] **Prepare Training Data**
  - Create combined CSV with SMILES, target (Polarizability), and split columns
  - File format:
    ```
    smile,Polarizability,split
    CC(C)C,...,train
    CN(C)C(=O)CNC1,...,val
    ...
    ```
  - Location: `../compare/chemprop_data.csv`

- [ ] **Launch Training**
  - Command:
    ```bash
    python scripts/train.py \
      --data_path ../compare/chemprop_data.csv \
      --dataset_type regression \
      --task_names Polarizability \
      --save_dir checkpoints/qm40_polarizability \
      --seed 42 \
      --num_folds 1 \
      --batch_size 32 \
      --epochs 100 \
      --lr 1e-4 \
      --metrics mse
    ```
  - Monitor training output (loss, metrics)
  - Verify model saves to: `checkpoints/qm40_polarizability/model.pt`

- [ ] **Training Completed**
  - Verify checkpoint file exists
  - Note final training metrics (MSE/MAE on train/val)
  - Record training time and computational resources used

## Prediction Generation Checklist

- [ ] **Generate Test Predictions**

  ```bash
  python scripts/predict.py \
    --test_path ../compare/test.csv \
    --checkpoint_path checkpoints/qm40_polarizability/model.pt \
    --output_path ../compare/chemprop_predictions.csv
  ```

  - File should contain columns: `smiles`, `Polarizability`, `prediction`

- [ ] **Verify Predictions File**
  - Location: `compare/chemprop_predictions.csv`
  - Expected columns: SMILES, target values, predictions
  - Number of rows should match test set size
  - No NaN values in predictions

## QPred Model Extraction Checklist

- [ ] **Extract QPred Test Predictions**
  - Get QPred test set predictions for Polarizability
  - Save as CSV with columns: `Zinc_id`, `actual`, `predicted`
  - Location: `compare/qpred_predictions.csv`
  - Ensure same test molecules as Chemprop
  - **Important:** Denormalize if QPred predictions are normalized!
    - Check QPred's data.py for normalization constants
    - Apply inverse transformation: `value = (normalized - mean) * mad`

## Comparison & Results Checklist

- [ ] **Run Comparison Script**

  ```bash
  cd compare
  python compare_predictions.py
  ```

- [ ] **Document Comparison Results**
  - Record both models' test set metrics:
    - **QPred MAE:** **\_\_\_**
    - **Chemprop MAE:** **\_\_\_**
    - **RMSE (QPred):** **\_\_\_**
    - **RMSE (Chemprop):** **\_\_\_**
    - **Correlation (QPred):** **\_\_\_**
    - **Correlation (Chemprop):** **\_\_\_**
    - **R² (QPred):** **\_\_\_**
    - **R² (Chemprop):** **\_\_\_**

- [ ] **Analyze Differences**
  - [ ] Difference is due to:
    - [ ] Split mismatch? (Verify using same seed=42 splits)
    - [ ] Target normalization? (Check raw units)
    - [ ] Model architecture? (Document both architectures)
    - [ ] Hyperparameters? (Record all settings)
    - [ ] Data preprocessing? (Check feature engineering)

- [ ] **Save Detailed Report**
  - File: `compare/benchmark_report.txt`
  - Should include:
    - QPred architecture and hyperparameters
    - Chemprop architecture and hyperparameters
    - Train/val/test split sizes
    - Random seed (42)
    - Target property (Polarizability)
    - Comparison metrics
    - MAE difference and % improvement
    - Analysis of difference source

## Issue Troubleshooting Checklist

### If Split Sizes Don't Match QPred

- [ ] Verify `create_split.py` uses `np.random.seed(42)`
- [ ] Check that invalid SMILES are filtered
- [ ] Ensure same `main.csv` file (no modifications)
- [ ] Re-run: `python create_split.py`

### If Chemprop Training Fails

- [ ] Verify CSV format (columns: SMILES, target, split)
- [ ] Check no NaN values in Polarizability column
- [ ] Ensure SMILES are valid (can parse with RDKit)
- [ ] Try reduced batch size: `--batch_size 16`
- [ ] Verify Chemprop installation: `pip install -e .`

### If Prediction Comparison Fails

- [ ] Verify CSV files have matching molecule counts
- [ ] Check column names in compare_predictions.py
- [ ] Ensure predictions are in raw units, not normalized
- [ ] Verify test set Zinc_ids are identical

### If MAE Difference is Large

- [ ] **Check Normalization:**
  - QPred uses mean/MAD normalization during training
  - Must denormalize predictions before comparison!
  - Formula: `raw_value = (normalized_value * mad) + mean`

- [ ] **Check Train/Val/Test Split:**
  - Verify both models use exact same split
  - Check seed=42 was used
  - Ensure test sets have same molecules

- [ ] **Check Target Property:**
  - Verify both models predict Polarizability
  - Check units match (Bohr³ in QM40)
  - No unit conversions applied

## Final Documentation

### Configuration Record

- **Target Property:** Polarizability (Bohr³)
- **Random Seed:** 42
- **Train/Val/Test Split:** 80% / 10% / 10%
- **Dataset:** QM40 (~163k molecules, max 40 heavy atoms)
- **Test Set Size:** [Number from create_split.py output]

### Chemprop Configuration

- **Architecture:** [Default or specify modifications]
- **Batch Size:** 32
- **Learning Rate:** 1e-4
- **Epochs:** 100
- **Optimizer:** [Specify from Chemprop output]
- **Activation Functions:** [ReLU, etc.]

### Results Summary

- **QPred MAE:** **\_\_\_** (Bohr³)
- **Chemprop MAE:** **\_\_\_** (Bohr³)
- **Better Model:** [QPred / Chemprop]
- **Improvement:** **\_\_\_** %
- **Root Cause of Difference:** [explanation]

### Reproducibility Notes

- All split files saved: `compare/train.csv`, `compare/val.csv`, `compare/test.csv`
- Predictions saved: `compare/chemprop_predictions.csv`, `compare/qpred_predictions.csv`
- Report saved: `compare/benchmark_report.txt`
- Repository state: Git commit hash of Chemprop used
- Python version: [e.g., 3.11]
- PyTorch version: [e.g., 2.0]

---

**Status:** [ ] Complete [ ] In Progress [ ] Blocked

**Notes:**

```
[Add any special notes, issues, or observations here]
```
