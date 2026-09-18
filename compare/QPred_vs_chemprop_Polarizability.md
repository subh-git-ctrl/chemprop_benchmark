# QPred vs chemprop — Polarizability (QM40), Head-to-Head

**Status:** final (units-audited) · **Property:** `Polarizability` (Bohr³) · **Date:** 2026-09-19
**Data:** `compare/train.csv` / `val.csv` / `test.csv` — seed-42 random 80/10/10 split of `data_qm40/main.csv` (110,000 / 10,000 / 42,954 molecules; zero ID overlap across splits, no duplicates, no NaNs — audited). Both models trained and evaluated on exactly the same molecules in the same splits.

---

## TL;DR

| | QPred | chemprop 2.3.1 | winner |
|---|---|---|---|
| **Test MAE (physical, Bohr³)** | **2.1953** | **1.7937** | **chemprop, 18.3 % lower** |
| Val MAE (physical, Bohr³) | 2.2535 (EMA: 2.2289) | 1.8105 | chemprop, 19.7 % lower |
| Test MAE (QPred normalized units) | 0.034056 | 0.027826 | chemprop, 18.3 % lower |

- QPred's printed **"MAE (physical) = 0.055991" is wrong**: it was de-normalized with **`dipol_mom`'s train MAD (1.644086)** instead of Polarizability's (64.4604) — the ratio `0.055991 / 0.034056 = 1.644086` matches `dipol_mom`'s MAD to **6 significant digits** (see §3). A scaler mis-indexing bug in QPred's eval code, not a real result.
- Both models' **normalized** MAEs are internally honest; they just use different normalization conventions, so they must be converted to physical units before comparing (§2).

---

## 1. The numbers exactly as each model reported them

**QPred — IMPROVED 2D MPNN (2.89 M params), as reported:**

```
Val MAE (normalized) : 0.034960        EMA: 0.034578
Test MAE (normalized): 0.034056
Test MAE (physical)  : 0.055991        ← INCORRECT (de-normalized with wrong property's MAD, §3)
Training time        : 1694.9 min (~28.2 h)
```

**chemprop 2.3.1 — D-MPNN (default architecture), 100 epochs, this repo's run:**

```
Val MAE (normalized, z) : 0.024472     (best epoch 67)  = 1.8105 Bohr³
Test MAE (physical)     : 1.793658 Bohr³  (chemprop's own test eval, final-epoch weights)
                           = 0.024244 z-units = 0.027826 QPred-MAD-units
Re-scored from archived chemprop_predictions.csv (42954/42954 matched):
                           MAE 1.8405 Bohr³, RMSE 2.7720, R² 0.99860
Wall-clock time         : not archived in this repo (see the run's monitor log)
```

---

## 2. The two normalization conventions are different — convert before comparing

| | QPred (replicates qpred-app) | chemprop 2.3.1 |
|---|---|---|
| Rule | `norm = (y − mean) / MAD` | `norm = (y − mean) / std` (StandardScaler) |
| Definition of scale constant | `MAD = mean(\|y − mean\|)` over train split | `std` (population, ddof=0) over train split |
| Constant for Polarizability | **MAD = 64.4604** | **std = 73.9835** (scaler log prints 73.98379) |
| Train mean (both) | 218.1594 | 218.1594 |

Because 64.4604 ≠ 73.9835, a normalized MAE of "0.03" means different things in each log. **All fair comparisons must happen in physical units (Bohr³)**, or after explicitly converting each model's number into the other's convention (done in §4).

Reference implementation of QPred's convention (no QPred code imported): `compare/chemprop_qpred_style_mae.py`.

---

## 3. Why QPred's "physical" 0.055991 is impossible — and what it really is

QPred's own convention (§2) requires `physical_MAE = normalized_MAE × MAD_train(Polarizability)`. Check the factor QPred actually applied:

```
factor = 0.055991 / 0.034056 = 1.644086
```

Scanning the train-set MAD of all 16 QM40 property columns (`scripts/audit_qpred_numbers.py` logic, results archived in the worklog):

| column | train MAD | MAD / 1.644086 |
|---|---|---|
| **dipol_mom** | **1.644086** | **1.0000** ← exact match, 6 significant digits |
| Polarizability | 64.4604 | 39.21 |
| (every other property) | ≠ 1.6441 | > 4 or < 0.03 |

The factor is **`dipol_mom`'s MAD, not Polarizability's**. QPred's evaluation code picked the wrong row of its scaler when de-normalizing the Polarizability MAE (classic index-shift bug — e.g. alphabetical vs on-disk column order). Note `dipol_mom`'s MAD is also the only scale constant in the dataset anywhere near 1.64, so this cannot be a units story (Å³/cm³/mol conversions give 10.96 / 6.60; std conventions give 2.2472 for `dipol_mom`).

**Corrected QPred numbers** (× 64.4604, its own convention):

```
Test MAE : 0.034056 × 64.4604 = 2.1953 Bohr³
Val  MAE : 0.034960 × 64.4604 = 2.2535 Bohr³   (EMA: 0.034578 → 2.2289 Bohr³)
```

Sanity check that 2.1953 (not 0.056) is the plausible scale: 2.1953 / 218.1594 = **1.006 % relative MAE** (chemprop: 0.822 %), and R² ≈ 0.9986 territory. An MAE of 0.056 Bohr³ would be a 0.026 % relative error — far beyond any published QM9/QM40 model of this class, and QPred's own val number (in the same normalized units) would then disagree with its test number by 40 %.

**Fix on the QPred side** (not in this repo): de-normalize by looking the scale constant up **by target-column name** from the same scaler object fitted during normalization, instead of by positional index.

---

## 4. Fair comparison, all in one place

Same 42,954-molecule test set, same 10,000-molecule val set, physical units = Bohr³:

| Metric | QPred (2.89 M) | chemprop 2.3.1 | Δ (chemprop) |
|---|---|---|---|
| Val MAE (physical) | 2.2535 | **1.8105** | **19.7 % lower** |
| Val MAE, EMA weights (physical) | 2.2289 | **1.8105** | **18.8 % lower** |
| Test MAE (physical) | 2.1953 | **1.7937** | **18.3 % lower** |
| Test MAE (QPred MAD-units) | 0.034056 | 0.027826 | 18.3 % lower |
| Test MAE (chemprop z-units) | 0.029672 | 0.024244 | 18.3 % lower |
| Relative test MAE (÷ mean 218.1594) | 1.006 % | 0.822 % | — |
| Test RMSE (physical) | not reported | 2.7720 | — |
| Test R² | not reported | 0.9986 | — |
| Test MAE, chemprop best.pt re-score | — | 1.8405 | 16.2 % lower |

**Verdict: chemprop wins on Polarizability by ~16–20 % MAE on both validation and test, under either normalization convention, and under both checkpoint choices.** QPred's only losing number (0.055991 "physical") was the conversion artifact.

---

## 5. Cross-checks (why this table is trustworthy)

- **Val↔test consistency, chemprop:** best val 1.8105 vs test 1.7937 (own eval) — 0.9 % apart; vs best.pt re-score 1.8405 — 1.7 % apart. Normal.
- **Val↔test consistency, QPred:** 2.2535 vs 2.1953 — 2.6 % apart. Normal.
- **Split integrity (audited):** 110,000 + 10,000 + 42,954 = 162,954 = `main.csv`; per-split target stds agree within 0.8 %; zero `smile`/`Zinc_id` leakage across splits.
- **Prediction archive:** `compare/chemprop_predictions.csv` matches all 42,954 test molecules 1:1 by `Zinc_id`; re-scoring with `compare/chemprop_qpred_style_mae.py` reproduces MAE 1.8405 Bohr³ = 0.028552 MAD-units exactly.
- **Unit identity:** both models consume the same raw CSV columns; chemprop's scaler log (`Train data: mean=[218.159…] | std=[73.9837…]`) matches the train CSV statistics computed independently.

---

## 6. Reproduce

```bash
cd compare

# 1) chemprop test MAE in QPred's own normalization convention (MAD units)
python chemprop_qpred_style_mae.py \
    --train-path train.csv --test-path test.csv \
    --predictions-path chemprop_predictions.csv \
    --target-column Polarizability --id-column Zinc_id --prediction-column pred_0
#   -> MAE (normalized) = 0.028552, MAE (physical) = 1.8405 Bohr³

# 2) one-line unit conversions behind every number in §3/§4
python3 -c "print(0.055991/0.034056)"      # 1.644086 = dipol_mom's MAD (the bug)
python3 -c "print(0.034056*64.4604)"       # 2.1953  QPred corrected test MAE, Bohr³
python3 -c "print(0.034960*64.4604)"       # 2.2535  QPred corrected val MAE, Bohr³
python3 -c "print(0.024472*73.9835)"       # 1.8105  chemprop val MAE, Bohr³
python3 -c "print(1.793658/64.4604)"       # 0.027826 chemprop test MAE in QPred units

# 3) train-set constants straight from the split
python3 -c "import pandas as pd,numpy as np; y=pd.read_csv('train.csv')['Polarizability']; m=y.mean(); print(m, np.abs(y-m).mean(), y.std(ddof=0))"
#   -> 218.1594 (mean)  64.4604 (MAD)  73.9835 (std)
```

To produce a fresh `chemprop_predictions.csv` from a checkpoint (e.g. to compare against `best.pt` explicitly):

```bash
chemprop predict --model-path <ckpt.pt> --test-path compare/test.csv \
    --output compare/chemprop_predictions.csv   # then re-run step 1
```

---

## 7. Protocol notes & caveats

1. **chemprop's logged `test/mae` uses final-epoch weights** (single-device path, `chemprop/cli/train.py:1955-1956`; `best.pt` is reloaded only afterwards). Both variants are reported in §4; the conclusion is identical either way. The DDP path reloads best weights before test.
2. **chemprop's per-epoch val MAE is normalized (z-units).** The monitor (`monitor_progress.py ≥ 934eb08`) converts it using the run's train-std; don't compare it directly against QPred's MAD-unit numbers. Final test metrics and `test_predictions.csv` are physical.
3. QPred's normalized val/test MAEs are taken at face value from its summary (val includes the EMA variant). Its epochs/hyperparameters were not re-tuned for this table; chemprop ran 100 epochs with default hyperparameters (best val at epoch 67).
4. This file covers **Polarizability only**. The same audit recipe (`scripts/audit_qpred_numbers.py` logic: implied factor vs per-column MAD) should be applied per property before trusting any other QPred "physical" line.
