#!/bin/bash
# ==============================================================================
# Script: run_chemprop_bg.sh
# Purpose: Train Chemprop on QM40 dataset in the background on a VM
#          Runs continuously until the last epoch even if SSH / VM session is closed.
#
# Usage:
#   ./run_chemprop_bg.sh <property_name> [num_epochs] [batch_size]
#
# Note:
#   This trains on the ENTIRE dataset (all 110,000 training molecules, 10,000 val, 42,956 test).
#   'batch_size' (default 64) is only the GPU mini-batch size per optimization step.
#
# Example:
#   ./run_chemprop_bg.sh Polarizability 100
#   ./run_chemprop_bg.sh HOMO 100
#   ./run_chemprop_bg.sh "spatial extent" 100
# ==============================================================================

# Ensure script directory is the base
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Print usage if no argument provided
if [ $# -eq 0 ]; then
    echo "=================================================================="
    echo " Chemprop VM Background Training Script"
    echo "=================================================================="
    echo "Usage:"
    echo "  $0 <property_name> [num_epochs] [batch_size]"
    echo ""
    echo "  Trains on the ENTIRE QM40 dataset (~163,000 total molecules):"
    echo "    - Training set   : 110,000 molecules"
    echo "    - Validation set : 10,000 molecules"
    echo "    - Test set       : 42,956 molecules"
    echo ""
    echo "Available QM40 Properties:"
    echo "  1)  Polarizability    (Bohr^3 - Default benchmark target)"
    echo "  2)  dipol_mom         (Debye)"
    echo "  3)  HOMO              (Hartree)"
    echo "  4)  LUMO              (Hartree)"
    echo "  5)  HL_gap            (Hartree)"
    echo "  6)  \"spatial extent\"  (Bohr^2 - Note: use quotes for spaces)"
    echo "  7)  ZPE               (kcal/mol)"
    echo "  8)  \"Internal_E(0K)\"  (Hartree)"
    echo "  9)  \"Inter_E(298)\"    (Hartree)"
    echo "  10) Enthalpy          (Hartree)"
    echo "  11) Free_E            (Hartree)"
    echo "  12) CV                (cal/mol·K)"
    echo "  13) Entropy           (cal/mol·K)"
    echo ""
    echo "Examples:"
    echo "  $0 Polarizability 100"
    echo "  $0 Polarizability 100 64"
    echo "=================================================================="
    exit 1
fi

PROPERTY="$1"
EPOCHS="${2:-100}"
BATCH_SIZE="${3:-64}"

# Create safe filename identifier for logs & checkpoints (remove spaces/parentheses)
SAFE_NAME=$(echo "$PROPERTY" | tr ' ()' '___')

# Directories
LOG_DIR="$SCRIPT_DIR/logs"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints/${SAFE_NAME}"
mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/${SAFE_NAME}_${TIMESTAMP}.log"
ERR_FILE="$LOG_DIR/${SAFE_NAME}_${TIMESTAMP}.err"
PID_FILE="$LOG_DIR/${SAFE_NAME}.pid"
LATEST_LOG="$LOG_DIR/${SAFE_NAME}_latest.log"

# ------------------------------------------------------------------------------
# 1. Environment Activation (Conda or Virtualenv)
# ------------------------------------------------------------------------------
ENV_FOUND=0

# Try Conda environments (qpred or chemprop)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    ENV_FOUND=1
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    ENV_FOUND=1
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    ENV_FOUND=1
fi

if [ $ENV_FOUND -eq 1 ]; then
    # Prioritize chemprop environment (Python 3.11+), fallback to qpred
    if conda info --envs | grep -q "^chemprop "; then
        conda activate chemprop
        echo "[Environment] Activated conda environment: chemprop"
    elif conda info --envs | grep -q "^qpred "; then
        conda activate qpred
        echo "[Environment] Activated conda environment: qpred"
    fi
fi

# Fallback to local python venv if conda is not activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    if [ -f "$SCRIPT_DIR/chemprop/venv/bin/activate" ]; then
        source "$SCRIPT_DIR/chemprop/venv/bin/activate"
        echo "[Environment] Activated venv: chemprop/venv"
    elif [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
        source "$SCRIPT_DIR/venv/bin/activate"
        echo "[Environment] Activated venv: ./venv"
    fi
fi

echo "[Environment] Python executable: $(which python3 || which python)"

# Validate Python version >= 3.10
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" &>/dev/null; then
    CURRENT_PY=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "unknown")
    echo "=================================================================="
    echo " ERROR: Python $CURRENT_PY is incompatible with Chemprop."
    echo " Chemprop requires Python 3.10 or higher."
    echo "=================================================================="
    exit 1
fi

# Check if chemprop is installed in the active environment; install if missing
if ! python3 -c "import chemprop" &>/dev/null; then
    echo "[Setup] chemprop not found in current environment. Installing with 'pip install -e chemprop'..."
    pip install -e "$SCRIPT_DIR/chemprop"
fi

# Determine Chemprop executable command
if command -v chemprop &> /dev/null; then
    TRAIN_RUNNER="chemprop train"
else
    TRAIN_RUNNER="python3 -m chemprop.cli train"
fi

# ------------------------------------------------------------------------------
# 2. Dataset Verification
# ------------------------------------------------------------------------------
TRAIN_CSV="$SCRIPT_DIR/compare/train.csv"
VAL_CSV="$SCRIPT_DIR/compare/val.csv"
TEST_CSV="$SCRIPT_DIR/compare/test.csv"

if [ ! -f "$TRAIN_CSV" ] || [ ! -f "$VAL_CSV" ] || [ ! -f "$TEST_CSV" ]; then
    echo "[Dataset] Split CSV files not found in compare/. Generating splits now..."
    cd "$SCRIPT_DIR/compare"
    python3 create_split.py || python create_split.py
    cd "$SCRIPT_DIR"
fi

if [ ! -f "$TRAIN_CSV" ]; then
    echo "ERROR: Training data not found at $TRAIN_CSV"
    exit 1
fi

# ------------------------------------------------------------------------------
# 3. Launch Background Training with nohup and disown
# ------------------------------------------------------------------------------
# Setting PYTHONUNBUFFERED=1 is CRUCIAL so that logs update in real-time
# when piped/redirected to files.
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SCRIPT_DIR/chemprop:$PYTHONPATH"

echo "=================================================================="
echo " Starting Background Chemprop Training"
echo "=================================================================="
echo " Target Property   : $PROPERTY"
echo " Epochs            : $EPOCHS"
echo " Dataset Size      : 110,000 molecules (ENTIRE training set)"
echo " Validation Size   : 10,000 molecules"
echo " Test Set Size     : 42,956 molecules"
echo " Mini-Batch Size   : $BATCH_SIZE molecules per GPU gradient step"
echo " Save Directory    : $CHECKPOINT_DIR"
echo " Log File          : $LOG_FILE"
echo " Error File        : $ERR_FILE"
echo " Latest Symlink    : $LATEST_LOG"
echo "=================================================================="

# Symlink latest log for convenience
ln -sf "$LOG_FILE" "$LATEST_LOG"

nohup $TRAIN_RUNNER \
    -i "$TRAIN_CSV" "$VAL_CSV" "$TEST_CSV" \
    --smiles-columns smile \
    --target-columns "$PROPERTY" \
    --task-type regression \
    --output-dir "$CHECKPOINT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --metrics mae mse \
    --accelerator auto \
    >> "$LOG_FILE" 2> "$ERR_FILE" &

PID=$!
echo "$PID" > "$PID_FILE"
disown "$PID"

echo ""
echo " Training successfully detached into background!"
echo " Process PID       : $PID (saved in $PID_FILE)"
echo ""
echo " Useful Commands:"
echo "   Monitor Real-Time : tail -f $LOG_FILE"
echo "   Monitor Latest    : tail -f $LATEST_LOG"
echo "   Check if Running  : ps -p $PID"
echo "   Stop Training     : kill $PID"
echo "   GPU Usage         : watch -n 1 nvidia-smi"
echo "=================================================================="
