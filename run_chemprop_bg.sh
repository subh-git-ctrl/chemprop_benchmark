#!/bin/bash
# ==============================================================================
# Script: run_chemprop_bg.sh
# Purpose: One-command background training of Chemprop on the QM40 dataset.
#
#   ./run_chemprop_bg.sh <property_name> [num_epochs] [batch_size] [num_workers]
#
# What it does (fully automatic):
#   1. Builds/uses a DEDICATED Python environment for chemprop — it never
#      touches your `qpred` environment (this was the root cause of the old
#      "not running in the qpred env" failures: qpred has a different Python).
#         - $CHEMPROP_ENV_DIR set (absolute path)   -> use/create the env THERE
#         - elif $HOME/.conda/envs/chemprop exists  -> use it
#         - elif a conda env named `chemprop` exists -> use it, but ONLY if it
#            physically lives under $HOME (an env on any other disk is ignored)
#         - elif conda is available                 -> create the env at the
#            EXPLICIT prefix $HOME/.conda/envs/chemprop (py3.11), so conda's
#            envs_dirs can never redirect it to another disk
#         - else                                    -> create `.venv-chemprop`
#            from the newest python3.10+ found on the system
#      The env therefore lives on the home filesystem unless YOU point
#      CHEMPROP_ENV_DIR at a path on a disk of your choice. This script never
#      mounts anything, never names a device and never requires a data disk.
#   2. Auto-installs anything missing inside that env only:
#         torch (CUDA build on Linux via PyPI wheels) + `pip install -e ./chemprop`
#   3. Verifies/generates the train/val/test split files in compare/.
#   4. Launches chemprop training in the background (nohup + disown), so it
#      survives SSH disconnects, writing a raw log.
#   5. Launches monitor_progress.py in the background, which maintains a CLEAN
#      progress log with one line per finished epoch:
#         Epoch 34/500 | val_MAE=0.0342 (physical units) | val_MSE=... | ...
#      plus a final summary incl. TEST-set MAE/MSE.
#
# Logs (in ./logs/):
#   <prop>_<timestamp>.log    raw chemprop output (stdout+stderr)
#   <prop>_latest.log         symlink to the raw log above
#   <prop>_progress.log       CLEAN per-epoch progress  <-- watch this one
#   <prop>.pid / <prop>_monitor.pid   PIDs of trainer / monitor
#
# Examples:
#   ./run_chemprop_bg.sh Polarizability 100
#   ./run_chemprop_bg.sh HOMO 500
#   ./run_chemprop_bg.sh "spatial extent" 100 64 4
#   ./stop_chemprop.sh Polarizability        # to stop a running training
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------------------------------------------------------
# 0. Arguments
# ------------------------------------------------------------------------------
if [ $# -eq 0 ]; then
    echo "=================================================================="
    echo " Chemprop VM Background Training Script (one command, auto env)"
    echo "=================================================================="
    echo "Usage:"
    echo "  $0 <property_name> [num_epochs=100] [batch_size=64] [num_workers=2]"
    echo ""
    echo "Available QM40 Properties (must match the CSV column names):"
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
    echo " 11) Free_E            (Hartree)"
    echo "  12) CV                (cal/mol·K)"
    echo "  13) Entropy           (cal/mol·K)"
    echo ""
    echo "Examples:"
    echo "  $0 Polarizability 100"
    echo "  $0 Polarizability 500 64 4"
    echo "  $0 \"Internal_E(0K)\" 100"
    echo ""
    echo "Optional environment variables:"
    echo "  CHEMPROP_ENV_DIR    absolute path where the env lives / gets created"
    echo "                      (default: \$HOME/.conda/envs/\$CHEMPROP_ENV_NAME —"
    echo "                       point it at a path on any disk YOU mounted,"
    echo "                       wherever you like; the script itself never"
    echo "                       mounts anything and never touches other disks)"
    echo "  CHEMPROP_ENV_NAME   env name for the default home location (default: chemprop)"
    echo "  CHEMPROP_PYTHON     python executable for venv     (e.g. /usr/bin/python3.11)"
    echo "  TORCH_INDEX_URL     pip index for torch            (e.g. https://download.pytorch.org/whl/cu121)"
    echo "=================================================================="
    exit 1
fi

PROPERTY="$1"
EPOCHS="${2:-100}"
BATCH_SIZE="${3:-64}"
NUM_WORKERS="${4:-2}"
ENV_NAME="${CHEMPROP_ENV_NAME:-chemprop}"

SAFE_NAME=$(echo "$PROPERTY" | tr ' ()' '___')

LOG_DIR="$SCRIPT_DIR/logs"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints/${SAFE_NAME}"
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/${SAFE_NAME}_${TIMESTAMP}.log"
PROGRESS_LOG="$LOG_DIR/${SAFE_NAME}_progress.log"
PID_FILE="$LOG_DIR/${SAFE_NAME}.pid"
MON_PID_FILE="$LOG_DIR/${SAFE_NAME}_monitor.pid"
LATEST_LOG="$LOG_DIR/${SAFE_NAME}_latest.log"

# ------------------------------------------------------------------------------
# 1. Guard: is this property already training?
# ------------------------------------------------------------------------------
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null)
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "=================================================================="
        echo " ERROR: A training run for '$PROPERTY' is already running (PID $OLD_PID)."
        echo " Stop it first with:  ./stop_chemprop.sh $PROPERTY"
        echo " Or watch progress :  tail -f $PROGRESS_LOG"
        echo "=================================================================="
        exit 1
    fi
fi
# Kill any stale monitor from a previous run of this property
if [ -f "$MON_PID_FILE" ]; then
    OLD_MON=$(cat "$MON_PID_FILE" 2>/dev/null)
    [ -n "$OLD_MON" ] && kill "$OLD_MON" 2>/dev/null
fi

echo "=================================================================="
echo " [1/4] Setting up the dedicated chemprop environment"
echo "=================================================================="

# ------------------------------------------------------------------------------
# 1b. Resolve WHERE the chemprop env will live + preflight disk-space guard
#     Policy: the env lives on the HOME filesystem by default, created at an
#     EXPLICIT prefix so conda's envs_dirs can never redirect it elsewhere.
#     This script NEVER mounts anything, never names a device and never
#     requires a separate data disk. To place the env on a disk of YOUR
#     choice, export CHEMPROP_ENV_DIR=/path/on/that/disk.
# ------------------------------------------------------------------------------
fs_avail_bytes()  { df -B1 --output=avail "$1" 2>/dev/null | tail -n 1 | tr -dc '0-9'; }
human_gb()        { echo $(( ($1 + 1024*1024*1024 - 1) / (1024*1024*1024) )); }
MIN_ENV_FREE_BYTES=$(( 10 * 1024 * 1024 * 1024 ))    # torch build + chemprop deps head-room
MIN_CACHE_FREE_BYTES=$(( 4 * 1024 * 1024 * 1024 ))   # pip wheel cache during install

dir_has_env() { [ -x "$1/bin/python" ] || [ -d "$1/conda-meta" ]; }

# --- locate conda (if any) ---
CONDA_SH=""
for c in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge"; do
    [ -f "$c/etc/profile.d/conda.sh" ] && CONDA_SH="$c/etc/profile.d/conda.sh" && break
done
if [ -z "$CONDA_SH" ] && command -v conda &>/dev/null; then
    CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
    [ -f "$CONDA_SH" ] || CONDA_SH=""
fi

# --- detect an existing reusable repo venv ---
VENV_DIR="$SCRIPT_DIR/.venv-chemprop"
VENV_EXISTS=0
for v in "$VENV_DIR" "$SCRIPT_DIR/venv" "$SCRIPT_DIR/chemprop/venv"; do
    if [ -x "$v/bin/python" ]; then
        if "$v/bin/python" -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
            VENV_DIR="$v"; VENV_EXISTS=1; break
        fi
    fi
done

# --- resolve the env prefix ---
if [ -n "${CHEMPROP_ENV_DIR:-}" ]; then
    case "$CHEMPROP_ENV_DIR" in
        /*) ENV_PREFIX="$CHEMPROP_ENV_DIR" ;;
        *)  ENV_PREFIX="$PWD/$CHEMPROP_ENV_DIR" ;;
    esac
    mkdir -p "$(dirname "$ENV_PREFIX")" 2>/dev/null
    ENV_PREFIX="$(cd "$(dirname "$ENV_PREFIX")" 2>/dev/null && pwd)/$(basename "$ENV_PREFIX")"
    echo "[Disk] CHEMPROP_ENV_DIR override active — env prefix: $ENV_PREFIX"
else
    ENV_PREFIX="$HOME/.conda/envs/$ENV_NAME"
fi

# --- a NAMED conda env is reused ONLY if it physically lives under $HOME.
#     An env sitting on some other disk is ignored: this script never uses a
#     disk you did not explicitly choose via CHEMPROP_ENV_DIR. ---
NAMED_ENV_PATH=""
if [ -n "$CONDA_SH" ] && [ "$(basename "$ENV_PREFIX")" = "$ENV_NAME" ]; then
    NAMED_ENV_PATH="$(conda env list 2>/dev/null | awk -v n="$ENV_NAME" '$1 == n && NF >= 2 {print $NF}')"
    [ "$NAMED_ENV_PATH" = "$ENV_PREFIX" ] && NAMED_ENV_PATH=""   # same dir the prefix check already covers
fi
NAMED_ENV_REUSABLE=0
case "$NAMED_ENV_PATH" in
    "$HOME"/*) NAMED_ENV_REUSABLE=1 ;;
    "")
        ;;
    *)
        echo "[Env] Note: a conda env '$ENV_NAME' exists at $NAMED_ENV_PATH (outside \$HOME)."
        echo "[Env]       Ignoring it — this script keeps the env under \$HOME unless you"
        echo "[Env]       explicitly choose that path:  CHEMPROP_ENV_DIR=$NAMED_ENV_PATH"
        NAMED_ENV_PATH=""
        ;;
esac

# --- effective creation target: prefix when conda (or an explicit override),
#     repo venv otherwise ---
if [ -n "$CONDA_SH" ] || [ -n "${CHEMPROP_ENV_DIR:-}" ]; then
    ENV_TARGET="$ENV_PREFIX"
else
    ENV_TARGET="$SCRIPT_DIR/.venv-chemprop"
fi

# --- space guard: runs ONLY if a fresh env would actually be created ---
if ! dir_has_env "$ENV_PREFIX" && [ "$VENV_EXISTS" -eq 0 ] && [ "$NAMED_ENV_REUSABLE" -eq 0 ]; then
    TARGET_AVAIL=$(fs_avail_bytes "$ENV_TARGET")
    if [ -n "$TARGET_AVAIL" ] && [ "$TARGET_AVAIL" -lt "$MIN_ENV_FREE_BYTES" ]; then
        echo "=================================================================="
        echo " ERROR: only $(human_gb "$TARGET_AVAIL") GiB free on the filesystem that would"
        echo " hold the chemprop environment ($ENV_TARGET)."
        echo ""
        echo " The PyTorch build + chemprop dependencies need ~10 GiB while"
        echo " installing. Free up space there first (e.g. 'pip cache purge',"
        echo " 'conda clean --all', old checkpoints), or point the env at any"
        echo " path with room — on a disk YOU mounted wherever you like:"
        echo ""
        echo "   export CHEMPROP_ENV_DIR=/your/mounted/path/envs/chemprop"
        echo ""
        echo " Aborting BEFORE downloading ~3 GB of torch (which would end in"
        echo " '[Errno 28] No space left on device'). Nothing was downloaded."
        echo "=================================================================="
        exit 1
    fi
fi

# ------------------------------------------------------------------------------
# 2. Environment bootstrap — NEVER touches qpred; env stays under $HOME
#    unless CHEMPROP_ENV_DIR says otherwise
# ------------------------------------------------------------------------------
ACTIVATED=0

if dir_has_env "$ENV_PREFIX"; then
    # --- reuse the env at the resolved prefix (default home path or override) ---
    if [ -n "$CONDA_SH" ]; then
        source "$CONDA_SH"
        conda activate "$ENV_PREFIX"
    else
        # no conda on this box: the prefix env can only be a venv-style env
        # shellcheck disable=SC1091
        source "$ENV_PREFIX/bin/activate" 2>/dev/null || export PATH="$ENV_PREFIX/bin:$PATH"
    fi
    echo "[Env] Using existing environment: $ENV_PREFIX"
    ACTIVATED=1
elif [ "$NAMED_ENV_REUSABLE" -eq 1 ]; then
    # --- reuse an existing named conda env that lives under $HOME ---
    source "$CONDA_SH"
    conda activate "$ENV_NAME"
    echo "[Env] Using existing conda environment: $ENV_NAME ($NAMED_ENV_PATH)"
    ACTIVATED=1
elif [ "$VENV_EXISTS" -eq 1 ]; then
    # --- reuse existing venv ---
    source "$VENV_DIR/bin/activate"
    echo "[Env] Using existing virtual environment: $VENV_DIR"
    ACTIVATED=1
elif [ -n "$CONDA_SH" ]; then
    # --- build a fresh dedicated conda env at an EXPLICIT prefix (py3.11).
    #     Using -p instead of -n: creation lands exactly here no matter how
    #     conda's envs_dirs is configured, so the env can never silently end
    #     up on some other disk. ---
    echo "[Env] Creating dedicated conda environment at $ENV_PREFIX (python 3.11) ..."
    source "$CONDA_SH"
    if conda create -y -p "$ENV_PREFIX" python=3.11; then
        conda activate "$ENV_PREFIX"
        echo "[Env] Created and activated conda environment: $ENV_PREFIX"
        ACTIVATED=1
    else
        echo "[Env] WARNING: conda env creation failed; falling back to venv."
    fi
fi

if [ "$ACTIVATED" -eq 0 ]; then
    # --- build a fresh venv from the newest python3.10+ available ---
    PY_CANDIDATES=("${CHEMPROP_PYTHON:-}" python3.13 python3.12 python3.11 python3.10 python3 python)
    PY_BASE=""
    for cand in "${PY_CANDIDATES[@]}"; do
        [ -z "$cand" ] && continue
        if command -v "$cand" &>/dev/null; then
            # require py>=3.10 AND a working venv module (ensurepip) — the
            # newest interpreter sometimes ships without it (Ubuntu: install
            # python3.XX-venv to fix)
            if "$cand" -c "import sys, ensurepip; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
                PY_BASE="$cand"; break
            fi
        fi
    done
    if [ -z "$PY_BASE" ]; then
        echo "=================================================================="
        echo " ERROR: no Python >= 3.10 found on this system."
        echo " Install python3.11 (e.g. 'sudo apt install python3.11 python3.11-venv')"
        echo " or set CHEMPROP_PYTHON=/path/to/python3.11 before running."
        echo "=================================================================="
        exit 1
    fi
    if [ -n "${CHEMPROP_ENV_DIR:-}" ]; then VENV_TARGET="$ENV_PREFIX"; else VENV_TARGET="$VENV_DIR"; fi
    echo "[Env] Creating venv at $VENV_TARGET using $($PY_BASE --version 2>&1) ..."
    "$PY_BASE" -m venv "$VENV_TARGET" || { echo "ERROR: venv creation failed (is python3-venv installed?)"; exit 1; }
    source "$VENV_TARGET/bin/activate"
    echo "[Env] Created and activated virtual environment: $VENV_TARGET"
fi

PY_EXE="$(command -v python)"
echo "[Env] Python in use: $PY_EXE ($($PY_EXE --version 2>&1))"

# Import checks MUST ignore the current directory: this repo contains a
# `chemprop/` checkout folder, and with the repo root as cwd Python would
# resolve `import chemprop` to that folder as a namespace package (it has no
# chemprop.cli inside), shadowing the real pip-installed package and causing
# "ModuleNotFoundError: No module named 'chemprop.cli'".
PY_FILTER='import sys; sys.path = [p for p in sys.path if p not in ("", ".")]'

# Final safety gate: the env must be >= 3.10 for chemprop 2.x
if ! "$PY_EXE" -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
    echo "=================================================================="
    echo " ERROR: the active environment has Python < 3.10 — chemprop 2.x"
    echo " requires >= 3.10. This script never uses the qpred environment;"
    echo " delete the broken '$ENV_NAME' env / .venv-chemprop and re-run."
    echo "=================================================================="
    exit 1
fi

# ------------------------------------------------------------------------------
# 3. Auto-install missing dependencies (inside the dedicated env ONLY)
# ------------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo " [2/4] Checking dependencies (installs only what is missing)"
echo "=================================================================="

if ! "$PY_EXE" -c "$PY_FILTER; import torch" 2>/dev/null; then
    # torch wheels: ~3 GB download + ~6 GB installed. Verify head-room on
    # BOTH the env filesystem and the pip-cache filesystem BEFORE downloading,
    # or pip dies mid-install with "[Errno 28] No space left on device".
    ENV_FS_AVAIL=$(fs_avail_bytes "$(dirname "$PY_EXE")")
    if [ -n "$ENV_FS_AVAIL" ] && [ "$ENV_FS_AVAIL" -lt "$MIN_ENV_FREE_BYTES" ]; then
        echo "=================================================================="
        echo " ERROR: only $(human_gb "$ENV_FS_AVAIL") GiB free on the filesystem holding the"
        echo " chemprop env ($(dirname "$PY_EXE")). The torch build needs ~10 GiB"
        echo " head-room. Free up space there, or re-run with the env on another"
        echo " path of your choice:"
        echo "   export CHEMPROP_ENV_DIR=/your/mounted/path/envs/chemprop"
        echo " Nothing was downloaded yet."
        echo "=================================================================="
        exit 1
    fi
    PIP_CACHE_EFF="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
    CACHE_PROBE="$PIP_CACHE_EFF"
    while [ ! -d "$CACHE_PROBE" ] && [ "$CACHE_PROBE" != "/" ]; do
        CACHE_PROBE=$(dirname "$CACHE_PROBE")
    done
    CACHE_FS_AVAIL=$(fs_avail_bytes "$CACHE_PROBE")
    if [ -n "$CACHE_FS_AVAIL" ] && [ "$CACHE_FS_AVAIL" -lt "$MIN_CACHE_FREE_BYTES" ]; then
        echo "=================================================================="
        echo " ERROR: only $(human_gb "$CACHE_FS_AVAIL") GiB free on the pip-cache filesystem"
        echo " ($PIP_CACHE_EFF). The torch wheels are a ~3 GB download. Free up space"
        echo " there, or point the cache at any path with room first, e.g.:"
        echo "   export PIP_CACHE_DIR='$CHEMPROP_ENV_DIR/pip-cache'"
        echo " then re-run. Nothing was downloaded yet."
        echo "=================================================================="
        exit 1
    fi
    echo "[Setup] Installing PyTorch (Linux PyPI wheels bundle CUDA support)..."
    if [ -n "$TORCH_INDEX_URL" ]; then
        "$PY_EXE" -m pip install torch --index-url "$TORCH_INDEX_URL" || { echo "ERROR: torch install failed"; exit 1; }
    else
        "$PY_EXE" -m pip install torch || { echo "ERROR: torch install failed"; exit 1; }
    fi
else
    echo "[Setup] torch: already installed."
fi

if ! "$PY_EXE" -c "$PY_FILTER; import chemprop" 2>/dev/null; then
    echo "[Setup] Installing chemprop from ./chemprop (editable)..."
    "$PY_EXE" -m pip install -e "$SCRIPT_DIR/chemprop" || { echo "ERROR: chemprop install failed"; exit 1; }
else
    echo "[Setup] chemprop: already installed."
fi

# Reclaim the wheel cache (~2-3 GB) once everything installed cleanly — keeps
# the env footprint small on whichever filesystem it lives.
"$PY_EXE" -m pip cache purge >/dev/null 2>&1 || true

"$PY_EXE" -c "$PY_FILTER; import chemprop, torch; print('[Setup] chemprop', getattr(chemprop, '__version__', '?'), '| torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

# ------------------------------------------------------------------------------
# 4. Resolve the training command
# ------------------------------------------------------------------------------
# NOTE: `python3 -m chemprop.cli train` is NOT a valid invocation (chemprop has
# no cli/__main__.py), so we prefer the console script installed by pip, with a
# safe python -c wrapper as fallback.
if command -v chemprop &>/dev/null; then
    TRAIN_CMD=(chemprop train)
else
    TRAIN_CMD=("$PY_EXE" -c "$PY_FILTER; from chemprop.cli.main import main; sys.argv = ['chemprop'] + sys.argv[1:]; main()" train)
fi

# ------------------------------------------------------------------------------
# 5. Dataset verification
# ------------------------------------------------------------------------------
TRAIN_CSV="$SCRIPT_DIR/compare/train.csv"
VAL_CSV="$SCRIPT_DIR/compare/val.csv"
TEST_CSV="$SCRIPT_DIR/compare/test.csv"

# chemprop defaults to --warmup-epochs 2 and REQUIRES epochs > warmup epochs,
# so small test runs (1-2 epochs) would crash. Auto-relax warmup in that case.
WARMUP_ARGS=()
if [ "$EPOCHS" != "-1" ] && [ "$EPOCHS" -le 2 ] 2>/dev/null; then
    WARMUP_ARGS=(--warmup-epochs 0)
    echo "[Note] epochs <= 2 requested -> passing --warmup-epochs 0"
fi

if [ ! -f "$TRAIN_CSV" ] || [ ! -f "$VAL_CSV" ] || [ ! -f "$TEST_CSV" ]; then
    echo "[Dataset] Split CSVs not found in compare/ — generating (seed 42, QPred-matching)..."
    ( cd "$SCRIPT_DIR/compare" && "$PY_EXE" create_split.py ) || { echo "ERROR: split generation failed"; exit 1; }
fi

for f in "$TRAIN_CSV" "$VAL_CSV" "$TEST_CSV"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
echo "[Dataset] train/val/test CSVs found."

# ------------------------------------------------------------------------------
# 6. Launch background training + progress monitor
# ------------------------------------------------------------------------------
export PYTHONUNBUFFERED=1   # real-time logs even when redirected to files
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo ""
echo "=================================================================="
echo " [3/4] Launching background training"
echo "=================================================================="
echo " Target Property   : $PROPERTY"
echo " Epochs            : $EPOCHS"
echo " Mini-Batch Size   : $BATCH_SIZE"
echo " Data-loader workers: $NUM_WORKERS"
echo " Save Directory    : $CHECKPOINT_DIR"
echo " Raw Log           : $LOG_FILE"
echo " Progress Log      : $PROGRESS_LOG"
echo "=================================================================="

# Fresh raw log + fresh progress log for this run
: > "$LOG_FILE"
: > "$PROGRESS_LOG"
ln -sf "$LOG_FILE" "$LATEST_LOG"

nohup "${TRAIN_CMD[@]}" \
    -i "$TRAIN_CSV" "$VAL_CSV" "$TEST_CSV" \
    --smiles-columns smile \
    --target-columns "$PROPERTY" \
    --task-type regression \
    --output-dir "$CHECKPOINT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --metrics mae mse \
    --accelerator auto \
    "${WARMUP_ARGS[@]}" \
    >> "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"
disown "$PID"

# Give the trainer a second to fail fast on bad args, then start the monitor
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
    echo "=================================================================="
    echo " ERROR: chemprop exited immediately — last log lines:"
    tail -n 30 "$LOG_FILE"
    echo "=================================================================="
    exit 1
fi

echo " [4/4] Starting progress monitor (writes one clean line per epoch)..."
nohup "$PY_EXE" "$SCRIPT_DIR/monitor_progress.py" \
    --property "$PROPERTY" \
    --epochs "$EPOCHS" \
    --pid "$PID" \
    --run-log "$LOG_FILE" \
    --out "$PROGRESS_LOG" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --poll 20 \
    >> "$LOG_DIR/${SAFE_NAME}_monitor.err" 2>&1 &

MON_PID=$!
echo "$MON_PID" > "$MON_PID_FILE"
disown "$MON_PID"

echo ""
echo "=================================================================="
echo " Training successfully detached into background!"
echo "=================================================================="
echo " Trainer  PID : $PID   (saved in $PID_FILE)"
echo " Monitor  PID : $MON_PID (saved in $MON_PID_FILE)"
echo ""
echo " Useful commands:"
echo "   Watch clean progress : tail -f $PROGRESS_LOG"
echo "   Watch raw output     : tail -f $LATEST_LOG"
echo "   Check if running     : ps -p $PID"
echo "   Stop training        : ./stop_chemprop.sh $PROPERTY"
echo "   GPU usage            : watch -n 1 nvidia-smi"
echo ""
echo " After training finishes:"
echo "   - Final TEST metrics are appended to $PROGRESS_LOG"
echo "   - Test predictions  : $CHECKPOINT_DIR/model_0/test_predictions.csv"
echo "   - QPred-style MAE   : see 'Post-training evaluation' in commands.md"
echo "=================================================================="
