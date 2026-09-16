#!/bin/bash
# ==============================================================================
# Script: stop_chemprop.sh
# Purpose: Safely stop a running background chemprop training launched with
#          run_chemprop_bg.sh (kills both the trainer and the progress monitor).
#
# Usage:
#   ./stop_chemprop.sh <property_name>
#
# Example:
#   ./stop_chemprop.sh Polarizability
#   ./stop_chemprop.sh "Internal_E(0K)"
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <property_name>"
    echo ""
    echo "Currently tracked runs (logs/*.pid):"
    ls -1 "$SCRIPT_DIR/logs"/*.pid 2>/dev/null | sed "s|.*/||; s|\.pid$||; s/^/  /" || echo "  (none)"
    exit 1
fi

PROPERTY="$1"
SAFE_NAME=$(echo "$PROPERTY" | tr ' ()' '___')

PID_FILE="$SCRIPT_DIR/logs/${SAFE_NAME}.pid"
MON_PID_FILE="$SCRIPT_DIR/logs/${SAFE_NAME}_monitor.pid"
PROGRESS_LOG="$SCRIPT_DIR/logs/${SAFE_NAME}_progress.log"

stopped=0

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo "[Stop] Sending SIGTERM to trainer PID $PID ..."
        kill "$PID" 2>/dev/null
        # Wait up to 20s for a clean shutdown (Lightning saves last checkpoint)
        for _ in $(seq 1 20); do
            kill -0 "$PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$PID" 2>/dev/null; then
            echo "[Stop] Still alive after 20s — sending SIGKILL."
            kill -9 "$PID" 2>/dev/null
        fi
        echo "[Stop] Trainer stopped."
        stopped=1
    else
        echo "[Stop] Trainer (PID $PID) is not running."
    fi
else
    echo "[Stop] No PID file for '$PROPERTY' ($PID_FILE)."
fi

if [ -f "$MON_PID_FILE" ]; then
    MON_PID=$(cat "$MON_PID_FILE")
    if [ -n "$MON_PID" ] && kill -0 "$MON_PID" 2>/dev/null; then
        kill "$MON_PID" 2>/dev/null
        echo "[Stop] Monitor (PID $MON_PID) stopped."
        stopped=1
    fi
fi

if [ "$stopped" -eq 0 ]; then
    echo "[Stop] Nothing to stop. (Check: pgrep -fl 'chemprop.*train')"
    exit 1
fi

echo "[Stop] Done. Partial artifacts remain in checkpoints/ (best checkpoint is kept)."
echo "       Final progress summary: tail -n 20 $PROGRESS_LOG"
