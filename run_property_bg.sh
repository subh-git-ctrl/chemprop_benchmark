#!/bin/bash
# ==============================================================================
# Script: run_property_bg.sh
# Purpose: Alias for run_chemprop_bg.sh — kept for backward compatibility.
#          Trains Chemprop on QM40 in the background with auto environment
#          setup and a clean per-epoch progress log.
# Usage:
#   ./run_property_bg.sh <property_name> [num_epochs] [batch_size] [num_workers]
#
# Examples:
#   ./run_property_bg.sh Polarizability 100
#   ./run_property_bg.sh HOMO 500
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_chemprop_bg.sh" "$@"
