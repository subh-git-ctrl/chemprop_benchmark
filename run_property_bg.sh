#!/bin/bash
# ==============================================================================
# Script: run_property_bg.sh
# Purpose: Train Chemprop in background on the ENTIRE QM40 dataset (~163,000 molecules)
# Usage:
#   ./run_property_bg.sh <property_name> [num_epochs] [batch_size]
#
# Examples:
#   ./run_property_bg.sh Polarizability 100
#   ./run_property_bg.sh HOMO 100
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_chemprop_bg.sh" "$@"
