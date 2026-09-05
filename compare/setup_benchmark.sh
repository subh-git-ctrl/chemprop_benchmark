#!/bin/bash
# Complete Chemprop Benchmark Setup Script
# Run step-by-step from chemprop-benchmark directory

set -e  # Exit on error

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$BENCHMARK_DIR/data_qm40"
CHEMPROP_DIR="$BENCHMARK_DIR/chemprop"
COMPARE_DIR="$BENCHMARK_DIR/compare"

echo "=========================================="
echo "QPred vs Chemprop Benchmark Setup"
echo "=========================================="
echo "Benchmark Directory: $BENCHMARK_DIR"
echo ""

# Step 1: Create data split matching QPred
echo "Step 1: Creating train/val/test split (seed=42)..."
cd "$COMPARE_DIR"
python create_split.py

if [ ! -f "train.csv" ] || [ ! -f "val.csv" ] || [ ! -f "test.csv" ]; then
    echo "ERROR: Split creation failed!"
    exit 1
fi

echo "✓ Split created successfully"
echo ""

# Step 2: Set up Chemprop environment
echo "Step 2: Setting up Chemprop environment..."
cd "$CHEMPROP_DIR"

if [ ! -d "venv" ]; then
    echo "  Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    
    echo "  Installing Chemprop..."
    pip install --upgrade pip
    pip install -e .
    pip install -r requirements.txt
    
    echo "✓ Chemprop environment ready"
else
    source venv/bin/activate
    echo "✓ Using existing Chemprop environment"
fi

echo ""

# Step 3: Prepare Chemprop input files
echo "Step 3: Preparing Chemprop input files..."

# Create combined CSV with split information for Chemprop
python << 'EOF'
import pandas as pd
import os

compare_dir = os.path.abspath("../compare")
chemprop_dir = os.path.abspath(".")

# Load split data
train_df = pd.read_csv(os.path.join(compare_dir, "train.csv"))
val_df = pd.read_csv(os.path.join(compare_dir, "val.csv"))
test_df = pd.read_csv(os.path.join(compare_dir, "test.csv"))

# Add split column
train_df['split'] = 'train'
val_df['split'] = 'val'
test_df['split'] = 'test'

# Combine
combined = pd.concat([train_df, val_df, test_df], ignore_index=True)

# For Chemprop, we need: SMILES, target property, split
output_file = os.path.join(compare_dir, "chemprop_data.csv")
combined[['smile', 'Polarizability', 'split']].to_csv(output_file, index=False)

print(f"Created Chemprop input file: {output_file}")
print(f"Total molecules: {len(combined)}")
print(f"  Train: {len(train_df)}")
print(f"  Val: {len(val_df)}")
print(f"  Test: {len(test_df)}")
EOF

echo "✓ Chemprop input files ready"
echo ""

# Step 4: Train Chemprop
echo "Step 4: Training Chemprop on Polarizability..."
echo ""
echo "Run this command to start training:"
echo ""
echo "python scripts/train.py \\"
echo "  --data_path ../compare/chemprop_data.csv \\"
echo "  --dataset_type regression \\"
echo "  --task_names Polarizability \\"
echo "  --save_dir checkpoints/qm40_polarizability \\"
echo "  --seed 42 \\"
echo "  --num_folds 1 \\"
echo "  --batch_size 32 \\"
echo "  --epochs 100 \\"
echo "  --lr 1e-4 \\"
echo "  --metrics mse"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Navigate to: cd chemprop-benchmark/chemprop"
echo "2. Source environment: source venv/bin/activate"
echo "3. Run training command (see above)"
echo "4. After training, generate predictions:"
echo "   python scripts/predict.py \\"
echo "     --test_path ../compare/test.csv \\"
echo "     --checkpoint_path checkpoints/qm40_polarizability/model.pt \\"
echo "     --output_path ../compare/chemprop_predictions.csv"
echo ""
echo "5. Compare with QPred using:"
echo "   cd ../compare"
echo "   python compare_predictions.py"
echo ""
