#!/usr/bin/env python3
"""
Create train/val/test split matching QPred model exactly:
- Seed: 42
- Split: 80% train, 10% val, 10% test
- Uses numpy random permutation (same as QPred)
"""

import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

def create_split(csv_path, output_dir, seed=42, sample_size=None):
    """
    Create matching train/val/test split with same seed as QPred model.
    
    Args:
        csv_path: Path to main.csv
        output_dir: Output directory for split files
        seed: Random seed (default 42 - matches QPred)
        sample_size: Optional sample size for testing
    """
    
    print(f"Creating train/val/test split from {csv_path}")
    print(f"Using seed: {seed}")
    
    # Read dataset
    df = pd.read_csv(csv_path)
    print(f"Total molecules in dataset: {len(df)}")
    
    # Get valid SMILES (matching QPred logic)
    from rdkit import Chem
    valid_indices = []
    for i, smile in enumerate(df['smile']):
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            valid_indices.append(i)
    
    print(f"Valid molecules (parseable SMILES): {len(valid_indices)}")
    
    # Shuffle with seed (exactly like QPred)
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(valid_indices)
    
    # Apply sample size if specified
    if sample_size is not None and sample_size < len(shuffled_indices):
        print(f"Sampling {sample_size} molecules from {len(shuffled_indices)}")
        shuffled_indices = shuffled_indices[:sample_size]
        num_train = int(0.8 * len(shuffled_indices))
        num_val = int(0.1 * len(shuffled_indices))
    else:
        # QPred caps at 110k train, 10k val
        num_train = min(110000, int(0.8 * len(shuffled_indices)))
        num_val = min(10000, int(0.1 * len(shuffled_indices)))
    
    # Split indices
    train_indices = shuffled_indices[:num_train]
    val_indices = shuffled_indices[num_train : num_train + num_val]
    test_indices = shuffled_indices[num_train + num_val:]
    
    print(f"Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get Zinc_ids for each split
    train_ids = df.iloc[train_indices]['Zinc_id'].tolist()
    val_ids = df.iloc[val_indices]['Zinc_id'].tolist()
    test_ids = df.iloc[test_indices]['Zinc_id'].tolist()
    
    # Save split files (format: one Zinc_id per line)
    train_path = os.path.join(output_dir, 'train_ids.txt')
    val_path = os.path.join(output_dir, 'val_ids.txt')
    test_path = os.path.join(output_dir, 'test_ids.txt')
    
    with open(train_path, 'w') as f:
        f.write('\n'.join(train_ids))
    
    with open(val_path, 'w') as f:
        f.write('\n'.join(val_ids))
    
    with open(test_path, 'w') as f:
        f.write('\n'.join(test_ids))
    
    print(f"Split files saved to {output_dir}/")
    print(f"  - train_ids.txt ({len(train_ids)} molecules)")
    print(f"  - val_ids.txt ({len(val_ids)} molecules)")
    print(f"  - test_ids.txt ({len(test_ids)} molecules)")
    
    # Also save as CSV files for easier inspection
    train_csv_path = os.path.join(output_dir, 'train.csv')
    val_csv_path = os.path.join(output_dir, 'val.csv')
    test_csv_path = os.path.join(output_dir, 'test.csv')
    
    df.iloc[train_indices].to_csv(train_csv_path, index=False)
    df.iloc[val_indices].to_csv(val_csv_path, index=False)
    df.iloc[test_indices].to_csv(test_csv_path, index=False)
    
    print(f"CSV files saved:")
    print(f"  - train.csv")
    print(f"  - val.csv")
    print(f"  - test.csv")
    
    return train_indices, val_indices, test_indices

if __name__ == '__main__':
    # Script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    csv_path = os.path.join(parent_dir, 'data_qm40', 'main.csv')
    output_dir = script_dir
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
    
    create_split(csv_path, output_dir, seed=42)
    
    print("\n✓ Split created successfully!")
    print("Next: Use the split files with Chemprop for fair comparison")
