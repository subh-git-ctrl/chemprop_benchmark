#!/usr/bin/env python3
"""
Compare QPred vs Chemprop predictions on the same test set.
Calculates MAE and other metrics to quantify model performance difference.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

def load_predictions(qpred_path, chemprop_path):
    """Load predictions from both models."""
    print(f"Loading QPred predictions from: {qpred_path}")
    qpred_df = pd.read_csv(qpred_path)
    
    print(f"Loading Chemprop predictions from: {chemprop_path}")
    chemprop_df = pd.read_csv(chemprop_path)
    
    return qpred_df, chemprop_df

def align_predictions(qpred_df, chemprop_df, id_column='Zinc_id'):
    """
    Align predictions by molecule ID to ensure fair comparison.
    Both models should be evaluated on the exact same test molecules.
    """
    print(f"\nAligning predictions by {id_column}...")
    
    # Get common molecules
    qpred_ids = set(qpred_df[id_column])
    chemprop_ids = set(chemprop_df[id_column])
    
    common_ids = qpred_ids & chemprop_ids
    print(f"QPred test molecules: {len(qpred_ids)}")
    print(f"Chemprop test molecules: {len(chemprop_ids)}")
    print(f"Common molecules: {len(common_ids)}")
    
    if len(common_ids) == 0:
        print("ERROR: No common molecules found!")
        print("Ensure both models were trained/tested on the same split")
        return None, None
    
    # Filter to common molecules
    qpred_aligned = qpred_df[qpred_df[id_column].isin(common_ids)].set_index(id_column).sort_index()
    chemprop_aligned = chemprop_df[chemprop_df[id_column].isin(common_ids)].set_index(id_column).sort_index()
    
    return qpred_aligned, chemprop_aligned

def calculate_metrics(y_true, y_pred_model1, y_pred_model2, model1_name, model2_name):
    """Calculate regression metrics."""
    
    mae_m1 = np.mean(np.abs(y_true - y_pred_model1))
    mae_m2 = np.mean(np.abs(y_true - y_pred_model2))
    
    rmse_m1 = np.sqrt(np.mean((y_true - y_pred_model1) ** 2))
    rmse_m2 = np.sqrt(np.mean((y_true - y_pred_model2) ** 2))
    
    # Pearson correlation
    corr_m1 = np.corrcoef(y_true, y_pred_model1)[0, 1]
    corr_m2 = np.corrcoef(y_true, y_pred_model2)[0, 1]
    
    # R² score
    ss_res_m1 = np.sum((y_true - y_pred_model1) ** 2)
    ss_res_m2 = np.sum((y_true - y_pred_model2) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_m1 = 1 - (ss_res_m1 / ss_tot)
    r2_m2 = 1 - (ss_res_m2 / ss_tot)
    
    return {
        model1_name: {
            'MAE': mae_m1,
            'RMSE': rmse_m1,
            'Correlation': corr_m1,
            'R²': r2_m1
        },
        model2_name: {
            'MAE': mae_m2,
            'RMSE': rmse_m2,
            'Correlation': corr_m2,
            'R²': r2_m2
        }
    }

def print_comparison_table(metrics):
    """Print comparison table."""
    print("\n" + "="*80)
    print("PREDICTION COMPARISON")
    print("="*80)
    
    models = list(metrics.keys())
    metrics_types = list(metrics[models[0]].keys())
    
    print(f"\n{'Metric':<15} {models[0]:<20} {models[1]:<20} {'Difference':<15}")
    print("-" * 70)
    
    for metric in metrics_types:
        val1 = metrics[models[0]][metric]
        val2 = metrics[models[1]][metric]
        diff = val2 - val1  # Negative means model1 is better for MAE/RMSE
        
        symbol = "✓" if (metric in ['Correlation', 'R²'] and val1 > val2) or \
                       (metric in ['MAE', 'RMSE'] and val1 < val2) else " "
        
        print(f"{metric:<15} {val1:<20.6f} {val2:<20.6f} {diff:>14.6f} {symbol}")

def generate_report(qpred_path, chemprop_path, output_dir=None, target_property='Polarizability', 
                   id_column='Zinc_id', y_true_column='actual', 
                   qpred_pred_col='predicted', chemprop_pred_col='prediction'):
    """
    Generate comprehensive comparison report.
    
    Args:
        qpred_path: Path to QPred predictions CSV
        chemprop_path: Path to Chemprop predictions CSV
        output_dir: Optional output directory for report
        target_property: Property being predicted
        id_column: Column name with molecule IDs
        y_true_column: Column name with ground truth values
        qpred_pred_col: Column name for QPred predictions
        chemprop_pred_col: Column name for Chemprop predictions
    """
    
    # Load predictions
    qpred_df, chemprop_df = load_predictions(qpred_path, chemprop_path)
    
    # Align by molecule ID
    qpred_aligned, chemprop_aligned = align_predictions(qpred_df, chemprop_df, id_column)
    
    if qpred_aligned is None or chemprop_aligned is None:
        sys.exit(1)
    
    # Extract values
    y_true = qpred_aligned[y_true_column].values
    y_pred_qpred = qpred_aligned[qpred_pred_col].values
    y_pred_chemprop = chemprop_aligned[chemprop_pred_col].values
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred_qpred, y_pred_chemprop, 'QPred', 'Chemprop')
    
    # Print results
    print_comparison_table(metrics)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    mae_diff = metrics['Chemprop']['MAE'] - metrics['QPred']['MAE']
    improvement = (metrics['QPred']['MAE'] - metrics['Chemprop']['MAE']) / metrics['Chemprop']['MAE'] * 100
    
    print(f"Target Property: {target_property}")
    print(f"Test Set Size: {len(y_true)} molecules")
    print(f"\nBest MAE: {'QPred' if metrics['QPred']['MAE'] < metrics['Chemprop']['MAE'] else 'Chemprop'}")
    print(f"MAE Difference: {abs(mae_diff):.6f} {'(QPred better)' if mae_diff > 0 else '(Chemprop better)'}")
    print(f"Percentage Difference: {abs(improvement):.2f}%")
    
    # Save report if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, 'comparison_report.txt')
        with open(report_path, 'w') as f:
            f.write("QPred vs Chemprop Benchmark Report\n")
            f.write("="*80 + "\n")
            f.write(f"Target Property: {target_property}\n")
            f.write(f"Test Set Size: {len(y_true)}\n")
            f.write(f"Comparison Date: {pd.Timestamp.now()}\n\n")
            
            f.write("Metrics Comparison:\n")
            f.write("-"*70 + "\n")
            for metric in metrics['QPred'].keys():
                qpred_val = metrics['QPred'][metric]
                chemprop_val = metrics['Chemprop'][metric]
                f.write(f"{metric:<20} QPred: {qpred_val:.6f}  Chemprop: {chemprop_val:.6f}\n")
            
            f.write(f"\nBetter Model (MAE): {'QPred' if metrics['QPred']['MAE'] < metrics['Chemprop']['MAE'] else 'Chemprop'}\n")
            f.write(f"Improvement: {abs(improvement):.2f}%\n")
        
        print(f"\nReport saved to: {report_path}")
    
    return metrics

if __name__ == '__main__':
    # Default paths (adjust based on your file locations)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    qpred_pred_path = os.path.join(script_dir, 'qpred_predictions.csv')
    chemprop_pred_path = os.path.join(script_dir, 'chemprop_predictions.csv')
    
    # Check if files exist
    if not os.path.exists(qpred_pred_path):
        print(f"ERROR: QPred predictions file not found: {qpred_pred_path}")
        print("Please provide QPred test predictions in CSV format")
        sys.exit(1)
    
    if not os.path.exists(chemprop_pred_path):
        print(f"ERROR: Chemprop predictions file not found: {chemprop_pred_path}")
        print("Please provide Chemprop test predictions in CSV format")
        sys.exit(1)
    
    # Generate report
    metrics = generate_report(
        qpred_pred_path,
        chemprop_pred_path,
        output_dir=script_dir,
        target_property='Polarizability'
    )
    
    print("\n✓ Comparison complete!")
