#!/usr/bin/env python3

import torch
import numpy as np
import time
import random
from typing import Dict, List, Optional

# Set seeds for reproducible results
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def run_mnist_experiment() -> Dict[str, float]:
    """Run MNIST experiment and extract real metrics"""
    print("Running MNIST experiment...")
    start_time = time.time()
    
    try:
        # Import and run MNIST training
        from mnist_ltn_gan import train_mnist_comparison
        
        # Actually run the MNIST training
        result = train_mnist_comparison(num_epochs=100)
        
        # Extract real metrics from training results
        metrics = {
            'baseline_g_loss': result['final_baseline_g_loss'],
            'baseline_d_loss': result['final_baseline_d_loss'],
            'baseline_quality': result['final_baseline_quality'],
            'ltn_g_loss': result['final_ltn_g_loss'],
            'ltn_g_adv_loss': result.get('final_ltn_g_adv_loss', result['final_ltn_g_loss']),  # Use total if adv not available
            'ltn_d_loss': result['final_ltn_d_loss'],
            'ltn_quality': result['final_ltn_quality'],
            'ltn_logic_satisfaction': result['final_ltn_logic_satisfaction']
        }
        
        print(f"MNIST experiment completed in {time.time() - start_time:.2f}s")
        return metrics
    except Exception as e:
        print(f"ERROR: MNIST experiment - {e}")
        return {}

def run_gaussian_experiment() -> Dict[str, float]:
    """Run Gaussian experiment and extract real metrics"""
    print("Running Gaussian experiment...")
    start_time = time.time()
    
    try:
        # Import and run Gaussian training
        from gaussian_ltn_gan import main as gaussian_main
        
        # Actually run the Gaussian training
        result = gaussian_main()
        
        # Extract real metrics from training results
        metrics = {
            'baseline_g_loss': result['final_baseline_g_loss'],
            'baseline_d_loss': result['final_baseline_d_loss'],
            'baseline_quality': result['final_baseline_quality'],
            'ltn_g_loss': result['final_ltn_g_loss'],
            'ltn_g_adv_loss': result.get('final_ltn_g_adv_loss', result['final_ltn_g_loss']),  # Use total if adv not available
            'ltn_d_loss': result['final_ltn_d_loss'],
            'ltn_quality': result['final_ltn_quality'],
            'ltn_logic_satisfaction': result['final_ltn_logic_satisfaction']
        }
        
        print(f"Gaussian experiment completed in {time.time() - start_time:.2f}s")
        return metrics
    except Exception as e:
        print(f"ERROR: Gaussian experiment - {e}")
        return {}

def run_grid_experiment() -> Dict[str, float]:
    """Run Grid experiment and extract real metrics"""
    print("Running Grid experiment...")
    start_time = time.time()
    
    try:
        # Import and run Grid training
        from grid_ltn_gan import main as grid_main
        
        # Actually run the Grid training
        result = grid_main()
        
        # Extract real metrics from training results
        metrics = {
            'baseline_g_loss': result['final_baseline_g_loss'],
            'baseline_d_loss': result['final_baseline_d_loss'],
            'baseline_quality': result['final_baseline_quality'],
            'ltn_g_loss': result['final_ltn_g_loss'],
            'ltn_g_adv_loss': result.get('final_ltn_g_adv_loss', result['final_ltn_g_loss']),  # Use total if adv not available
            'ltn_d_loss': result['final_ltn_d_loss'],
            'ltn_quality': result['final_ltn_quality'],
            'ltn_logic_satisfaction': result['final_ltn_logic_satisfaction']
        }
        
        print(f"Grid experiment completed in {time.time() - start_time:.2f}s")
        return metrics
    except Exception as e:
        print(f"ERROR: Grid experiment - {e}")
        return {}

def run_ring_experiment() -> Dict[str, float]:
    """Run Ring experiment and extract real metrics"""
    print("Running Ring experiment...")
    start_time = time.time()
    
    try:
        # Import and run Ring training
        from ring_ltn_gan import main as ring_main
        
        # Actually run the Ring training
        result = ring_main()
        
        # Extract real metrics from training results
        metrics = {
            'baseline_g_loss': result['final_baseline_g_loss'],
            'baseline_d_loss': result['final_baseline_d_loss'],
            'baseline_quality': result['final_baseline_quality'],
            'ltn_g_loss': result['final_ltn_g_loss'],
            'ltn_g_adv_loss': result.get('final_ltn_g_adv_loss', result['final_ltn_g_loss']),  # Use total if adv not available
            'ltn_d_loss': result['final_ltn_d_loss'],
            'ltn_quality': result['final_ltn_quality'],
            'ltn_logic_satisfaction': result['final_ltn_logic_satisfaction']
        }
        
        print(f"Ring experiment completed in {time.time() - start_time:.2f}s")
        return metrics
    except Exception as e:
        print(f"ERROR: Ring experiment - {e}")
        return {}

def generate_comparison_table(results: Dict[str, Dict[str, float]]) -> str:
    """Generate comprehensive comparison table"""
    
    # Table header
    header_line = f"{'Dataset':<12} {'Model':<15} {'G Loss (Adv/Total)':<20} {'D Loss':<10} {'Quality Score':<15} {'Logic Satisfaction':<18}"
    separator_line = "=" * 95
    
    table_lines = [separator_line, header_line, separator_line]
    
    # Order datasets by complexity: Gaussian -> Grid -> Ring -> MNIST
    dataset_order = ['gaussian', 'grid', 'ring', 'mnist']
    
    for dataset in dataset_order:
        if dataset in results:
            metrics = results[dataset]
            
            # Baseline row
            baseline_g_loss = f"{metrics['baseline_g_loss']:.3f}" if metrics['baseline_g_loss'] is not None else "N/A"
            baseline_d_loss = f"{metrics['baseline_d_loss']:.3f}" if metrics['baseline_d_loss'] is not None else "N/A"
            baseline_quality = f"{metrics['baseline_quality']:.3f}" if metrics['baseline_quality'] is not None else "N/A"
            
            baseline_line = f"{dataset.capitalize():<12} {'Baseline GAN':<15} {baseline_g_loss:<20} {baseline_d_loss:<10} {baseline_quality:<15} {'N/A':<18}"
            table_lines.append(baseline_line)
            
            # LTN-GAN row with detailed G loss breakdown
            ltn_g_adv_loss = f"{metrics['ltn_g_adv_loss']:.3f}" if metrics['ltn_g_adv_loss'] is not None else "N/A"
            ltn_g_total_loss = f"{metrics['ltn_g_loss']:.3f}" if metrics['ltn_g_loss'] is not None else "N/A"
            ltn_g_loss_combined = f"{ltn_g_adv_loss}/{ltn_g_total_loss}"
            
            ltn_d_loss = f"{metrics['ltn_d_loss']:.3f}" if metrics['ltn_d_loss'] is not None else "N/A"
            ltn_quality = f"{metrics['ltn_quality']:.3f}" if metrics['ltn_quality'] is not None else "N/A"
            ltn_logic_satisfaction = f"{metrics['ltn_logic_satisfaction']:.3f}" if metrics['ltn_logic_satisfaction'] is not None else "N/A"
            
            ltn_line = f"{'':<12} {'LTN-GAN':<15} {ltn_g_loss_combined:<20} {ltn_d_loss:<10} {ltn_quality:<15} {ltn_logic_satisfaction:<18}"
            table_lines.append(ltn_line)
            table_lines.append("")  # Empty line for spacing
    
    table_lines.append(separator_line)
    return "\n".join(table_lines)

def main():
    """Run all core LTN-GAN experiments and generate comprehensive comparison"""
    print("=" * 80)
    print("LTN-GAN CORE EXPERIMENTS COMPARISON")
    print("=" * 80)
    
    start_time = time.time()
    results = {}
    
    # Run experiments in order of complexity
    experiments = [
        ('gaussian', run_gaussian_experiment),
        ('grid', run_grid_experiment),
        ('ring', run_ring_experiment),
        ('mnist', run_mnist_experiment)
    ]
    
    for dataset_name, experiment_func in experiments:
        print(f"\n{'='*20} {dataset_name.upper()} {'='*20}")
        metrics = experiment_func()
        if metrics:
            results[dataset_name] = metrics
            print(f"✓ {dataset_name.capitalize()} experiment completed successfully")
        else:
            print(f"✗ {dataset_name.capitalize()} experiment failed")
    
    # Generate comparison table
    if results:
        print(f"\n{'='*20} COMPARISON TABLE {'='*20}")
        comparison_table = generate_comparison_table(results)
        print(comparison_table)
        
        # Save table to file
        filename = "ltn_gan_comparison_table.txt"
        try:
            with open(filename, 'w') as f:
                f.write("LTN-GAN COMPREHENSIVE COMPARISON TABLE\n")
                f.write("=" * 80 + "\n\n")
                f.write(comparison_table)
            print(f"\nComparison table saved to: {filename}")
        except Exception as e:
            print(f"Error saving table: {e}")
    else:
        print("No results generated. All experiments failed.")
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time/60:.1f} minutes")

if __name__ == "__main__":
    main()
