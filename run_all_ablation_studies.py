#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import random
from datetime import datetime

# Set seeds for reproducible results (for any random operations in this script)
random.seed(42)

def run_ablation_study(script_name, description):
    """Run a single ablation study and return success status."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"SCRIPT: {script_name}")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        # Run the ablation study
        result = subprocess.run([sys.executable, script_name],
                              capture_output=True, text=True, timeout=3600)  # 60 min timeout

        if result.returncode == 0:
            print(f"SUCCESS: {description}")
            print(f"Time taken: {time.time() - start_time:.1f} seconds")
            return True
        else:
            print(f"FAILED: {description}")
            print(f"Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {description} (60 minutes)")
        return False
    except Exception as e:
        print(f"ERROR: {description} - {e}")
        return False

def main():
    """Run all ablation studies."""
    print("LTN-GAN ABLATION STUDIES MASTER SCRIPT")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")

    # Define ablation studies
    ablation_studies = [
        ("mnist_ablation_framework.py", "MNIST Ablation Study"),
        ("gaussian_ablation_framework.py", "Gaussian Ablation Study"),
        ("grid_ablation_framework.py", "Grid Ablation Study"),
        ("ring_ablation_framework.py", "Ring Ablation Study")
    ]

    # Track results
    results = []
    total_start_time = time.time()

    print(f"\n PLANNED ABLATION STUDIES:")
    for i, (script, desc) in enumerate(ablation_studies, 1):
        print(f"   {i}. {desc} ({script})")

    print(f"\n STARTING EXECUTION...")
    print(f"Note: Ablation studies may take longer than core experiments")
    print(f"   Each study tests multiple variants and configurations")

    # Run each ablation study
    for script, description in ablation_studies:
        success = run_ablation_study(script, description)
        results.append((script, description, success))

    # Summary
    total_time = time.time() - total_start_time
    successful = sum(1 for _, _, success in results if success)
    failed = len(results) - successful

    print(f"\n{'='*80}")
    print(f" FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total ablation studies: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time/60:.1f} minutes")

    print(f"\n DETAILED RESULTS:")
    for script, description, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"   {status}: {description}")

    if failed == 0:
        print(f"\n ALL ABLATION STUDIES COMPLETED SUCCESSFULLY!")
        print(f" Check the following output directories:")
        print(f"   • mnist_ablation_results/")
        print(f"     - ablation_results.json")
        print(f"     - ablation_table.txt")
        print(f"     - visualizations/ablation_results.png")
        print(f"   • gaussian_ablation_results/")
        print(f"     - gaussian_ablation_results.json")
        print(f"     - gaussian_ablation_table.txt")
        print(f"     - visualizations/gaussian_ablation_results.png")
        print(f"   • grid_ablation_results/")
        print(f"     - grid_ablation_results.json")
        print(f"     - grid_ablation_table.txt")
        print(f"     - visualizations/grid_ablation_results.png")
        print(f"   • ring_ablation_results/")
        print(f"     - ring_ablation_results.json")
        print(f"     - ring_ablation_table.txt")
        print(f"     - visualizations/ring_ablation_results.png")
    else:
        print(f"\n  {failed} ABLATION STUDY(S) FAILED")
        print(f"Please check the error messages above and fix any issues.")

    print(f"\n Master script completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
